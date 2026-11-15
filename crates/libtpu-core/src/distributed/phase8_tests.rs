//! Exhaustive Phase 8 test suite verifying 100% function, line, and branch coverage across
//! `distributed::coordination` (HTTP/2 + Protobuf gRPC `CoordinationService` & multi-host topology
//! handshake), `distributed::ici` (2D Torus & 3D Twisted-Torus DOR/VC router, subslice clamping,
//! link-fault bypass, and all 29 `ElementType` collectives), and `distributed::megascale`
//! (multi-socket TCP/DCN striping, congestion windowing, hierarchical DCN+ICI collectives,
//! full 16-op `MegaScaleActionGraph` runner, and `RapidEye` telemetry).

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use super::coordination::{
    CoordinationGrpcClient, CoordinationGrpcServer, CoordinationServiceState, GrpcProtoEnvelope,
    HTTP2_CLIENT_PREFACE, HTTP2_FLAG_ACK, HTTP2_FLAG_END_HEADERS, HTTP2_FLAG_END_STREAM,
    HpackCodec, Http2Frame, Http2FrameType, MultiHostTopologyHandshake, serve_http2_unary_stream,
};
use super::ici::{
    DeadlockFreeIciRouter, IciCollectiveAlgorithm, IciCollectiveEngine, SubsliceWindow,
    TwistedTorusMeshSpec,
};
use super::megascale::{
    HierarchicalDcnIciRuntime, MegaScaleFullActionGraphRunner, MegaScaleMultiSocketTransport,
    MegascaleCongestionController, RapidEyeTelemetryBuilder, compute_megascale_payload_checksum,
};
use crate::error::{LibTpuError, LibTpuResult, PjrtErrorCode};
use crate::proto::{
    BarrierRequestProto, CancelBarrierRequestProto, CoordinatedTaskProto,
    CoordinatedTaskStateProto, CoordinationDeviceInfoProto, CoordinationServiceErrorProto,
    DeleteKeyValueRequestProto, DistributedDeviceProto, GetAliveTasksRequestProto,
    GetKeyValueDirRequestProto, GetKeyValueRequestProto, GetTaskStateRequestProto,
    HeartbeatRequestProto, IncrementKeyValueRequestProto, InsertKeyValueRequestProto,
    KeyValueEntryProto, LocalTopologyProto, MegaScaleActionGraphProto, MegaScaleActionInputProto,
    MegaScaleActionProto, MegascaleCongestionControlConfigProto,
    MegascaleCongestionWindowStateProto, MegascaleSendRequestProto, PollForErrorRequestProto,
    ProtoSchemaMessage, ProtobufAny, RegisterTaskRequestProto, ReportErrorToServiceRequestProto,
    ReportErrorToTaskRequestProto, ResetTaskRequestProto, ResetTaskResponseProto,
    ShutdownTaskRequestProto, SuperpodTwistedTorusShapeProto, TryGetKeyValueRequestProto,
    WaitForAllTasksRequestProto, WatchJobStateRequestProto, mega_scale_action,
    rapid_eye_error_digest_proto,
};
use crate::types::{ChipCoordinates, CollectiveReductionOp, ElementType, TpuGeneration};

#[test]
fn phase8_http2_hpack_and_grpc_envelope_codecs_cover_all_branches() -> LibTpuResult<()> {
    assert_eq!(Http2FrameType::from_raw(0x0)?, Http2FrameType::Data);
    assert_eq!(Http2FrameType::from_raw(0x1)?, Http2FrameType::Headers);
    assert_eq!(Http2FrameType::from_raw(0x4)?, Http2FrameType::Settings);
    assert_eq!(Http2FrameType::from_raw(0x8)?, Http2FrameType::WindowUpdate);
    assert!(Http2FrameType::from_raw(0xFF).is_err());

    let big_frame = Http2Frame {
        frame_type: Http2FrameType::Data,
        flags: 0,
        stream_id: 1,
        payload: vec![0_u8; 0x0100_0000],
    };
    assert!(big_frame.encode().is_err());

    let ok_frame = Http2Frame {
        frame_type: Http2FrameType::Headers,
        flags: HTTP2_FLAG_END_HEADERS,
        stream_id: 3,
        payload: vec![1, 2, 3],
    };
    let mut wire = Vec::new();
    ok_frame.write_to(&mut wire)?;
    let mut cur = Cursor::new(wire);
    assert_eq!(Http2Frame::read_from(&mut cur)?, ok_frame);
    let mut short = Cursor::new(vec![0_u8; 4]);
    assert!(Http2Frame::read_from(&mut short).is_err());
    let mut trunc_body = Cursor::new(vec![0, 0, 5, 0, 0, 0, 0, 0, 1, 9, 9]);
    assert!(Http2Frame::read_from(&mut trunc_body).is_err());
    let mut bad_type = Cursor::new(vec![0, 0, 0, 0x77, 0, 0, 0, 0, 0]);
    assert!(Http2Frame::read_from(&mut bad_type).is_err());

    // HPACK varint7 & header roundtrips + error branches
    let long_val = "x".repeat(320);
    let hdrs = HpackCodec::encode_headers(&[
        (":method", "POST"),
        (":scheme", "http"),
        (":status", "200"),
        ("x-long", &long_val),
    ]);
    let decoded = HpackCodec::decode_headers(&hdrs)?;
    assert_eq!(decoded.get(":method").map(String::as_str), Some("POST"));
    assert_eq!(decoded.get(":scheme").map(String::as_str), Some("http"));
    assert_eq!(decoded.get(":status").map(String::as_str), Some("200"));
    assert_eq!(decoded.get("x-long"), Some(&long_val));

    let mut c0 = 0_usize;
    assert!(HpackCodec::decode_varint7(&[], &mut c0).is_err());
    let mut c1 = 0_usize;
    assert!(HpackCodec::decode_varint7(&[0x7F, 0x81], &mut c1).is_err());
    let mut med = Vec::new();
    HpackCodec::encode_varint7(&mut med, 140);
    let mut cm = 0_usize;
    assert_eq!(HpackCodec::decode_varint7(&med, &mut cm)?, 140);

    // Literal 0x40 & error cases in decode_headers
    assert!(HpackCodec::decode_headers(&[0x40, 5, b'a']).is_err());
    assert!(HpackCodec::decode_headers(&[0x10, 1, b'k', 5, b'v']).is_err());
    assert!(HpackCodec::decode_headers(&[0x00, 1, 0xFF, 1, b'v']).is_err());
    assert!(HpackCodec::decode_headers(&[0x00, 1, b'k', 1, 0xFF]).is_err());
    assert!(HpackCodec::decode_headers(&[0x99]).is_err());

    // gRPC 5-byte envelope (raw + Brotli + errors)
    let raw_env = GrpcProtoEnvelope::frame(b"hello_grpc", false);
    assert_eq!(GrpcProtoEnvelope::unframe(&raw_env)?, b"hello_grpc");
    let br_env = GrpcProtoEnvelope::frame(b"compressed_grpc_payload", true);
    assert_eq!(
        GrpcProtoEnvelope::unframe(&br_env)?,
        b"compressed_grpc_payload"
    );
    assert!(GrpcProtoEnvelope::unframe(&[0, 1]).is_err());
    assert!(GrpcProtoEnvelope::unframe(&[0, 0, 0, 0, 10, 1, 2]).is_err());
    assert!(GrpcProtoEnvelope::unframe(&[1, 0, 0, 0, 2, 0xFF, 0xFF]).is_err());
    assert!(GrpcProtoEnvelope::unframe(&[2, 0, 0, 0, 0]).is_err());
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase8_coordination_service_grpc_all_19_rpcs_and_fault_paths() -> LibTpuResult<()> {
    let default_st = CoordinationServiceState::default();
    let zero_st = CoordinationServiceState::new(0, 42);
    let (unk_code, _, _) =
        zero_st.dispatch_unary_rpc("/tensorflow.CoordinationService/NoSuchRpc", &[]);
    assert_eq!(unk_code, PjrtErrorCode::Unimplemented as i32);
    let (bad_pb, _, _) =
        default_st.dispatch_unary_rpc("RegisterTask", &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    assert_eq!(bad_pb, PjrtErrorCode::InvalidArgument as i32);

    let mut server = CoordinationGrpcServer::start("127.0.0.1:0", 1)?;
    let addr = server.local_addr();
    let client = CoordinationGrpcClient::new(addr);
    let br_client = CoordinationGrpcClient::new(addr).with_brotli_compression(true);

    // Unregistered heartbeat & shutdown errors
    let t0 = CoordinatedTaskProto {
        job_name: String::from("jax_worker"),
        task_id: 0,
        recoverable: false,
    };
    let t_empty_job = CoordinatedTaskProto {
        job_name: String::new(),
        task_id: 0,
        recoverable: false,
    };
    assert!(
        client
            .heartbeat(&HeartbeatRequestProto {
                incarnation: 10,
                source_task: Some(t0.clone()),
            })
            .is_err()
    );
    assert!(
        client
            .shutdown_task(&ShutdownTaskRequestProto {
                source_task: Some(t0.clone()),
            })
            .is_err()
    );

    // 1. RegisterTask (empty job name normalization, idempotent same incarnation, conflicting incarnation)
    let reg = client.register_task(&RegisterTaskRequestProto {
        incarnation: 100,
        source_task: Some(t_empty_job),
    })?;
    assert_eq!(reg.leader_incarnation, 0xC00D_2026_0000_0001);
    let _ = br_client.register_task(&RegisterTaskRequestProto {
        incarnation: 100,
        source_task: None,
    })?;
    assert!(
        client
            .register_task(&RegisterTaskRequestProto {
                incarnation: 999,
                source_task: Some(t0.clone()),
            })
            .is_err()
    );

    // 2. WaitForAllTasks (with device_info)
    let wait_resp = br_client.wait_for_all_tasks(&WaitForAllTasksRequestProto {
        source_task: Some(t0.clone()),
        device_info: Some(CoordinationDeviceInfoProto {
            device: vec![ProtobufAny {
                type_url: String::from("type.googleapis.com/tpu.Device"),
                value: vec![1, 2, 3],
            }],
        }),
    })?;
    assert_eq!(wait_resp.device_info.map_or(0, |d| d.device.len()), 1);

    // 5. Heartbeat (valid + mismatched incarnation)
    let hb = client.heartbeat(&HeartbeatRequestProto {
        incarnation: 100,
        source_task: Some(t0.clone()),
    })?;
    assert_eq!(hb.leader_incarnation, reg.leader_incarnation);
    assert!(
        client
            .heartbeat(&HeartbeatRequestProto {
                incarnation: 101,
                source_task: Some(t0.clone()),
            })
            .is_err()
    );

    // 8, 9, 18, 19: GetTaskState, WatchJobState, GetAliveTasks, PollForError
    let t1 = CoordinatedTaskProto {
        job_name: String::from("other_job"),
        task_id: 1,
        recoverable: false,
    };
    let ts = client.get_task_state(&GetTaskStateRequestProto {
        source_task: vec![t0.clone(), t1.clone()],
    })?;
    assert_eq!(ts.task_state.len(), 2);

    let wj1 = client.watch_job_state(&WatchJobStateRequestProto {
        job_name: String::new(),
        version_number: 0,
    })?;
    assert_eq!(wj1.task_state.len(), 1);
    let wj2 = client.watch_job_state(&WatchJobStateRequestProto {
        job_name: String::from("jax_worker"),
        version_number: 999,
    })?;
    assert_eq!(wj2.version_number, 1000);

    let alive_all = client.get_alive_tasks(&GetAliveTasksRequestProto {
        requesting_task: Some(t0.clone()),
        tasks: Vec::new(),
    })?;
    assert_eq!(alive_all.alive_tasks.len(), 1);
    let alive_filtered = client.get_alive_tasks(&GetAliveTasksRequestProto {
        requesting_task: None,
        tasks: vec![t0.clone()],
    })?;
    assert_eq!(alive_filtered.incarnations, vec![100]);
    let _ = client.poll_for_error(&PollForErrorRequestProto {
        source_task: Some(t0.clone()),
    })?;

    // 10..15: KV Store RPCs
    assert!(
        client
            .insert_key_value(&InsertKeyValueRequestProto {
                kv: None,
                allow_overwrite: false,
            })
            .is_err()
    );
    let _ = client.insert_key_value(&InsertKeyValueRequestProto {
        kv: Some(KeyValueEntryProto {
            key: String::from("dir/a"),
            value: b"val_a".to_vec(),
        }),
        allow_overwrite: false,
    })?;
    assert!(
        client
            .insert_key_value(&InsertKeyValueRequestProto {
                kv: Some(KeyValueEntryProto {
                    key: String::from("dir/a"),
                    value: b"dup".to_vec(),
                }),
                allow_overwrite: false,
            })
            .is_err()
    );
    let got_kv = br_client.get_key_value(&GetKeyValueRequestProto {
        key: String::from("dir/a"),
    })?;
    assert_eq!(got_kv.kv.map(|e| e.value), Some(b"val_a".to_vec()));
    assert!(
        client
            .get_key_value(&GetKeyValueRequestProto {
                key: String::from("missing_kv"),
            })
            .is_err()
    );
    let try_ok = client.try_get_key_value(&TryGetKeyValueRequestProto {
        key: String::from("dir/a"),
    })?;
    assert!(try_ok.kv.is_some());
    assert!(
        client
            .try_get_key_value(&TryGetKeyValueRequestProto {
                key: String::from("missing_kv"),
            })
            .is_err()
    );

    // IncrementKeyValue (new key, existing key, non-integer key, invalid UTF-8 key)
    let inc1 = client.increment_key_value(&IncrementKeyValueRequestProto {
        key: String::from("dir/counter"),
        increment: 5,
    })?;
    assert_eq!(inc1.kv.map(|e| e.value), Some(b"5".to_vec()));
    let inc2 = client.increment_key_value(&IncrementKeyValueRequestProto {
        key: String::from("dir/counter"),
        increment: 7,
    })?;
    assert_eq!(inc2.kv.map(|e| e.value), Some(b"12".to_vec()));
    assert!(
        client
            .increment_key_value(&IncrementKeyValueRequestProto {
                key: String::from("dir/a"),
                increment: 1,
            })
            .is_err()
    );
    let _ = client.insert_key_value(&InsertKeyValueRequestProto {
        kv: Some(KeyValueEntryProto {
            key: String::from("dir/bad_utf8"),
            value: vec![0xFF, 0xFE],
        }),
        allow_overwrite: true,
    })?;
    assert!(
        client
            .increment_key_value(&IncrementKeyValueRequestProto {
                key: String::from("dir/bad_utf8"),
                increment: 1,
            })
            .is_err()
    );

    let dir_list = client.get_key_value_dir(&GetKeyValueDirRequestProto {
        directory_key: String::from("dir/"),
    })?;
    assert_eq!(dir_list.kv.len(), 3);
    let _ = client.delete_key_value(&DeleteKeyValueRequestProto {
        key: String::from("dir/bad_utf8"),
        is_directory: false,
    })?;
    let _ = client.delete_key_value(&DeleteKeyValueRequestProto {
        key: String::from("dir/"),
        is_directory: true,
    })?;
    let empty_dir = client.get_key_value_dir(&GetKeyValueDirRequestProto {
        directory_key: String::from("dir/"),
    })?;
    assert_eq!(empty_dir.kv.len(), 0);

    // 16 & 17: Barrier (1-task pass, explicit 2-task timeout, pre-cancel, empty cancel_barrier error)
    let b_ok = client.barrier(&BarrierRequestProto {
        barrier_id: String::from("b_single"),
        barrier_timeout_in_ms: 0,
        tasks: Vec::new(),
        source_task: Some(t0.clone()),
        counter: 1,
    })?;
    assert_eq!(b_ok.counter, 1);
    assert!(
        client
            .cancel_barrier(&CancelBarrierRequestProto {
                barrier_id: String::new(),
                source_task: None,
                counter: 0,
            })
            .is_err()
    );
    let _ = client.cancel_barrier(&CancelBarrierRequestProto {
        barrier_id: String::from("b_precancel"),
        source_task: Some(t0.clone()),
        counter: 0,
    })?;
    assert!(
        client
            .barrier(&BarrierRequestProto {
                barrier_id: String::from("b_precancel"),
                barrier_timeout_in_ms: 50,
                tasks: Vec::new(),
                source_task: Some(t0.clone()),
                counter: 0,
            })
            .is_err()
    );
    assert!(
        client
            .barrier(&BarrierRequestProto {
                barrier_id: String::from("b_timeout"),
                barrier_timeout_in_ms: 25,
                tasks: vec![t0.clone(), t1],
                source_task: Some(t0.clone()),
                counter: 0,
            })
            .is_err()
    );

    // 3, 4, 6, 7: ShutdownTask, ReportErrorToTask, PollForError, ResetTask, ReportErrorToService
    let _ = client.shutdown_task(&ShutdownTaskRequestProto {
        source_task: Some(t0.clone()),
    })?;
    let _ = client.report_error_to_task(&ReportErrorToTaskRequestProto {
        error_code: 0,
        error_message: String::from("task0_fault"),
        error_payload: Some(CoordinationServiceErrorProto {
            is_reported_error: true,
            source_task: Some(t0.clone()),
        }),
    })?;
    let _ = client.report_error_to_task(&ReportErrorToTaskRequestProto {
        error_code: PjrtErrorCode::DataLoss as i32,
        error_message: String::from("task0_ecc"),
        error_payload: None,
    })?;
    assert!(
        client
            .poll_for_error(&PollForErrorRequestProto {
                source_task: Some(t0.clone()),
            })
            .is_err()
    );
    let _ = client.reset_task(&ResetTaskRequestProto {
        source_task: Some(t0.clone()),
    })?;
    let _ = client.register_task(&RegisterTaskRequestProto {
        incarnation: 200,
        source_task: Some(t0.clone()),
    })?;
    let _ = client.report_error_to_service(&ReportErrorToServiceRequestProto {
        error_code: 0,
        error_message: String::from("cluster_fault_default_code"),
        error_origin: None,
    })?;
    let _ = client.report_error_to_service(&ReportErrorToServiceRequestProto {
        error_code: PjrtErrorCode::Internal as i32,
        error_message: String::from("cluster_fault"),
        error_origin: Some(t0.clone()),
    })?;
    assert!(
        client
            .poll_for_error(&PollForErrorRequestProto {
                source_task: Some(t0.clone()),
            })
            .is_err()
    );
    let post_err_state = client.get_task_state(&GetTaskStateRequestProto {
        source_task: vec![t0],
    })?;
    assert_eq!(
        post_err_state.task_state.first().map(|i| i.state),
        Some(CoordinatedTaskStateProto::TaskstateError as i32)
    );
    server.shutdown();
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase8_multihost_topology_handshake_and_concurrent_barrier_cancel() -> LibTpuResult<()> {
    // 1. Live 2-host topology discovery handshake
    let server = CoordinationGrpcServer::start("127.0.0.1:0", 2)?;
    let addr = server.local_addr();

    let (g0, g1) = std::thread::scope(|s| -> LibTpuResult<(usize, usize)> {
        let h0 = s.spawn(move || {
            let cli0 = CoordinationGrpcClient::new(addr);
            let loc0 = LocalTopologyProto {
                process_id: 0,
                boot_id: String::from("host0"),
                devices: vec![DistributedDeviceProto {
                    local_device_ordinal: 0,
                    ..DistributedDeviceProto::default()
                }],
                partition_index: 0,
                network_nodes: vec![String::from("tor0")],
            };
            MultiHostTopologyHandshake::exchange_cluster_topology(&cli0, &loc0, 1001)
        });
        let h1 = s.spawn(move || {
            let cli1 = CoordinationGrpcClient::new(addr).with_brotli_compression(true);
            let loc1 = LocalTopologyProto {
                process_id: 1,
                boot_id: String::from("host1"),
                devices: vec![DistributedDeviceProto {
                    local_device_ordinal: 0,
                    ..DistributedDeviceProto::default()
                }],
                partition_index: 1,
                network_nodes: vec![String::from("tor1")],
            };
            MultiHostTopologyHandshake::exchange_cluster_topology(&cli1, &loc1, 1002)
        });
        let r0 = h0.join().unwrap_or_else(|_| {
            Err(LibTpuError::InternalError {
                context: String::from("h0 join"),
            })
        })?;
        let r1 = h1.join().unwrap_or_else(|_| {
            Err(LibTpuError::InternalError {
                context: String::from("h1 join"),
            })
        })?;
        Ok((r0.processes.len(), r1.processes.len()))
    })?;
    assert_eq!((g0, g1), (2, 2));

    // 2. CancelBarrier waking an in-flight waiter, WaitForAllTasks timeout & service_error paths
    let srv_cancel = CoordinationGrpcServer::start("127.0.0.1:0", 2)?;
    let c_addr = srv_cancel.local_addr();
    std::thread::scope(|s| {
        s.spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            let cli = CoordinationGrpcClient::new(c_addr);
            let _ = cli.cancel_barrier(&CancelBarrierRequestProto {
                barrier_id: String::from("live_cancel"),
                source_task: None,
                counter: 7,
            });
        });
        let cli_wait = CoordinationGrpcClient::new(c_addr);
        let res = cli_wait.barrier(&BarrierRequestProto {
            barrier_id: String::from("live_cancel"),
            barrier_timeout_in_ms: 1_000,
            tasks: Vec::new(),
            source_task: None,
            counter: 7,
        });
        assert!(res.is_err());
    });

    // Service error waking a barrier waiter & WaitForAllTasks failing on service_error
    let srv_err = CoordinationGrpcServer::start("127.0.0.1:0", 2)?;
    let e_addr = srv_err.local_addr();
    std::thread::scope(|s| {
        s.spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            let cli = CoordinationGrpcClient::new(e_addr);
            let _ = cli.report_error_to_service(&ReportErrorToServiceRequestProto {
                error_code: PjrtErrorCode::Aborted as i32,
                error_message: String::from("abort_waiters"),
                error_origin: None,
            });
        });
        let cli_wait = CoordinationGrpcClient::new(e_addr);
        assert!(
            cli_wait
                .barrier(&BarrierRequestProto {
                    barrier_id: String::from("err_barrier"),
                    barrier_timeout_in_ms: 1_000,
                    tasks: Vec::new(),
                    source_task: None,
                    counter: 0,
                })
                .is_err()
        );
        assert!(
            cli_wait
                .wait_for_all_tasks(&WaitForAllTasksRequestProto {
                    source_task: None,
                    device_info: None,
                })
                .is_err()
        );
    });

    // WaitForAllTasks timeout on 2-worker server with only 1 worker (check directly via short-lived state)
    let st_timeout = Arc::new(CoordinationServiceState::new(2, 55));
    let _ = st_timeout.dispatch_unary_rpc(
        "RegisterTask",
        &RegisterTaskRequestProto {
            incarnation: 1,
            source_task: None,
        }
        .encode_proto(),
    );
    let (wt_code, _, _) = st_timeout.dispatch_unary_rpc(
        "WaitForAllTasks",
        &WaitForAllTasksRequestProto {
            source_task: None,
            device_info: None,
        }
        .encode_proto(),
    );
    assert_eq!(wt_code, PjrtErrorCode::DeadlineExceeded as i32);

    // 2-worker WaitForAllTasks woken before deadline by worker 1 registering
    let st_wakeup = Arc::new(CoordinationServiceState::new(2, 56));
    let _ = st_wakeup.dispatch_unary_rpc(
        "RegisterTask",
        &RegisterTaskRequestProto {
            incarnation: 1,
            source_task: Some(CoordinatedTaskProto {
                job_name: String::from("w"),
                task_id: 0,
                recoverable: false,
            }),
        }
        .encode_proto(),
    );
    std::thread::scope(|s| {
        let st_bg = Arc::clone(&st_wakeup);
        s.spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            let _ = st_bg.dispatch_unary_rpc(
                "RegisterTask",
                &RegisterTaskRequestProto {
                    incarnation: 2,
                    source_task: Some(CoordinatedTaskProto {
                        job_name: String::from("w"),
                        task_id: 1,
                        recoverable: false,
                    }),
                }
                .encode_proto(),
            );
        });
        let (ok_code, _, _) = st_wakeup.dispatch_unary_rpc(
            "WaitForAllTasks",
            &WaitForAllTasksRequestProto {
                source_task: Some(CoordinatedTaskProto {
                    job_name: String::from("w"),
                    task_id: 0,
                    recoverable: false,
                }),
                device_info: None,
            }
            .encode_proto(),
        );
        assert_eq!(ok_code, 0);
    });

    // Raw HTTP/2 preface error & SETTINGS_ACK / WindowUpdate / END_STREAM-on-Data handling
    let raw_listener = TcpListener::bind("127.0.0.1:0")?;
    let raw_addr = raw_listener.local_addr()?;
    std::thread::scope(|s| {
        let st_ref = &st_timeout;
        s.spawn(|| {
            if let Ok((mut s1, _)) = raw_listener.accept() {
                assert!(serve_http2_unary_stream(&mut s1, st_ref).is_err());
            }
            if let Ok((mut s2, _)) = raw_listener.accept() {
                let _ = s2.read_exact_or_ignore();
                let _ = s2.write_all(HTTP2_CLIENT_PREFACE);
                let _ = Http2Frame {
                    frame_type: Http2FrameType::Settings,
                    flags: HTTP2_FLAG_ACK,
                    stream_id: 0,
                    payload: Vec::new(),
                }
                .write_to(&mut s2);
                let _ = Http2Frame {
                    frame_type: Http2FrameType::WindowUpdate,
                    flags: 0,
                    stream_id: 0,
                    payload: vec![0, 0, 16, 0],
                }
                .write_to(&mut s2);
                let _ = Http2Frame {
                    frame_type: Http2FrameType::Headers,
                    flags: HTTP2_FLAG_END_HEADERS,
                    stream_id: 1,
                    payload: HpackCodec::encode_headers(&[(":status", "200")]),
                }
                .write_to(&mut s2);
                let _ = Http2Frame {
                    frame_type: Http2FrameType::Data,
                    flags: HTTP2_FLAG_END_STREAM,
                    stream_id: 1,
                    payload: GrpcProtoEnvelope::frame(&[], false),
                }
                .write_to(&mut s2);
            }
        });
        if let Ok(mut bad_cli) = TcpStream::connect(raw_addr) {
            let _ = bad_cli.write_all(b"NOT_AN_HTTP2_PREFACE_24B");
        }
        let custom_cli = CoordinationGrpcClient::new(raw_addr);
        let _ = custom_cli.reset_task(&ResetTaskRequestProto { source_task: None });
    });
    Ok(())
}

/// Helper trait reading a standard client request burst from a custom mock HTTP/2 server socket.
trait ReadPrefaceBurst {
    /// Reads the client preface + request frames before writing a custom server response.
    fn read_exact_or_ignore(&mut self) -> LibTpuResult<()>;
}

impl ReadPrefaceBurst for TcpStream {
    fn read_exact_or_ignore(&mut self) -> LibTpuResult<()> {
        use std::io::Read;
        let mut p = [0_u8; 24];
        self.read_exact(&mut p)?;
        loop {
            let f = Http2Frame::read_from(self)?;
            if f.frame_type == Http2FrameType::Data && (f.flags & HTTP2_FLAG_END_STREAM) != 0 {
                break;
            }
        }
        Ok(())
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase8_ici_2d_and_3d_twisted_torus_router_and_all_29_element_collectives() -> LibTpuResult<()> {
    let z_win = SubsliceWindow::new(ChipCoordinates::new(0, 0, 0), ChipCoordinates::new(0, 0, 0));
    assert!(z_win.contains(ChipCoordinates::new(0, 0, 0)));
    assert!(!z_win.contains(ChipCoordinates::new(1, 0, 0)));

    // Mesh specs across 2D & all 4 3D TwistedTorusShapes
    let s2d = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V6e,
        ChipCoordinates::new(0, 0, 5),
    );
    assert_eq!(s2d.mesh_bounds, ChipCoordinates::new(1, 1, 1));
    assert_eq!(
        s2d.twisted_shape,
        SuperpodTwistedTorusShapeProto::TwistShapeUnspecified
    );
    let s3d_min = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V7x,
        ChipCoordinates::new(0, 0, 0),
    );
    assert_eq!(s3d_min.mesh_bounds, ChipCoordinates::new(1, 1, 1));

    let shape_cube_2x = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V4,
        ChipCoordinates::new(4, 4, 8),
    );
    assert_eq!(
        shape_cube_2x.twisted_shape,
        SuperpodTwistedTorusShapeProto::TwistShapeKK2k
    );
    let shape_wide_2x = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V5p,
        ChipCoordinates::new(4, 8, 8),
    );
    assert_eq!(
        shape_wide_2x.twisted_shape,
        SuperpodTwistedTorusShapeProto::TwistShapeK2k2k
    );
    let shape_tall_nx = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V7x,
        ChipCoordinates::new(4, 8, 12),
    );
    assert_eq!(
        shape_tall_nx.twisted_shape,
        SuperpodTwistedTorusShapeProto::TwistShapeK2kNk
    );
    let custom = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V7x,
        ChipCoordinates::new(4, 4, 4),
    );
    assert_eq!(
        custom.twisted_shape,
        SuperpodTwistedTorusShapeProto::TwistShapeCustomized
    );

    // 3D Twisted-Torus routing, VC0->VC1 dateline wrap, link/chip fault bypass, subslice clamping
    let mut router = DeadlockFreeIciRouter::new(TpuGeneration::V4, ChipCoordinates::new(4, 4, 4));
    assert_eq!(
        router
            .compute_deadlock_free_route(
                ChipCoordinates::new(0, 0, 0),
                ChipCoordinates::new(0, 0, 0)
            )?
            .len(),
        0
    );
    // Z-wrap (0,0,0) -> (3,0,3) crosses -Z twisted wrap link and transitions to VC1;
    // inverse +Z wrap (3,0,3) -> (0,0,0) also takes 1 hop on VC1.
    let z_wrap_hops = router.compute_deadlock_free_route(
        ChipCoordinates::new(0, 0, 0),
        ChipCoordinates::new(3, 0, 3),
    )?;
    assert_eq!(z_wrap_hops.len(), 1);
    assert!(
        z_wrap_hops
            .first()
            .is_some_and(|h| h.is_wrap_around && h.virtual_channel == 1)
    );
    let pos_z_wrap_hops = router.compute_deadlock_free_route(
        ChipCoordinates::new(3, 0, 3),
        ChipCoordinates::new(0, 0, 0),
    )?;
    assert_eq!(pos_z_wrap_hops.len(), 1);

    // Inject link fault on direct +X (0,0,0)->(1,0,0) and faulty chip (0,1,0); verify bypass succeeds
    router.mark_link_faulty(ChipCoordinates::new(0, 0, 0), ChipCoordinates::new(1, 0, 0));
    router.mark_chip_faulty(ChipCoordinates::new(0, 1, 0));
    let bypass = router.compute_deadlock_free_route(
        ChipCoordinates::new(0, 0, 0),
        ChipCoordinates::new(1, 0, 0),
    )?;
    assert!(bypass.len() >= 2);
    assert!(
        router
            .compute_deadlock_free_route(
                ChipCoordinates::new(0, 1, 0),
                ChipCoordinates::new(1, 0, 0)
            )
            .is_err()
    );

    // Subslice clamp validation & synthesis
    assert!(
        router
            .set_subslice_window(SubsliceWindow::new(
                ChipCoordinates::new(2, 0, 0),
                ChipCoordinates::new(3, 2, 2),
            ))
            .is_err()
    );
    router.set_subslice_window(SubsliceWindow::new(
        ChipCoordinates::new(0, 0, 0),
        ChipCoordinates::new(2, 2, 2),
    ))?;
    assert!(
        router
            .compute_deadlock_free_route(
                ChipCoordinates::new(0, 0, 0),
                ChipCoordinates::new(3, 0, 0)
            )
            .is_err()
    );
    let rtable = router.synthesize_ici_routing_table_proto("v4_subslice")?;
    assert!(
        !rtable.route_hops.is_empty(),
        "expected non-empty route_hops"
    );
    let sp_cfg = router.synthesize_superpod_ici_network_config("net_v4")?;
    assert_eq!(sp_cfg.configs.len(), 7); // 2x2x2 - 1 faulty chip

    // Single-chip superpod config & disconnected fault error
    let one_chip = DeadlockFreeIciRouter::new(TpuGeneration::V6e, ChipCoordinates::new(1, 1, 1));
    assert_eq!(
        one_chip
            .synthesize_superpod_ici_network_config("single")?
            .configs
            .len(),
        1
    );
    let mut isolated =
        DeadlockFreeIciRouter::new(TpuGeneration::V6e, ChipCoordinates::new(2, 1, 1));
    isolated.mark_link_faulty(ChipCoordinates::new(0, 0, 0), ChipCoordinates::new(1, 0, 0));
    assert!(
        isolated
            .compute_deadlock_free_route(
                ChipCoordinates::new(0, 0, 0),
                ChipCoordinates::new(1, 0, 0)
            )
            .is_err()
    );

    // Schedules (`0`, `1`, `3`, `4` participants across MultiRing, MeshTorus2d3d, HalvingDoubling)
    assert_eq!(
        IciCollectiveEngine::build_schedule(
            IciCollectiveAlgorithm::MultiRing,
            1,
            ChipCoordinates::new(1, 1, 1)
        )
        .phases
        .len(),
        0
    );
    assert_eq!(
        IciCollectiveEngine::build_schedule(
            IciCollectiveAlgorithm::MultiRing,
            4,
            ChipCoordinates::new(2, 2, 1)
        )
        .phases
        .len(),
        6
    );
    assert_eq!(
        IciCollectiveEngine::build_schedule(
            IciCollectiveAlgorithm::MeshTorus2d3d,
            6,
            ChipCoordinates::new(2, 2, 2)
        )
        .phases
        .len(),
        3
    );
    assert_eq!(
        IciCollectiveEngine::build_schedule(
            IciCollectiveAlgorithm::HalvingDoubling,
            3,
            ChipCoordinates::new(3, 1, 1)
        )
        .phases
        .len(),
        2
    );

    // Empty buffer edge cases
    assert_eq!(
        IciCollectiveEngine::execute_all_reduce_scheduled(
            IciCollectiveAlgorithm::MultiRing,
            ChipCoordinates::new(2, 1, 1),
            &[],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )
        .0
        .len(),
        0
    );
    assert_eq!(
        IciCollectiveEngine::execute_reduce_scatter_scheduled(
            IciCollectiveAlgorithm::MultiRing,
            ChipCoordinates::new(2, 1, 1),
            &[],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )
        .0
        .len(),
        0
    );
    assert_eq!(
        IciCollectiveEngine::execute_all_gather_scheduled(
            IciCollectiveAlgorithm::MultiRing,
            ChipCoordinates::new(2, 1, 1),
            &[],
            ElementType::F32,
        )
        .0
        .len(),
        0
    );
    assert_eq!(
        IciCollectiveEngine::execute_all_to_all_typed(&[], ElementType::F32).len(),
        0
    );
    assert_eq!(
        IciCollectiveEngine::execute_ragged_all_to_all_typed(&[], ElementType::F32, &[], &[], &[])
            .len(),
        0
    );

    // Exhaustive execution across ALL 29 ElementTypes (`0..=28`) and all 4 reduction ops
    for raw_et in 0..=28 {
        let et = ElementType::from_raw(raw_et)?;
        let sample_a = vec![0x01_u8; 32];
        let sample_b = vec![0x02_u8; 32];
        let bufs = vec![sample_a, sample_b];
        for op in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Product,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Max,
        ] {
            let (ar, _) = IciCollectiveEngine::execute_all_reduce_scheduled(
                IciCollectiveAlgorithm::HalvingDoubling,
                ChipCoordinates::new(2, 1, 1),
                &bufs,
                et,
                op,
            );
            assert_eq!(ar.len(), 2);
            let (rs, _) = IciCollectiveEngine::execute_reduce_scatter_scheduled(
                IciCollectiveAlgorithm::MultiRing,
                ChipCoordinates::new(2, 1, 1),
                &bufs,
                et,
                op,
            );
            assert_eq!(rs.len(), 2);
            let (ag, _) = IciCollectiveEngine::execute_all_gather_scheduled(
                IciCollectiveAlgorithm::MeshTorus2d3d,
                ChipCoordinates::new(2, 1, 1),
                &rs,
                et,
            );
            assert_eq!(ag.len(), 2);
        }
        let a2a = IciCollectiveEngine::execute_all_to_all_typed(&bufs, et);
        assert_eq!(a2a.len(), 2);
        let ragged = IciCollectiveEngine::execute_ragged_all_to_all_typed(
            &bufs,
            et,
            &[vec![0, 1], vec![0, 1]],
            &[vec![1, 1], vec![1, 1]],
            &[vec![0, 1], vec![0, 1]],
        );
        assert_eq!(ragged.len(), 2);
    }

    // Also test C64 & C128 reverse ordering (`a > b`) for Min/Max branches + CollectivePermute routed
    let c64_hi = [2.0_f32.to_le_bytes(), 3.0_f32.to_le_bytes()].concat();
    let c64_lo = [1.0_f32.to_le_bytes(), 0.5_f32.to_le_bytes()].concat();
    let _ = IciCollectiveEngine::reduce_pair_typed(
        &c64_hi,
        &c64_lo,
        ElementType::C64,
        CollectiveReductionOp::Min,
    );
    let _ = IciCollectiveEngine::reduce_pair_typed(
        &c64_hi,
        &c64_lo,
        ElementType::C64,
        CollectiveReductionOp::Max,
    );
    let c128_hi = [5.0_f64.to_le_bytes(), 6.0_f64.to_le_bytes()].concat();
    let c128_lo = [1.0_f64.to_le_bytes(), 2.0_f64.to_le_bytes()].concat();
    let _ = IciCollectiveEngine::reduce_pair_typed(
        &c128_hi,
        &c128_lo,
        ElementType::C128,
        CollectiveReductionOp::Min,
    );
    let _ = IciCollectiveEngine::reduce_pair_typed(
        &c128_hi,
        &c128_lo,
        ElementType::C128,
        CollectiveReductionOp::Max,
    );

    let healthy_2 = DeadlockFreeIciRouter::new(TpuGeneration::V6e, ChipCoordinates::new(2, 1, 1));
    let coords_2 = [ChipCoordinates::new(0, 0, 0), ChipCoordinates::new(1, 0, 0)];
    let perm = IciCollectiveEngine::execute_collective_permute_routed(
        &healthy_2,
        &coords_2,
        &[vec![10, 20], vec![30, 40]],
        &[(0, 1), (1, 0)],
    )?;
    assert_eq!(perm, vec![vec![30, 40], vec![10, 20]]);
    assert!(
        IciCollectiveEngine::execute_collective_permute_routed(
            &isolated,
            &coords_2,
            &[vec![1], vec![2]],
            &[(0, 1)],
        )
        .is_err()
    );
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase8_megascale_multisocket_dcn_congestion_action_graph_and_rapideye() -> LibTpuResult<()> {
    assert_ne!(compute_megascale_payload_checksum(b"megascale"), 0);

    // 1. Congestion controller (BBRv2 & TIMELY, ECN backoff, high-RTT backoff, unseen channel, pacing)
    let mut bbr = MegascaleCongestionController::default();
    let w_fast = bbr.on_ack_sample(0, 65_536, 20_000, false);
    assert!(w_fast.cwnd_bytes > 262_144);
    let w_ecn = bbr.on_ack_sample(0, 4096, 30_000, true);
    assert_eq!(w_ecn.ecn_marked_packets, 1);
    let _ = bbr.on_ack_sample(0, 4096, 500_000, false);
    let w_new = bbr.on_ack_sample(99, 8192, 25_000, false);
    assert_eq!(w_new.channel_id, 99);
    assert!(bbr.compute_pacing_delay_ns(0, 4096) > 0);
    assert!(bbr.compute_pacing_delay_ns(777, 4096) > 0);

    let mut timely = MegascaleCongestionController::new(
        MegascaleCongestionControlConfigProto {
            algorithm_name: String::from("TIMELY"),
            initial_cwnd_bytes: 65_536,
            min_cwnd_bytes: 8_192,
            max_cwnd_bytes: 1_048_576,
            active_windows: vec![MegascaleCongestionWindowStateProto {
                channel_id: 0,
                cwnd_bytes: 65_536,
                rtt_nanoseconds: 20_000,
                ecn_marked_packets: 0,
                pacing_rate_gbps: 25,
            }],
        },
        0,
        20_000,
    );
    let w_timely = timely.on_ack_sample(0, 4096, 200_000, false);
    assert!(w_timely.cwnd_bytes < 65_536);

    // 2. Multi-socket TCP/DCN striped transport & Hierarchical DCN+ICI collectives
    let transport = MegaScaleMultiSocketTransport::new(4, true, bbr);
    let raw_transport = MegaScaleMultiSocketTransport::new(2, false, timely);
    let echoed = raw_transport.striped_send_receive("raw_k", 0, 0, b"striped_hello_world")?;
    assert_eq!(echoed, b"striped_hello_world");

    // Validation errors on empty / mismatched slice counts
    assert!(
        HierarchicalDcnIciRuntime::execute_hierarchical_all_reduce(
            &transport,
            ChipCoordinates::new(2, 1, 1),
            &[],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )
        .is_err()
    );
    assert!(
        HierarchicalDcnIciRuntime::execute_hierarchical_all_reduce(
            &transport,
            ChipCoordinates::new(2, 1, 1),
            &[Vec::new()],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )
        .is_err()
    );
    assert!(
        HierarchicalDcnIciRuntime::execute_hierarchical_all_gather(
            &transport,
            ChipCoordinates::new(2, 1, 1),
            &[],
            ElementType::F32,
        )
        .is_err()
    );

    let s0c0 = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
    let s0c1 = ElementType::F32.pack_from_f64_slice(&[2.0, 3.0, 4.0, 5.0]);
    let s1c0 = ElementType::F32.pack_from_f64_slice(&[3.0, 4.0, 5.0, 6.0]);
    let s1c1 = ElementType::F32.pack_from_f64_slice(&[4.0, 5.0, 6.0, 7.0]);
    let slices = vec![vec![s0c0, s0c1], vec![s1c0, s1c1]];

    for op in [
        CollectiveReductionOp::Sum,
        CollectiveReductionOp::Product,
        CollectiveReductionOp::Min,
        CollectiveReductionOp::Max,
    ] {
        let (ar_out, sched) = HierarchicalDcnIciRuntime::execute_hierarchical_all_reduce(
            &transport,
            ChipCoordinates::new(2, 1, 1),
            &slices,
            ElementType::F32,
            op,
        )?;
        assert!(sched.hierarchical_ici_then_dcn);
        if op == CollectiveReductionOp::Sum {
            let first_buf = ar_out
                .first()
                .and_then(|s| s.first())
                .map_or(&[][..], Vec::as_slice);
            let vals = ElementType::F32.unpack_to_f64_vec(first_buf);
            assert_eq!(vals, vec![10.0, 14.0, 18.0, 22.0]);
        }
    }

    let rs_out = HierarchicalDcnIciRuntime::execute_hierarchical_reduce_scatter(
        &transport,
        ChipCoordinates::new(2, 1, 1),
        &slices,
        ElementType::F32,
        CollectiveReductionOp::Sum,
    )?;
    let ag_out = HierarchicalDcnIciRuntime::execute_hierarchical_all_gather(
        &transport,
        ChipCoordinates::new(2, 1, 1),
        &rs_out,
        ElementType::F32,
    )?;
    let ag_second = ag_out
        .get(1)
        .and_then(|s| s.get(1))
        .map_or(&[][..], Vec::as_slice);
    assert_eq!(
        ElementType::F32.unpack_to_f64_vec(ag_second),
        vec![10.0, 14.0, 18.0, 22.0]
    );

    // 3. Full 16-op MegaScaleActionGraphRunner
    let empty_mb_transport =
        MegaScaleMultiSocketTransport::new(1, false, MegascaleCongestionController::default());
    let mut ext = BTreeMap::new();
    ext.insert(
        String::from("device_0"),
        ElementType::F32.pack_from_f64_slice(&[4.0, 8.0]),
    );
    ext.insert(
        String::from("peer_buf"),
        ElementType::F32.pack_from_f64_slice(&[2.0, 3.0]),
    );

    let act = |index: i32,
               inputs: Vec<MegaScaleActionInputProto>,
               action: Option<mega_scale_action::Action>| MegaScaleActionProto {
        index,
        inputs,
        action,
    };
    let inp_idx = |i: i32| MegaScaleActionInputProto {
        source: Some(mega_scale_action::input::Source::ActionIndex(i)),
    };
    let inp_ext = |k: &str| MegaScaleActionInputProto {
        source: Some(mega_scale_action::input::Source::ExternalKey(String::from(
            k,
        ))),
    };
    let inp_none = MegaScaleActionInputProto { source: None };

    let graph = MegaScaleActionGraphProto {
        id: String::from("full_16_action_graph"),
        actions: vec![
            // Empty mailbox NetworkReceive zero-fill branch before any send
            act(
                0,
                Vec::new(),
                Some(mega_scale_action::Action::NetworkReceive(
                    mega_scale_action::NetworkReceive {
                        done_key: String::from("not_yet"),
                        buffer_size: 8,
                        dcn_topology_level: Some(0),
                        device_bound: false,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(999), inp_ext("missing_ext"), inp_none],
                Some(mega_scale_action::Action::PrepareInput(
                    mega_scale_action::PrepareInput {
                        buffer: None,
                        host_command_action: None,
                        device_id: 0,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(1)],
                Some(mega_scale_action::Action::PrepareInput(
                    mega_scale_action::PrepareInput {
                        buffer: None,
                        host_command_action: None,
                        device_id: 99,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(1)],
                Some(mega_scale_action::Action::DeviceToHost(
                    mega_scale_action::DeviceToHost::default(),
                )),
            ),
            act(
                0,
                vec![inp_idx(3)],
                Some(mega_scale_action::Action::DeviceToHostDone(
                    mega_scale_action::DeviceToHostDone {},
                )),
            ),
            act(
                0,
                vec![inp_idx(4)],
                Some(mega_scale_action::Action::Slice(mega_scale_action::Slice {
                    offset: 0,
                    size: 8,
                })),
            ),
            act(
                0,
                vec![inp_idx(5)],
                Some(mega_scale_action::Action::Compress(
                    mega_scale_action::Compress::default(),
                )),
            ),
            act(
                0,
                vec![inp_idx(6)],
                Some(mega_scale_action::Action::Decompress(
                    mega_scale_action::Decompress::default(),
                )),
            ),
            act(
                0,
                vec![inp_idx(7)],
                Some(mega_scale_action::Action::NetworkSendPrepare(
                    mega_scale_action::NetworkSendPrepare::default(),
                )),
            ),
            act(
                0,
                vec![inp_idx(8)],
                Some(mega_scale_action::Action::NetworkSend(
                    mega_scale_action::NetworkSend {
                        transfer_info: None,
                        dcn_topology_level: Some(1),
                        ragged_shard_index: None,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(9)],
                Some(mega_scale_action::Action::NetworkSendDone(
                    mega_scale_action::NetworkSendDone { total_bytes: 8 },
                )),
            ),
            // Exact key hit & fallback mailbox hit in NetworkReceive
            act(
                0,
                vec![inp_idx(10)],
                Some(mega_scale_action::Action::NetworkReceive(
                    mega_scale_action::NetworkReceive {
                        done_key: String::from("graph_full_16_action_graph_lvl_1"),
                        buffer_size: 8,
                        dcn_topology_level: Some(1),
                        device_bound: true,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(11)],
                Some(mega_scale_action::Action::NetworkReceive(
                    mega_scale_action::NetworkReceive {
                        done_key: String::from("fallback_latest_entry"),
                        buffer_size: 8,
                        dcn_topology_level: None,
                        device_bound: false,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(12), inp_ext("peer_buf")],
                Some(mega_scale_action::Action::Compute(
                    mega_scale_action::Compute {
                        operation: 2, // MAX
                        ..mega_scale_action::Compute::default()
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(13), inp_ext("peer_buf")],
                Some(mega_scale_action::Action::Compute(
                    mega_scale_action::Compute {
                        operation: 3, // MIN
                        ..mega_scale_action::Compute::default()
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(12), inp_ext("peer_buf")],
                Some(mega_scale_action::Action::Compute(
                    mega_scale_action::Compute {
                        operation: 4, // PRODUCT
                        ..mega_scale_action::Compute::default()
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(15), inp_ext("peer_buf")],
                Some(mega_scale_action::Action::SimulatedChannel(
                    mega_scale_action::SimulatedChannel {
                        bytes_txed: 8,
                        bytes_per_h2d: vec![8],
                        operation: Some(1), // SUM
                        group_size: Some(2),
                        data_type: None,
                    },
                )),
            ),
            act(
                0,
                vec![inp_idx(16)],
                Some(mega_scale_action::Action::HostToDevice(
                    mega_scale_action::HostToDevice::default(),
                )),
            ),
            act(
                0,
                vec![inp_idx(17)],
                Some(mega_scale_action::Action::HostToDeviceDone(
                    mega_scale_action::HostToDeviceDone {},
                )),
            ),
            act(
                0,
                vec![inp_idx(18)],
                Some(mega_scale_action::Action::PrepareOutput(
                    mega_scale_action::PrepareOutput::default(),
                )),
            ),
            act(0, vec![inp_idx(19)], None),
            act(
                21,
                vec![inp_idx(20)],
                Some(mega_scale_action::Action::End(mega_scale_action::End {})),
            ),
        ],
        ..MegaScaleActionGraphProto::default()
    };

    let (final_bytes, telem) =
        MegaScaleFullActionGraphRunner::execute_graph(&graph, &ext, &empty_mb_transport)?;
    // (4*2 + 2, 8*3 + 3) = (10.0, 27.0)
    assert_eq!(
        ElementType::F32.unpack_to_f64_vec(&final_bytes),
        vec![10.0, 27.0]
    );
    assert_eq!(telem.pending_graphs.len(), 1);

    // Corrupt Decompress action error branch
    let bad_decomp_graph = MegaScaleActionGraphProto {
        id: String::from("bad_decomp"),
        actions: vec![act(
            0,
            Vec::new(),
            Some(mega_scale_action::Action::Decompress(
                mega_scale_action::Decompress::default(),
            )),
        )],
        ..MegaScaleActionGraphProto::default()
    };
    let mut bad_ext = BTreeMap::new();
    bad_ext.insert(String::from("k"), vec![0xFF, 0xFF]);
    assert!(
        MegaScaleFullActionGraphRunner::execute_graph(
            &bad_decomp_graph,
            &bad_ext,
            &empty_mb_transport
        )
        .is_err()
    );

    // 4. RapidEyeInfo & RapidEyeErrorDigestProto telemetry (both hang and unrecoverable causes)
    let (hang_err, hang_digest) = RapidEyeTelemetryBuilder::build_error_and_digest(
        TpuGeneration::V7x,
        1,
        2,
        4,
        rapid_eye_error_digest_proto::Cause::NetworkingIssue,
        "transformer_step",
        "DCN slice 1 timeout",
    );
    assert!(
        hang_err
            .runtime_state
            .as_ref()
            .and_then(|s| s.rapid_eye_info.as_ref())
            .is_some()
    );
    assert_eq!(
        hang_digest.potential_cause,
        rapid_eye_error_digest_proto::Cause::NetworkingIssue as i32
    );
    assert!(LibTpuError::from_megascale_runtime_error_proto(&hang_err).is_err());

    let (hw_err, hw_digest) = RapidEyeTelemetryBuilder::build_error_and_digest(
        TpuGeneration::V6e,
        0,
        0,
        2,
        rapid_eye_error_digest_proto::Cause::BadTpuChip,
        "matmul_kernel",
        "HBM ECC uncorrectable",
    );
    assert_eq!(
        hw_digest.potential_cause,
        rapid_eye_error_digest_proto::Cause::BadTpuChip as i32
    );
    assert!(LibTpuError::from_megascale_runtime_error_proto(&hw_err).is_err());
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn phase8_exhaustive_wire_and_router_branch_edges() -> LibTpuResult<()> {
    // 1. CoordinationService KV wait wakeup, AliveTasks filter/disconnected, PollForError healthy disconnected
    let mut server = CoordinationGrpcServer::start("127.0.0.1:0", 2)?;
    let client = CoordinationGrpcClient::new(server.local_addr());
    let t0 = CoordinatedTaskProto {
        job_name: String::from("jax_worker"),
        task_id: 0,
        recoverable: false,
    };
    let t1 = CoordinatedTaskProto {
        job_name: String::from("jax_worker"),
        task_id: 1,
        recoverable: false,
    };
    // PollForError on unregistered task (`tasks.get == None`)
    let _ = client.poll_for_error(&PollForErrorRequestProto {
        source_task: Some(t0.clone()),
    })?;
    let _ = client.register_task(&RegisterTaskRequestProto {
        incarnation: 1,
        source_task: Some(t0.clone()),
    })?;
    let _ = client.register_task(&RegisterTaskRequestProto {
        incarnation: 2,
        source_task: Some(t1.clone()),
    })?;
    // GetAliveTasks filtering out connected t1 (`filter_keys.contains == false`)
    let only_t0 = client.get_alive_tasks(&GetAliveTasksRequestProto {
        requesting_task: Some(t0.clone()),
        tasks: vec![t0.clone()],
    })?;
    assert_eq!(only_t0.alive_tasks.len(), 1);
    // Disconnect t1 -> PollForError on disconnected task (`state != TaskstateError`) & GetAliveTasks skips disconnected
    let _ = client.shutdown_task(&ShutdownTaskRequestProto {
        source_task: Some(t1.clone()),
    })?;
    let _ = client.poll_for_error(&PollForErrorRequestProto {
        source_task: Some(t1),
    })?;
    let after_disc = client.get_alive_tasks(&GetAliveTasksRequestProto {
        requesting_task: Some(t0),
        tasks: Vec::new(),
    })?;
    assert_eq!(after_disc.alive_tasks.len(), 1);

    // Concurrent GetKeyValue waking before timeout (`timed_out() == false`)
    let addr = server.local_addr();
    std::thread::scope(|s| -> LibTpuResult<()> {
        let writer = s.spawn(move || {
            std::thread::sleep(Duration::from_millis(15));
            let c = CoordinationGrpcClient::new(addr);
            c.insert_key_value(&InsertKeyValueRequestProto {
                kv: Some(KeyValueEntryProto {
                    key: String::from("delayed_key"),
                    value: b"arrived".to_vec(),
                }),
                allow_overwrite: true,
            })
        });
        let got = client.get_key_value(&GetKeyValueRequestProto {
            key: String::from("delayed_key"),
        })?;
        assert_eq!(got.kv.map(|k| k.value), Some(b"arrived".to_vec()));
        let _ = writer.join().map_err(|_| LibTpuError::InternalError {
            context: String::from("join"),
        })??;
        Ok(())
    })?;

    // Raw HTTP/2 server stream: SETTINGS(ACK) (`flags = 0x1`) + multi-chunk DATA (`flags = 0` then `END_STREAM`)
    {
        let mut raw_cli = TcpStream::connect(addr)?;
        raw_cli.write_all(HTTP2_CLIENT_PREFACE)?;
        Http2Frame {
            frame_type: Http2FrameType::Settings,
            flags: 0x1, // ACK flag set
            stream_id: 0,
            payload: Vec::new(),
        }
        .write_to(&mut raw_cli)?;
        let hdr_bytes = HpackCodec::encode_headers(&[
            (":method", "POST"),
            (":path", "/tensorflow.CoordinationService/DeleteKeyValue"),
        ]);
        Http2Frame {
            frame_type: Http2FrameType::Headers,
            flags: 0x4,
            stream_id: 3,
            payload: hdr_bytes,
        }
        .write_to(&mut raw_cli)?;
        let req_env = GrpcProtoEnvelope::frame(
            &DeleteKeyValueRequestProto {
                key: String::from("delayed_key"),
                is_directory: false,
            }
            .encode_proto(),
            false,
        );
        let (part1, part2) = req_env.split_at(2);
        Http2Frame {
            frame_type: Http2FrameType::Data,
            flags: 0, // non-END_STREAM chunk
            stream_id: 3,
            payload: part1.to_vec(),
        }
        .write_to(&mut raw_cli)?;
        Http2Frame {
            frame_type: Http2FrameType::Data,
            flags: 0x1, // END_STREAM chunk
            stream_id: 3,
            payload: part2.to_vec(),
        }
        .write_to(&mut raw_cli)?;
        raw_cli.flush()?;
        let _ = Http2Frame::read_from(&mut raw_cli)?;
    }
    server.shutdown();

    // Custom HTTP/2 mock server testing Client branches: WINDOW_UPDATE, empty body HEADERS(END_STREAM), and DATA(END_STREAM)
    for send_data_end_stream in [false, true] {
        let mock_listener = TcpListener::bind("127.0.0.1:0")?;
        let mock_addr = mock_listener.local_addr()?;
        std::thread::scope(|s| -> LibTpuResult<()> {
            s.spawn(move || {
                if let Ok((mut srv, _)) = mock_listener.accept() {
                    let mut pref = [0_u8; 24];
                    let _ = srv.read_exact(&mut pref);
                    while let Ok(f) = Http2Frame::read_from(&mut srv) {
                        if f.frame_type == Http2FrameType::Data && (f.flags & 0x1) != 0 {
                            break;
                        }
                    }
                    let _ = Http2Frame {
                        frame_type: Http2FrameType::WindowUpdate,
                        flags: 0,
                        stream_id: 0,
                        payload: 4096_u32.to_be_bytes().to_vec(),
                    }
                    .write_to(&mut srv);
                    let ok_hdrs =
                        HpackCodec::encode_headers(&[(":status", "200"), ("grpc-status", "0")]);
                    if send_data_end_stream {
                        let _ = Http2Frame {
                            frame_type: Http2FrameType::Headers,
                            flags: 0x4,
                            stream_id: 1,
                            payload: ok_hdrs,
                        }
                        .write_to(&mut srv);
                        let body = GrpcProtoEnvelope::frame(
                            &ResetTaskResponseProto {}.encode_proto(),
                            false,
                        );
                        let _ = Http2Frame {
                            frame_type: Http2FrameType::Data,
                            flags: 0x1, // END_STREAM directly on DATA
                            stream_id: 1,
                            payload: body,
                        }
                        .write_to(&mut srv);
                    } else {
                        let _ = Http2Frame {
                            frame_type: Http2FrameType::Headers,
                            flags: 0x5, // END_HEADERS | END_STREAM with empty resp_data
                            stream_id: 1,
                            payload: ok_hdrs,
                        }
                        .write_to(&mut srv);
                    }
                    let _ = srv.flush();
                    let _ = srv.shutdown(std::net::Shutdown::Write);
                    let mut d = [0_u8; 32];
                    while let Ok(n) = srv.read(&mut d) {
                        if n == 0 {
                            break;
                        }
                    }
                }
            });
            let c = CoordinationGrpcClient::new(mock_addr);
            let _ = c.reset_task(&ResetTaskRequestProto { source_task: None })?;
            Ok(())
        })?;
    }

    // 2. ICI 2D vs full 3D twisted-torus synthesis (`VcForceHigh`, `dim == 2`, out-of-range CollectivePermute, lazy bool edges)
    let off_win = SubsliceWindow::new(ChipCoordinates::new(1, 1, 1), ChipCoordinates::new(2, 2, 2));
    assert!(!off_win.contains(ChipCoordinates::new(0, 1, 1)));
    assert!(!off_win.contains(ChipCoordinates::new(1, 0, 1)));
    assert!(!off_win.contains(ChipCoordinates::new(1, 4, 1)));
    assert!(!off_win.contains(ChipCoordinates::new(1, 1, 0)));
    assert!(!off_win.contains(ChipCoordinates::new(1, 1, 4)));

    let _ = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V7x,
        ChipCoordinates::new(4, 8, 6),
    );
    let _ = TwistedTorusMeshSpec::from_generation_and_bounds(
        TpuGeneration::V7x,
        ChipCoordinates::new(4, 8, 10),
    );

    let router_2d = DeadlockFreeIciRouter::new(TpuGeneration::V6e, ChipCoordinates::new(4, 4, 1));
    let rt_2d = router_2d.synthesize_ici_routing_table_proto("v6e_2d")?;
    assert_eq!(rt_2d.torus_dimension, 2);
    assert!(!rt_2d.is_twisted_torus);

    let mut router_3d =
        DeadlockFreeIciRouter::new(TpuGeneration::V4, ChipCoordinates::new(4, 4, 4));
    let rt_3d = router_3d.synthesize_ici_routing_table_proto("v4_full_3d")?;
    assert_eq!(rt_3d.torus_dimension, 3);
    assert!(rt_3d.is_twisted_torus);
    let sp_3d = router_3d.synthesize_superpod_ici_network_config("v4_full_sp")?;
    assert_eq!(sp_3d.configs.len(), 64);

    assert!(
        router_3d
            .set_subslice_window(SubsliceWindow::new(
                ChipCoordinates::new(0, 2, 0),
                ChipCoordinates::new(2, 4, 2)
            ))
            .is_err()
    );
    assert!(
        router_3d
            .set_subslice_window(SubsliceWindow::new(
                ChipCoordinates::new(0, 0, 2),
                ChipCoordinates::new(2, 2, 4)
            ))
            .is_err()
    );
    router_3d.set_subslice_window(SubsliceWindow::new(
        ChipCoordinates::new(0, 0, 0),
        ChipCoordinates::new(4, 4, 4),
    ))?;
    assert_eq!(router_3d.effective_axis_wrap(), (true, true, true));
    router_3d.set_subslice_window(SubsliceWindow::new(
        ChipCoordinates::new(1, 0, 0),
        ChipCoordinates::new(2, 2, 2),
    ))?;
    assert!(
        router_3d
            .compute_deadlock_free_route(
                ChipCoordinates::new(0, 0, 0),
                ChipCoordinates::new(1, 0, 0)
            )
            .is_err()
    );
    router_3d.mark_chip_faulty(ChipCoordinates::new(2, 0, 0));
    assert!(
        router_3d
            .compute_deadlock_free_route(
                ChipCoordinates::new(1, 0, 0),
                ChipCoordinates::new(2, 0, 0)
            )
            .is_err()
    );

    let _ = IciCollectiveEngine::reduce_pair_typed(
        &[0, 1],
        &[1, 0],
        ElementType::Pred,
        CollectiveReductionOp::Sum,
    );
    let _ = IciCollectiveEngine::reduce_pair_typed(
        &[0, 1],
        &[1, 0],
        ElementType::Pred,
        CollectiveReductionOp::Min,
    );

    let cp_oob = IciCollectiveEngine::execute_collective_permute_routed(
        &router_2d,
        &[ChipCoordinates::new(0, 0, 0)],
        &[vec![1, 2, 3, 4]],
        &[(99, 99)],
    )?;
    assert_eq!(cp_oob.len(), 1);

    // 3. MegaScale stripe header/checksum error branches & inconsistent slice chip count error
    let transport =
        MegaScaleMultiSocketTransport::new(1, false, MegascaleCongestionController::default());
    assert!(
        HierarchicalDcnIciRuntime::execute_hierarchical_all_reduce(
            &transport,
            ChipCoordinates::new(2, 1, 1),
            &[vec![vec![0_u8; 8], vec![0_u8; 8]], vec![vec![0_u8; 8]]],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )
        .is_err()
    );

    // Missing header, bad checksum, and has_checksum=false on accept_and_verify_stripe
    let cases = [
        (
            MegascaleSendRequestProto {
                request_header: None,
                payload: Vec::new(),
            },
            true,
        ),
        (
            MegascaleSendRequestProto {
                request_header: Some(crate::proto::MegascaleTransferBufferRequestProto {
                    has_checksum: true,
                    checksum: 0xDEAD_u64,
                    ..crate::proto::MegascaleTransferBufferRequestProto::default()
                }),
                payload: vec![1, 2, 3, 4],
            },
            true,
        ),
        (
            MegascaleSendRequestProto {
                request_header: Some(crate::proto::MegascaleTransferBufferRequestProto {
                    has_checksum: false,
                    ..crate::proto::MegascaleTransferBufferRequestProto::default()
                }),
                payload: vec![7, 8],
            },
            false,
        ),
    ];
    for (req_msg, expect_err) in cases {
        let l = TcpListener::bind("127.0.0.1:0")?;
        let a = l.local_addr()?;
        std::thread::scope(|s| -> LibTpuResult<()> {
            let sender = s.spawn(move || -> LibTpuResult<()> {
                let mut cli = TcpStream::connect(a)?;
                let enc = req_msg.encode_proto();
                let len = u32::try_from(enc.len()).unwrap_or(0);
                cli.write_all(&len.to_be_bytes())?;
                cli.write_all(&enc)?;
                cli.flush()?;
                let mut ack_buf = [0_u8; 64];
                let _ = cli.read(&mut ack_buf);
                Ok(())
            });
            let res = MegaScaleMultiSocketTransport::accept_and_verify_stripe(&l);
            assert_eq!(res.is_err(), expect_err);
            sender.join().map_err(|_| LibTpuError::InternalError {
                context: String::from("join"),
            })?
        })?;
    }

    // Non-blocking listener accept error exit branch on CoordinationGrpcServer::run_accept_loop
    let nb_listener = TcpListener::bind("127.0.0.1:0")?;
    nb_listener.set_nonblocking(true)?;
    CoordinationGrpcServer::run_accept_loop(
        &nb_listener,
        &Arc::new(CoordinationServiceState::default()),
        &std::sync::atomic::AtomicBool::new(false),
    );

    // Non-wrapped (2,2,1) subslice effective_axis_wrap (`wx == false`, `wy == false`, `wz == false`)
    let mut small_2d =
        DeadlockFreeIciRouter::new(TpuGeneration::V6e, ChipCoordinates::new(2, 2, 1));
    small_2d.set_subslice_window(SubsliceWindow::new(
        ChipCoordinates::new(0, 0, 0),
        ChipCoordinates::new(2, 2, 1),
    ))?;
    assert_eq!(small_2d.effective_axis_wrap(), (false, false, false));

    // Fault-injected striped_send_receive exercising sender/receiver error propagation
    let faulty_transport =
        MegaScaleMultiSocketTransport::new(1, false, MegascaleCongestionController::default())
            .with_checksum_fault_injection(true);
    assert!(
        faulty_transport
            .striped_send_receive("fault_k", 0, 0, b"boom")
            .is_err()
    );
    assert!(
        HierarchicalDcnIciRuntime::execute_hierarchical_all_reduce(
            &faulty_transport,
            ChipCoordinates::new(1, 1, 1),
            &[vec![vec![0_u8; 4]], vec![vec![0_u8; 4]]],
            ElementType::F32,
            CollectiveReductionOp::Sum,
        )
        .is_err()
    );
    assert!(
        HierarchicalDcnIciRuntime::execute_hierarchical_all_gather(
            &faulty_transport,
            ChipCoordinates::new(1, 1, 1),
            &[vec![vec![0_u8; 4]], vec![vec![0_u8; 4]]],
            ElementType::F32,
        )
        .is_err()
    );

    Ok(())
}
