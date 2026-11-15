//! Wire-compatible HTTP/2 + Protobuf gRPC client and server for `tensorflow.CoordinationService`
//! and `OpenXLA` multi-host `LocalTopologyProto` -> `GlobalTopologyProto` discovery (`Phase 8.1`).
//!
//! Implements:
//! - RFC 9113 HTTP/2 connection preface (`PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n`) and 9-byte binary frame
//!   encoding/decoding (`SETTINGS`, `WINDOW_UPDATE`, `HEADERS`, `DATA`).
//! - RFC 7541 HPACK static/literal header block codec for `:method`, `:scheme`, `:path`, `:status`,
//!   `content-type: application/grpc+proto`, `grpc-status`, and `grpc-message`.
//! - Canonical 5-byte gRPC length-prefixed Protobuf message framing (`[compressed: u8][len_be: u32]`).
//! - Complete client (`CoordinationGrpcClient`) and server (`CoordinationGrpcServer`) for all 19
//!   `tensorflow.CoordinationService` RPCs, mapping every wire and status fault into [`LibTpuError`].
//! - Multi-host `LocalTopologyProto` -> `GlobalTopologyProto` cluster handshake (`MultiHostTopologyHandshake`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::distributed::MultiSliceTopologyAssembler;
use crate::error::{LibTpuError, LibTpuResult, PjrtErrorCode};
use crate::proto::{
    BarrierRequestProto, BarrierResponseProto, CancelBarrierRequestProto,
    CancelBarrierResponseProto, CoordinatedTaskProto, CoordinatedTaskStateInfoProto,
    CoordinatedTaskStateProto, CoordinationDeviceInfoProto, CoordinationServiceErrorProto,
    DeleteKeyValueRequestProto, DeleteKeyValueResponseProto, GetAliveTasksRequestProto,
    GetAliveTasksResponseProto, GetKeyValueDirRequestProto, GetKeyValueDirResponseProto,
    GetKeyValueRequestProto, GetKeyValueResponseProto, GetTaskStateRequestProto,
    GetTaskStateResponseProto, GlobalTopologyProto, HeartbeatRequestProto, HeartbeatResponseProto,
    IncrementKeyValueRequestProto, IncrementKeyValueResponseProto, InsertKeyValueRequestProto,
    InsertKeyValueResponseProto, KeyValueEntryProto, LocalTopologyProto, PollForErrorRequestProto,
    PollForErrorResponseProto, ProtoSchemaMessage, ProtobufAny, RegisterTaskRequestProto,
    RegisterTaskResponseProto, ReportErrorToServiceRequestProto, ReportErrorToServiceResponseProto,
    ReportErrorToTaskRequestProto, ReportErrorToTaskResponseProto, ResetTaskRequestProto,
    ResetTaskResponseProto, ShutdownTaskRequestProto, ShutdownTaskResponseProto,
    TryGetKeyValueRequestProto, TryGetKeyValueResponseProto, WaitForAllTasksRequestProto,
    WaitForAllTasksResponseProto, WatchJobStateRequestProto, WatchJobStateResponseProto,
};

/// RFC 9113 Section 3.4 24-byte HTTP/2 client connection preface.
pub const HTTP2_CLIENT_PREFACE: &[u8; 24] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// HTTP/2 binary frame type codes (RFC 9113 Section 6).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Http2FrameType {
    /// `DATA` frame (`0x0`) carrying gRPC length-prefixed message bytes.
    Data = 0x0,
    /// `HEADERS` frame (`0x1`) carrying initial request/response headers or gRPC trailers.
    Headers = 0x1,
    /// `SETTINGS` frame (`0x4`) exchanging connection parameters.
    Settings = 0x4,
    /// `WINDOW_UPDATE` frame (`0x8`) advertising flow-control credit.
    WindowUpdate = 0x8,
}

impl Http2FrameType {
    /// Converts a raw 8-bit HTTP/2 frame type tag into an [`Http2FrameType`].
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw HTTP/2 frame type byte (`0x0`, `0x1`, `0x4`, or `0x8`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `raw` is unsupported.
    pub fn from_raw(raw: u8) -> LibTpuResult<Self> {
        match raw {
            0x0 => Ok(Self::Data),
            0x1 => Ok(Self::Headers),
            0x4 => Ok(Self::Settings),
            0x8 => Ok(Self::WindowUpdate),
            other => Err(LibTpuError::DeserializationFailed {
                reason: format!("Unsupported HTTP/2 frame type 0x{other:02x}"),
            }),
        }
    }
}

/// HTTP/2 `END_STREAM` flag (`0x1`).
pub const HTTP2_FLAG_END_STREAM: u8 = 0x1;

/// HTTP/2 `ACK` flag (`0x1`) for `SETTINGS` frames.
pub const HTTP2_FLAG_ACK: u8 = 0x1;

/// HTTP/2 `END_HEADERS` flag (`0x4`) for `HEADERS` frames.
pub const HTTP2_FLAG_END_HEADERS: u8 = 0x4;

/// Decoded RFC 9113 HTTP/2 binary frame (`9`-byte header + variable payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Http2Frame {
    /// Frame type (`DATA`, `HEADERS`, `SETTINGS`, or `WINDOW_UPDATE`).
    pub frame_type: Http2FrameType,
    /// Frame bitflags (`END_STREAM`, `END_HEADERS`, `ACK`).
    pub flags: u8,
    /// 31-bit unsigned stream identifier (`0` for connection control, odd `>= 1` for RPC streams).
    pub stream_id: u32,
    /// Raw frame payload bytes.
    pub payload: Vec<u8>,
}

impl Http2Frame {
    /// Encodes this [`Http2Frame`] into its RFC 9113 9-byte header + payload wire representation.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::SerializationFailed`] if `payload.len()` exceeds the 24-bit (`16 MiB - 1`) limit.
    pub fn encode(&self) -> LibTpuResult<Vec<u8>> {
        if self.payload.len() > 0x00FF_FFFF {
            return Err(LibTpuError::SerializationFailed {
                reason: format!(
                    "HTTP/2 frame payload {} exceeds 24-bit maximum",
                    self.payload.len()
                ),
            });
        }
        let len_u32 = u32::try_from(self.payload.len()).unwrap_or(0);
        let len_be = len_u32.to_be_bytes();
        let sid_be = (self.stream_id & 0x7FFF_FFFF).to_be_bytes();
        let mut out = Vec::with_capacity(9 + self.payload.len());
        out.extend_from_slice(&[len_be[1], len_be[2], len_be[3]]);
        out.push(self.frame_type as u8);
        out.push(self.flags);
        out.extend_from_slice(&sid_be);
        out.extend_from_slice(&self.payload);
        Ok(out)
    }

    /// Writes this [`Http2Frame`] to `writer`.
    ///
    /// # Arguments
    ///
    /// * `writer` - Destination byte stream.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if frame encoding or stream I/O fails.
    pub fn write_to(&self, writer: &mut dyn Write) -> LibTpuResult<()> {
        let raw = self.encode()?;
        writer.write_all(&raw)?;
        Ok(())
    }

    /// Reads and decodes a single [`Http2Frame`] from `reader`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Source byte stream positioned at an HTTP/2 frame boundary.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if reading the 9-byte header/payload fails or the frame type is unknown.
    pub fn read_from(reader: &mut dyn Read) -> LibTpuResult<Self> {
        let mut hdr = [0_u8; 9];
        reader.read_exact(&mut hdr)?;
        let length = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
        let frame_type = Http2FrameType::from_raw(hdr[3])?;
        let flags = hdr[4];
        let stream_id = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) & 0x7FFF_FFFF;
        let mut payload = vec![0_u8; length];
        reader.read_exact(&mut payload)?;
        Ok(Self {
            frame_type,
            flags,
            stream_id,
            payload,
        })
    }
}

/// RFC 7541 HPACK header block encoder/decoder for HTTP/2 gRPC headers and trailers.
#[derive(Debug, Clone, Copy, Default)]
pub struct HpackCodec;

impl HpackCodec {
    /// Appends an RFC 7541 7-bit prefix unsigned integer onto `dst`.
    ///
    /// # Arguments
    ///
    /// * `dst` - Output byte buffer.
    /// * `value` - Non-negative integer length to encode.
    pub fn encode_varint7(dst: &mut Vec<u8>, mut value: usize) {
        if value < 127 {
            #[allow(clippy::cast_possible_truncation)]
            dst.push(value as u8);
            return;
        }
        dst.push(0x7F);
        value -= 127;
        while value >= 128 {
            #[allow(clippy::cast_possible_truncation)]
            dst.push(((value & 0x7F) as u8) | 0x80);
            value >>= 7;
        }
        #[allow(clippy::cast_possible_truncation)]
        dst.push(value as u8);
    }

    /// Decodes an RFC 7541 7-bit prefix unsigned integer from `src` starting at `*cursor`.
    ///
    /// # Arguments
    ///
    /// * `src` - Input HPACK block slice.
    /// * `cursor` - Mutable byte offset into `src`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `src` ends mid-varint.
    pub fn decode_varint7(src: &[u8], cursor: &mut usize) -> LibTpuResult<usize> {
        let Some(&first) = src.get(*cursor) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("HPACK string length truncated"),
            });
        };
        *cursor += 1;
        let prefix = usize::from(first & 0x7F);
        if prefix < 127 {
            return Ok(prefix);
        }
        let mut val = 127_usize;
        let mut shift = 0_u32;
        loop {
            let Some(&b) = src.get(*cursor) else {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from("HPACK multi-byte varint truncated"),
                });
            };
            *cursor += 1;
            val = val.saturating_add(usize::from(b & 0x7F) << shift);
            if (b & 0x80) == 0 {
                return Ok(val);
            }
            shift = shift.saturating_add(7);
        }
    }

    /// Encodes `(name, value)` header pairs into an RFC 7541 HPACK header block using static-table
    /// indices for `:method = POST` (`0x83`), `:scheme = http` (`0x86`), and `:status = 200` (`0x88`),
    /// and literal-without-indexing (`0x00`) fields for all other headers.
    ///
    /// # Arguments
    ///
    /// * `headers` - Ordered slice of `(name, value)` HTTP/2 header pairs.
    #[must_use]
    pub fn encode_headers(headers: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(name, value) in headers {
            match (name, value) {
                (":method", "POST") => out.push(0x83),
                (":scheme", "http") => out.push(0x86),
                (":status", "200") => out.push(0x88),
                _ => {
                    out.push(0x00);
                    Self::encode_varint7(&mut out, name.len());
                    out.extend_from_slice(name.as_bytes());
                    Self::encode_varint7(&mut out, value.len());
                    out.extend_from_slice(value.as_bytes());
                }
            }
        }
        out
    }

    /// Decodes an HPACK header block produced by [`Self::encode_headers`] or standard HTTP/2 peers.
    ///
    /// # Arguments
    ///
    /// * `block` - Raw HPACK header block bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if any entry is truncated or non-UTF-8.
    pub fn decode_headers(block: &[u8]) -> LibTpuResult<BTreeMap<String, String>> {
        let mut map = BTreeMap::new();
        let mut cursor = 0_usize;
        while let Some(&tag) = block.get(cursor) {
            match tag {
                0x83 => {
                    map.insert(String::from(":method"), String::from("POST"));
                    cursor += 1;
                }
                0x86 => {
                    map.insert(String::from(":scheme"), String::from("http"));
                    cursor += 1;
                }
                0x88 => {
                    map.insert(String::from(":status"), String::from("200"));
                    cursor += 1;
                }
                0x00 | 0x10 | 0x40 => {
                    cursor += 1;
                    let name_len = Self::decode_varint7(block, &mut cursor)?;
                    let end_name = cursor.saturating_add(name_len);
                    let Some(name_bytes) = block.get(cursor..end_name) else {
                        return Err(LibTpuError::DeserializationFailed {
                            reason: String::from("HPACK literal header name truncated"),
                        });
                    };
                    cursor = end_name;
                    let val_len = Self::decode_varint7(block, &mut cursor)?;
                    let end_val = cursor.saturating_add(val_len);
                    let Some(val_bytes) = block.get(cursor..end_val) else {
                        return Err(LibTpuError::DeserializationFailed {
                            reason: String::from("HPACK literal header value truncated"),
                        });
                    };
                    cursor = end_val;
                    let name_str = std::str::from_utf8(name_bytes).map_err(|e| {
                        LibTpuError::DeserializationFailed {
                            reason: format!("HPACK header name UTF-8 error: {e}"),
                        }
                    })?;
                    let val_str = std::str::from_utf8(val_bytes).map_err(|e| {
                        LibTpuError::DeserializationFailed {
                            reason: format!("HPACK header value UTF-8 error: {e}"),
                        }
                    })?;
                    map.insert(String::from(name_str), String::from(val_str));
                }
                other => {
                    return Err(LibTpuError::DeserializationFailed {
                        reason: format!("Unsupported HPACK representation byte 0x{other:02x}"),
                    });
                }
            }
        }
        Ok(map)
    }
}

/// Wire-level gRPC 5-byte length-prefixed Protobuf message framing (`[1B compressed][4B len_be][body]`).
#[derive(Debug, Clone, Copy, Default)]
pub struct GrpcProtoEnvelope;

impl GrpcProtoEnvelope {
    /// Frames a serialized Protobuf message in the canonical 5-byte gRPC `DATA` payload envelope,
    /// optionally applying RFC 7932 Brotli compression (`compressed_flag = 1`).
    ///
    /// # Arguments
    ///
    /// * `proto_bytes` - Raw `prost`-encoded Protobuf message bytes.
    /// * `compress` - When `true`, compresses `proto_bytes` with Brotli and sets `compressed_flag = 1`.
    #[must_use]
    pub fn frame(proto_bytes: &[u8], compress: bool) -> Vec<u8> {
        let body = if compress {
            let mut comp = Vec::new();
            {
                let mut writer = brotli::CompressorWriter::new(&mut comp, 4096, 4, 20);
                let _ = writer.write_all(proto_bytes);
            }
            comp
        } else {
            proto_bytes.to_vec()
        };
        let len_u32 = u32::try_from(body.len()).unwrap_or(0);
        let mut out = Vec::with_capacity(5 + body.len());
        out.push(u8::from(compress));
        out.extend_from_slice(&len_u32.to_be_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Unframes and (when `compressed_flag == 1`) Brotli-decompresses a gRPC 5-byte length-prefixed envelope.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw HTTP/2 `DATA` frame payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the 5-byte prefix is missing, truncated,
    /// uses an invalid compression flag (`> 1`), or fails Brotli decompression.
    pub fn unframe(data: &[u8]) -> LibTpuResult<Vec<u8>> {
        let Some(&[comp_flag, b0, b1, b2, b3]) = data.get(0..5) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!("gRPC DATA frame shorter than 5 bytes ({})", data.len()),
            });
        };
        let msg_len = u32::from_be_bytes([b0, b1, b2, b3]) as usize;
        let end = 5_usize.saturating_add(msg_len);
        let Some(raw_msg) = data.get(5..end) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "gRPC message truncated: expected {msg_len} bytes, got {}",
                    data.len().saturating_sub(5)
                ),
            });
        };
        match comp_flag {
            0 => Ok(raw_msg.to_vec()),
            1 => {
                let mut out = Vec::new();
                let mut dec = brotli::Decompressor::new(raw_msg, 4096);
                dec.read_to_end(&mut out)
                    .map_err(|e| LibTpuError::DeserializationFailed {
                        reason: format!("gRPC Brotli decompression failed: {e}"),
                    })?;
                Ok(out)
            }
            other => Err(LibTpuError::DeserializationFailed {
                reason: format!("Invalid gRPC compressed_flag {other}"),
            }),
        }
    }
}

/// Canonical `(job_name, task_id)` key identifying a worker task in `CoordinationService`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TaskKey {
    /// Coordinated job name (defaults to `"jax_worker"` when empty).
    job_name: String,
    /// Zero-based worker task index.
    task_id: i32,
}

impl TaskKey {
    /// Normalizes an optional [`CoordinatedTaskProto`] into a [`TaskKey`].
    ///
    /// # Arguments
    ///
    /// * `task` - Optional [`CoordinatedTaskProto`] descriptor.
    fn from_proto(task: Option<&CoordinatedTaskProto>) -> Self {
        let (job, id) = task.map_or_else(
            || (String::from("jax_worker"), 0),
            |t| {
                let j = if t.job_name.is_empty() {
                    String::from("jax_worker")
                } else {
                    t.job_name.clone()
                };
                (j, t.task_id)
            },
        );
        Self {
            job_name: job,
            task_id: id,
        }
    }

    /// Converts this [`TaskKey`] back into a [`CoordinatedTaskProto`].
    fn to_proto(&self) -> CoordinatedTaskProto {
        CoordinatedTaskProto {
            job_name: self.job_name.clone(),
            task_id: self.task_id,
            recoverable: false,
        }
    }
}

/// Active state of a named distributed barrier inside [`CoordinationServiceState`].
#[derive(Debug, Clone, Default)]
struct BarrierSlot {
    /// Expected set of participating tasks (or cluster size when `explicit_tasks` is empty).
    arrived_tasks: BTreeSet<TaskKey>,
    /// Whether `CancelBarrier` was invoked for `(barrier_id, counter)`.
    cancelled: bool,
}

/// Mutable inner cluster state synchronized by `Mutex` + `Condvar` inside [`CoordinationServiceState`].
#[derive(Debug)]
struct InnerCoordState {
    /// Leader service incarnation ID (`fixed64`).
    leader_incarnation: u64,
    /// Expected total cluster task count (`num_tasks`).
    expected_num_tasks: usize,
    /// Per-task state records keyed by `(job_name, task_id)`.
    tasks: BTreeMap<TaskKey, CoordinatedTaskStateInfoProto>,
    /// Per-task `DeviceInfo` contributed during `WaitForAllTasks`.
    task_devices: BTreeMap<TaskKey, Vec<ProtobufAny>>,
    /// Monotonic job state version incremented on every task state transition.
    job_state_version: i64,
    /// Distributed Key-Value store (`key -> bytes`).
    kv_store: BTreeMap<String, Vec<u8>>,
    /// Active barrier slots keyed by `(barrier_id, counter)`.
    barriers: BTreeMap<(String, i64), BarrierSlot>,
    /// Optional service-wide fatal error `(status_code, message, source_task)` reported via `ReportErrorToService`.
    service_error: Option<(i32, String, CoordinatedTaskProto)>,
}

/// Thread-safe state machine implementing all 19 `tensorflow.CoordinationService` RPCs.
#[derive(Debug)]
pub struct CoordinationServiceState {
    /// Synchronized cluster state.
    inner: Mutex<InnerCoordState>,
    /// Condition variable notified on KV inserts, task arrivals, barrier updates, and errors.
    cv: Condvar,
}

impl Default for CoordinationServiceState {
    fn default() -> Self {
        Self::new(1, 0xC00D_2026_0000_0001)
    }
}

impl CoordinationServiceState {
    /// Creates a new [`CoordinationServiceState`] expecting `expected_num_tasks` workers.
    ///
    /// # Arguments
    ///
    /// * `expected_num_tasks` - Total tasks required before `WaitForAllTasks` and implicit barriers release.
    /// * `leader_incarnation` - Unique `fixed64` leader incarnation identifier.
    #[must_use]
    pub const fn new(expected_num_tasks: usize, leader_incarnation: u64) -> Self {
        let count = if expected_num_tasks == 0 {
            1
        } else {
            expected_num_tasks
        };
        Self {
            inner: Mutex::new(InnerCoordState {
                leader_incarnation,
                expected_num_tasks: count,
                tasks: BTreeMap::new(),
                task_devices: BTreeMap::new(),
                job_state_version: 1,
                kv_store: BTreeMap::new(),
                barriers: BTreeMap::new(),
                service_error: None,
            }),
            cv: Condvar::new(),
        }
    }

    /// Dispatches a raw unary gRPC method path (`/tensorflow.CoordinationService/<Method>`) and
    /// serialized request Protobuf payload, returning `(grpc_status_code, grpc_message, resp_proto_bytes)`.
    ///
    /// # Arguments
    ///
    /// * `path` - HTTP/2 `:path` header value.
    /// * `req_bytes` - Unframed Protobuf request bytes.
    #[must_use]
    pub fn dispatch_unary_rpc(&self, path: &str, req_bytes: &[u8]) -> (i32, String, Vec<u8>) {
        let method = path
            .strip_prefix("/tensorflow.CoordinationService/")
            .unwrap_or(path);
        match self.dispatch_method_inner(method, req_bytes) {
            Ok(resp_bytes) => (0, String::new(), resp_bytes),
            Err(err) => {
                let (code, msg) = err.grpc_status_and_message();
                (code, msg, Vec::new())
            }
        }
    }

    /// Decodes `T`, invokes `handler`, and encodes the response message.
    ///
    /// # Arguments
    ///
    /// * `rpc_method` - Static `CoordinationService` RPC name.
    /// * `req_bytes` - Raw request Protobuf payload bytes.
    /// * `handler` - Typed request handler closure.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CoordinationServiceError`] if request decoding or `handler` fails.
    fn decode_and_run<Req, Resp, F>(
        rpc_method: &'static str,
        req_bytes: &[u8],
        handler: F,
    ) -> LibTpuResult<Vec<u8>>
    where
        Req: ProtoSchemaMessage,
        Resp: ProtoSchemaMessage,
        F: FnOnce(Req) -> LibTpuResult<Resp>,
    {
        let req = match Req::decode_proto(req_bytes) {
            Ok(val) => val,
            Err(e) => {
                return Err(LibTpuError::CoordinationServiceError {
                    rpc_method,
                    task_id: 0,
                    status_code: PjrtErrorCode::InvalidArgument as i32,
                    message: e.to_string(),
                });
            }
        };
        let resp = handler(req)?;
        Ok(resp.encode_proto())
    }

    /// Routes `method` across all 19 `CoordinationService` RPCs.
    ///
    /// # Arguments
    ///
    /// * `method` - Unqualified `CoordinationService` RPC method name.
    /// * `req_bytes` - Raw request Protobuf payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CoordinationServiceError`] if `method` is unknown or handler execution fails.
    fn dispatch_method_inner(&self, method: &str, req_bytes: &[u8]) -> LibTpuResult<Vec<u8>> {
        match method {
            "RegisterTask" => {
                Self::decode_and_run("RegisterTask", req_bytes, |r| self.handle_register_task(&r))
            }
            "WaitForAllTasks" => Self::decode_and_run("WaitForAllTasks", req_bytes, |r| {
                self.handle_wait_for_all_tasks(r)
            }),
            "ShutdownTask" => {
                Self::decode_and_run("ShutdownTask", req_bytes, |r| self.handle_shutdown_task(&r))
            }
            "ResetTask" => {
                Self::decode_and_run("ResetTask", req_bytes, |r| Ok(self.handle_reset_task(&r)))
            }
            "Heartbeat" => {
                Self::decode_and_run("Heartbeat", req_bytes, |r| self.handle_heartbeat(&r))
            }
            "ReportErrorToTask" => Self::decode_and_run("ReportErrorToTask", req_bytes, |r| {
                Ok(self.handle_report_error_to_task(r))
            }),
            "ReportErrorToService" => {
                Self::decode_and_run("ReportErrorToService", req_bytes, |r| {
                    Ok(self.handle_report_error_to_service(r))
                })
            }
            "GetTaskState" => Self::decode_and_run("GetTaskState", req_bytes, |r| {
                Ok(self.handle_get_task_state(&r))
            }),
            "WatchJobState" => Self::decode_and_run("WatchJobState", req_bytes, |r| {
                Ok(self.handle_watch_job_state(&r))
            }),
            "InsertKeyValue" => Self::decode_and_run("InsertKeyValue", req_bytes, |r| {
                self.handle_insert_key_value(r)
            }),
            "GetKeyValue" => Self::decode_and_run("GetKeyValue", req_bytes, |r| {
                self.handle_get_key_value(&r, Duration::from_millis(250))
            }),
            "TryGetKeyValue" => Self::decode_and_run("TryGetKeyValue", req_bytes, |r| {
                self.handle_try_get_key_value(&r)
            }),
            "IncrementKeyValue" => Self::decode_and_run("IncrementKeyValue", req_bytes, |r| {
                self.handle_increment_key_value(&r)
            }),
            "GetKeyValueDir" => Self::decode_and_run("GetKeyValueDir", req_bytes, |r| {
                Ok(self.handle_get_key_value_dir(&r))
            }),
            "DeleteKeyValue" => Self::decode_and_run("DeleteKeyValue", req_bytes, |r| {
                Ok(self.handle_delete_key_value(&r))
            }),
            "Barrier" => Self::decode_and_run("Barrier", req_bytes, |r| self.handle_barrier(&r)),
            "CancelBarrier" => Self::decode_and_run("CancelBarrier", req_bytes, |r| {
                self.handle_cancel_barrier(&r)
            }),
            "GetAliveTasks" => Self::decode_and_run("GetAliveTasks", req_bytes, |r| {
                Ok(self.handle_get_alive_tasks(&r))
            }),
            "PollForError" => Self::decode_and_run("PollForError", req_bytes, |r| {
                self.handle_poll_for_error(&r)
            }),
            _ => Err(LibTpuError::CoordinationServiceError {
                rpc_method: "UnknownMethod",
                task_id: 0,
                status_code: PjrtErrorCode::Unimplemented as i32,
                message: format!("Unknown CoordinationService RPC '{method}'"),
            }),
        }
    }

    /// Handles `RegisterTask`: registers `(job_name, task_id)` with `incarnation`, rejecting
    /// duplicate registrations with a conflicting incarnation while `TASKSTATE_CONNECTED`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`RegisterTaskRequestProto`] payload.
    fn handle_register_task(
        &self,
        req: &RegisterTaskRequestProto,
    ) -> Result<RegisterTaskResponseProto, LibTpuError> {
        let key = TaskKey::from_proto(req.source_task.as_ref());
        let mut guard = self.inner.lock();
        if let Some(existing) = guard.tasks.get(&key)
            && existing.state == CoordinatedTaskStateProto::TaskstateConnected as i32
            && existing.incarnation != req.incarnation
        {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "RegisterTask",
                task_id: key.task_id,
                status_code: PjrtErrorCode::Aborted as i32,
                message: format!(
                    "Task {}:{} already connected with incarnation {} != {}",
                    key.job_name, key.task_id, existing.incarnation, req.incarnation
                ),
            });
        }
        let info = CoordinatedTaskStateInfoProto {
            task: Some(key.to_proto()),
            incarnation: req.incarnation,
            state: CoordinatedTaskStateProto::TaskstateConnected as i32,
            error_code: 0,
            error_message: String::new(),
            error_payload: None,
        };
        guard.tasks.insert(key, info);
        guard.job_state_version = guard.job_state_version.saturating_add(1);
        let leader = guard.leader_incarnation;
        drop(guard);
        self.cv.notify_all();
        Ok(RegisterTaskResponseProto {
            leader_incarnation: leader,
        })
    }

    /// Handles `WaitForAllTasks`: records local `device_info` and blocks (up to `2s`) until
    /// `expected_num_tasks` tasks are connected, returning merged cluster [`CoordinationDeviceInfoProto`].
    ///
    /// # Arguments
    ///
    /// * `req` - [`WaitForAllTasksRequestProto`] payload.
    fn handle_wait_for_all_tasks(
        &self,
        req: WaitForAllTasksRequestProto,
    ) -> Result<WaitForAllTasksResponseProto, LibTpuError> {
        let key = TaskKey::from_proto(req.source_task.as_ref());
        let devs = req.device_info.map(|d| d.device).unwrap_or_default();
        let mut guard = self.inner.lock();
        guard.task_devices.insert(key.clone(), devs);
        self.cv.notify_all();

        let deadline = Instant::now() + Duration::from_secs(2);
        while guard
            .tasks
            .values()
            .filter(|t| t.state == CoordinatedTaskStateProto::TaskstateConnected as i32)
            .count()
            < guard.expected_num_tasks
            && guard.service_error.is_none()
        {
            if self.cv.wait_until(&mut guard, deadline).timed_out() {
                return Err(LibTpuError::CoordinationServiceError {
                    rpc_method: "WaitForAllTasks",
                    task_id: key.task_id,
                    status_code: PjrtErrorCode::DeadlineExceeded as i32,
                    message: String::from("Timed out waiting for all cluster tasks to register"),
                });
            }
        }
        if let Some((code, ref msg, _)) = guard.service_error {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "WaitForAllTasks",
                task_id: key.task_id,
                status_code: code,
                message: msg.clone(),
            });
        }
        let mut merged = Vec::new();
        for dlist in guard.task_devices.values() {
            merged.extend(dlist.iter().cloned());
        }
        let leader = guard.leader_incarnation;
        drop(guard);
        Ok(WaitForAllTasksResponseProto {
            leader_incarnation: leader,
            device_info: Some(CoordinationDeviceInfoProto { device: merged }),
        })
    }

    /// Handles `ShutdownTask`: transitions the task to `TASKSTATE_DISCONNECTED`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`ShutdownTaskRequestProto`] payload.
    fn handle_shutdown_task(
        &self,
        req: &ShutdownTaskRequestProto,
    ) -> Result<ShutdownTaskResponseProto, LibTpuError> {
        let key = TaskKey::from_proto(req.source_task.as_ref());
        let mut guard = self.inner.lock();
        let Some(entry) = guard.tasks.get_mut(&key) else {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "ShutdownTask",
                task_id: key.task_id,
                status_code: PjrtErrorCode::FailedPrecondition as i32,
                message: format!("Task {}:{} is not registered", key.job_name, key.task_id),
            });
        };
        entry.state = CoordinatedTaskStateProto::TaskstateDisconnected as i32;
        guard.job_state_version = guard.job_state_version.saturating_add(1);
        drop(guard);
        self.cv.notify_all();
        Ok(ShutdownTaskResponseProto {})
    }

    /// Handles `ResetTask`: resets the task to `TASKSTATE_UNINITIALIZED` and clears task errors.
    ///
    /// # Arguments
    ///
    /// * `req` - [`ResetTaskRequestProto`] payload.
    fn handle_reset_task(&self, req: &ResetTaskRequestProto) -> ResetTaskResponseProto {
        let key = TaskKey::from_proto(req.source_task.as_ref());
        let mut guard = self.inner.lock();
        guard.tasks.insert(
            key.clone(),
            CoordinatedTaskStateInfoProto {
                task: Some(key.to_proto()),
                incarnation: 0,
                state: CoordinatedTaskStateProto::TaskstateUninitialized as i32,
                error_code: 0,
                error_message: String::new(),
                error_payload: None,
            },
        );
        guard.task_devices.remove(&key);
        guard.job_state_version = guard.job_state_version.saturating_add(1);
        drop(guard);
        self.cv.notify_all();
        ResetTaskResponseProto {}
    }

    /// Handles `Heartbeat`: verifies `incarnation` matches the registered task and returns `leader_incarnation`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`HeartbeatRequestProto`] payload.
    fn handle_heartbeat(
        &self,
        req: &HeartbeatRequestProto,
    ) -> Result<HeartbeatResponseProto, LibTpuError> {
        let key = TaskKey::from_proto(req.source_task.as_ref());
        let guard = self.inner.lock();
        let Some(entry) = guard.tasks.get(&key) else {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "Heartbeat",
                task_id: key.task_id,
                status_code: PjrtErrorCode::InvalidArgument as i32,
                message: format!("Unregistered task {}:{}", key.job_name, key.task_id),
            });
        };
        if entry.incarnation != req.incarnation {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "Heartbeat",
                task_id: key.task_id,
                status_code: PjrtErrorCode::Aborted as i32,
                message: format!(
                    "Task incarnation mismatch: expected {}, got {}",
                    entry.incarnation, req.incarnation
                ),
            });
        }
        Ok(HeartbeatResponseProto {
            leader_incarnation: guard.leader_incarnation,
        })
    }

    /// Handles `ReportErrorToTask`: transitions the target task into `TASKSTATE_ERROR`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`ReportErrorToTaskRequestProto`] payload.
    fn handle_report_error_to_task(
        &self,
        req: ReportErrorToTaskRequestProto,
    ) -> ReportErrorToTaskResponseProto {
        let src_task = req
            .error_payload
            .as_ref()
            .and_then(|p| p.source_task.as_ref());
        let key = TaskKey::from_proto(src_task);
        let mut guard = self.inner.lock();
        let code = if req.error_code == 0 {
            PjrtErrorCode::Internal as i32
        } else {
            req.error_code
        };
        guard.tasks.insert(
            key.clone(),
            CoordinatedTaskStateInfoProto {
                task: Some(key.to_proto()),
                incarnation: 0,
                state: CoordinatedTaskStateProto::TaskstateError as i32,
                error_code: code,
                error_message: req.error_message,
                error_payload: req.error_payload,
            },
        );
        guard.job_state_version = guard.job_state_version.saturating_add(1);
        drop(guard);
        self.cv.notify_all();
        ReportErrorToTaskResponseProto {}
    }

    /// Handles `ReportErrorToService`: records a cluster-wide fatal error and marks all tasks in error.
    ///
    /// # Arguments
    ///
    /// * `req` - [`ReportErrorToServiceRequestProto`] payload.
    fn handle_report_error_to_service(
        &self,
        req: ReportErrorToServiceRequestProto,
    ) -> ReportErrorToServiceResponseProto {
        let origin_key = TaskKey::from_proto(req.error_origin.as_ref());
        let origin_proto = origin_key.to_proto();
        let code = if req.error_code == 0 {
            PjrtErrorCode::Internal as i32
        } else {
            req.error_code
        };
        let mut guard = self.inner.lock();
        for info in guard.tasks.values_mut() {
            info.state = CoordinatedTaskStateProto::TaskstateError as i32;
            info.error_code = code;
            info.error_message.clone_from(&req.error_message);
            info.error_payload = Some(CoordinationServiceErrorProto {
                is_reported_error: true,
                source_task: Some(origin_proto.clone()),
            });
        }
        guard.service_error = Some((code, req.error_message, origin_proto));
        guard.job_state_version = guard.job_state_version.saturating_add(1);
        drop(guard);
        self.cv.notify_all();
        ReportErrorToServiceResponseProto {}
    }

    /// Handles `GetTaskState`: returns [`CoordinatedTaskStateInfoProto`] for each requested task.
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetTaskStateRequestProto`] payload.
    fn handle_get_task_state(&self, req: &GetTaskStateRequestProto) -> GetTaskStateResponseProto {
        let guard = self.inner.lock();
        let states = req
            .source_task
            .iter()
            .map(|t| {
                let k = TaskKey::from_proto(Some(t));
                guard
                    .tasks
                    .get(&k)
                    .cloned()
                    .unwrap_or_else(|| CoordinatedTaskStateInfoProto {
                        task: Some(k.to_proto()),
                        incarnation: 0,
                        state: CoordinatedTaskStateProto::TaskstateUninitialized as i32,
                        error_code: 0,
                        error_message: String::new(),
                        error_payload: None,
                    })
            })
            .collect();
        GetTaskStateResponseProto { task_state: states }
    }

    /// Handles `WatchJobState`: returns all task states matching `job_name` (or all tasks if empty)
    /// along with the current monotonic `version_number`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`WatchJobStateRequestProto`] payload.
    fn handle_watch_job_state(
        &self,
        req: &WatchJobStateRequestProto,
    ) -> WatchJobStateResponseProto {
        let guard = self.inner.lock();
        let states = guard
            .tasks
            .iter()
            .filter(|(k, _)| req.job_name.is_empty() || k.job_name == req.job_name)
            .map(|(_, v)| v.clone())
            .collect();
        let ver = if guard.job_state_version <= req.version_number {
            req.version_number.saturating_add(1)
        } else {
            guard.job_state_version
        };
        WatchJobStateResponseProto {
            task_state: states,
            version_number: ver,
        }
    }

    /// Handles `InsertKeyValue`: inserts `kv` and wakes blocked `GetKeyValue` callers, rejecting
    /// duplicate keys with `AlreadyExists` when `!allow_overwrite`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`InsertKeyValueRequestProto`] payload.
    fn handle_insert_key_value(
        &self,
        req: InsertKeyValueRequestProto,
    ) -> Result<InsertKeyValueResponseProto, LibTpuError> {
        let Some(kv) = req.kv else {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "InsertKeyValue",
                task_id: 0,
                status_code: PjrtErrorCode::InvalidArgument as i32,
                message: String::from("InsertKeyValue missing KeyValueEntry"),
            });
        };
        let mut guard = self.inner.lock();
        if !req.allow_overwrite && guard.kv_store.contains_key(&kv.key) {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "InsertKeyValue",
                task_id: 0,
                status_code: PjrtErrorCode::AlreadyExists as i32,
                message: format!("Key '{}' already exists in CoordinationService", kv.key),
            });
        }
        guard.kv_store.insert(kv.key, kv.value);
        drop(guard);
        self.cv.notify_all();
        Ok(InsertKeyValueResponseProto {})
    }

    /// Handles `GetKeyValue`: waits up to `timeout` for `req.key` to appear, returning `NotFound` if absent.
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetKeyValueRequestProto`] payload.
    /// * `timeout` - Maximum wait duration for `req.key`.
    fn handle_get_key_value(
        &self,
        req: &GetKeyValueRequestProto,
        timeout: Duration,
    ) -> Result<GetKeyValueResponseProto, LibTpuError> {
        let found = {
            let mut guard = self.inner.lock();
            let deadline = Instant::now() + timeout;
            while !guard.kv_store.contains_key(&req.key) {
                if self.cv.wait_until(&mut guard, deadline).timed_out() {
                    break;
                }
            }
            guard.kv_store.get(&req.key).cloned()
        };
        let Some(val) = found else {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "GetKeyValue",
                task_id: 0,
                status_code: PjrtErrorCode::NotFound as i32,
                message: format!("Key '{}' not found in CoordinationService", req.key),
            });
        };
        Ok(GetKeyValueResponseProto {
            kv: Some(KeyValueEntryProto {
                key: req.key.clone(),
                value: val,
            }),
        })
    }

    /// Handles `TryGetKeyValue`: immediately returns `req.key` or `NotFound` without blocking.
    ///
    /// # Arguments
    ///
    /// * `req` - [`TryGetKeyValueRequestProto`] payload.
    fn handle_try_get_key_value(
        &self,
        req: &TryGetKeyValueRequestProto,
    ) -> Result<TryGetKeyValueResponseProto, LibTpuError> {
        let found = self.inner.lock().kv_store.get(&req.key).cloned();
        let Some(val) = found else {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "TryGetKeyValue",
                task_id: 0,
                status_code: PjrtErrorCode::NotFound as i32,
                message: format!("Key '{}' not present", req.key),
            });
        };
        Ok(TryGetKeyValueResponseProto {
            kv: Some(KeyValueEntryProto {
                key: req.key.clone(),
                value: val,
            }),
        })
    }

    /// Handles `IncrementKeyValue`: parses existing ASCII decimal value (or `0`), adds `increment`,
    /// stores the updated decimal string, and returns the updated [`KeyValueEntryProto`].
    ///
    /// # Arguments
    ///
    /// * `req` - [`IncrementKeyValueRequestProto`] payload.
    fn handle_increment_key_value(
        &self,
        req: &IncrementKeyValueRequestProto,
    ) -> Result<IncrementKeyValueResponseProto, LibTpuError> {
        let mut guard = self.inner.lock();
        let cur = match guard.kv_store.get(&req.key) {
            Some(bytes) => {
                let s = std::str::from_utf8(bytes).map_err(|_| {
                    LibTpuError::CoordinationServiceError {
                        rpc_method: "IncrementKeyValue",
                        task_id: 0,
                        status_code: PjrtErrorCode::FailedPrecondition as i32,
                        message: format!("Key '{}' is not valid UTF-8 integer", req.key),
                    }
                })?;
                s.parse::<i64>()
                    .map_err(|_| LibTpuError::CoordinationServiceError {
                        rpc_method: "IncrementKeyValue",
                        task_id: 0,
                        status_code: PjrtErrorCode::FailedPrecondition as i32,
                        message: format!(
                            "Key '{}' value '{s}' is not a signed 64-bit integer",
                            req.key
                        ),
                    })?
            }
            None => 0_i64,
        };
        let next = cur.saturating_add(req.increment);
        let encoded = next.to_string().into_bytes();
        guard.kv_store.insert(req.key.clone(), encoded.clone());
        drop(guard);
        self.cv.notify_all();
        Ok(IncrementKeyValueResponseProto {
            kv: Some(KeyValueEntryProto {
                key: req.key.clone(),
                value: encoded,
            }),
        })
    }

    /// Handles `GetKeyValueDir`: returns all KV entries matching `directory_key` prefix.
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetKeyValueDirRequestProto`] payload.
    fn handle_get_key_value_dir(
        &self,
        req: &GetKeyValueDirRequestProto,
    ) -> GetKeyValueDirResponseProto {
        let entries = self
            .inner
            .lock()
            .kv_store
            .iter()
            .filter(|(k, _)| k.starts_with(&req.directory_key))
            .map(|(k, v)| KeyValueEntryProto {
                key: k.clone(),
                value: v.clone(),
            })
            .collect();
        GetKeyValueDirResponseProto {
            directory_key: req.directory_key.clone(),
            kv: entries,
        }
    }

    /// Handles `DeleteKeyValue`: deletes `req.key` or all keys sharing prefix `req.key` when `is_directory`.
    ///
    /// # Arguments
    ///
    /// * `req` - [`DeleteKeyValueRequestProto`] payload.
    fn handle_delete_key_value(
        &self,
        req: &DeleteKeyValueRequestProto,
    ) -> DeleteKeyValueResponseProto {
        {
            let mut guard = self.inner.lock();
            if req.is_directory {
                guard.kv_store.retain(|k, _| !k.starts_with(&req.key));
            } else {
                guard.kv_store.remove(&req.key);
            }
        }
        DeleteKeyValueResponseProto {}
    }

    /// Handles `Barrier`: blocks the calling task at `(barrier_id, counter)` until all participating
    /// tasks arrive, the barrier is cancelled via `CancelBarrier`, or `barrier_timeout_in_ms` expires.
    ///
    /// # Arguments
    ///
    /// * `req` - [`BarrierRequestProto`] payload.
    fn handle_barrier(
        &self,
        req: &BarrierRequestProto,
    ) -> Result<BarrierResponseProto, LibTpuError> {
        let src_key = TaskKey::from_proto(req.source_task.as_ref());
        let target_count = if req.tasks.is_empty() {
            self.inner.lock().expected_num_tasks
        } else {
            req.tasks.len()
        };
        let timeout_ms = if req.barrier_timeout_in_ms <= 0 {
            2_000_u64
        } else {
            u64::try_from(req.barrier_timeout_in_ms).unwrap_or(2_000)
        };
        let bkey = (req.barrier_id.clone(), req.counter);

        let mut guard = self.inner.lock();
        {
            let slot = guard.barriers.entry(bkey.clone()).or_default();
            if slot.cancelled {
                drop(guard);
                return Err(LibTpuError::CoordinationServiceError {
                    rpc_method: "Barrier",
                    task_id: src_key.task_id,
                    status_code: PjrtErrorCode::Cancelled as i32,
                    message: format!("Barrier '{}' (counter={}) was cancelled", bkey.0, bkey.1),
                });
            }
            slot.arrived_tasks.insert(src_key.clone());
        }
        self.cv.notify_all();

        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            if let Some((code, ref msg, _)) = guard.service_error {
                let err_msg = msg.clone();
                drop(guard);
                return Err(LibTpuError::CoordinationServiceError {
                    rpc_method: "Barrier",
                    task_id: src_key.task_id,
                    status_code: code,
                    message: err_msg,
                });
            }
            let (arrived, cancelled) = guard
                .barriers
                .get(&bkey)
                .map_or((0, true), |s| (s.arrived_tasks.len(), s.cancelled));
            if cancelled {
                drop(guard);
                return Err(LibTpuError::CoordinationServiceError {
                    rpc_method: "Barrier",
                    task_id: src_key.task_id,
                    status_code: PjrtErrorCode::Cancelled as i32,
                    message: format!("Barrier '{}' cancelled while waiting", bkey.0),
                });
            }
            if arrived >= target_count {
                drop(guard);
                return Ok(BarrierResponseProto {
                    counter: req.counter,
                });
            }
            if self.cv.wait_until(&mut guard, deadline).timed_out() {
                drop(guard);
                return Err(LibTpuError::CoordinationServiceError {
                    rpc_method: "Barrier",
                    task_id: src_key.task_id,
                    status_code: PjrtErrorCode::DeadlineExceeded as i32,
                    message: format!(
                        "Barrier '{}' timed out ({arrived}/{target_count} arrived)",
                        bkey.0
                    ),
                });
            }
        }
    }

    /// Handles `CancelBarrier`: marks `(barrier_id, counter)` cancelled and wakes any blocked waiters.
    ///
    /// # Arguments
    ///
    /// * `req` - [`CancelBarrierRequestProto`] payload.
    fn handle_cancel_barrier(
        &self,
        req: &CancelBarrierRequestProto,
    ) -> Result<CancelBarrierResponseProto, LibTpuError> {
        if req.barrier_id.is_empty() {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "CancelBarrier",
                task_id: 0,
                status_code: PjrtErrorCode::InvalidArgument as i32,
                message: String::from("CancelBarrier requires non-empty barrier_id"),
            });
        }
        let mut guard = self.inner.lock();
        let slot = guard
            .barriers
            .entry((req.barrier_id.clone(), req.counter))
            .or_default();
        slot.cancelled = true;
        drop(guard);
        self.cv.notify_all();
        Ok(CancelBarrierResponseProto {})
    }

    /// Handles `GetAliveTasks`: returns all currently `TASKSTATE_CONNECTED` tasks among `req.tasks`
    /// (or across the whole cluster when `req.tasks` is empty) and their `fixed64` incarnations.
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetAliveTasksRequestProto`] payload.
    fn handle_get_alive_tasks(
        &self,
        req: &GetAliveTasksRequestProto,
    ) -> GetAliveTasksResponseProto {
        let filter_keys: BTreeSet<TaskKey> = req
            .tasks
            .iter()
            .map(|t| TaskKey::from_proto(Some(t)))
            .collect();
        let mut alive_tasks = Vec::new();
        let mut incarnations = Vec::new();
        {
            let guard = self.inner.lock();
            for (k, info) in &guard.tasks {
                if info.state == CoordinatedTaskStateProto::TaskstateConnected as i32
                    && (filter_keys.is_empty() || filter_keys.contains(k))
                {
                    alive_tasks.push(k.to_proto());
                    incarnations.push(info.incarnation);
                }
            }
        }
        GetAliveTasksResponseProto {
            alive_tasks,
            incarnations,
        }
    }

    /// Handles `PollForError`: returns `Ok` if both the cluster and calling task are healthy, or
    /// returns the active service/task error status code and message.
    ///
    /// # Arguments
    ///
    /// * `req` - [`PollForErrorRequestProto`] payload.
    fn handle_poll_for_error(
        &self,
        req: &PollForErrorRequestProto,
    ) -> Result<PollForErrorResponseProto, LibTpuError> {
        let key = TaskKey::from_proto(req.source_task.as_ref());
        let active_err = {
            let guard = self.inner.lock();
            if let Some((code, ref msg, _)) = guard.service_error {
                Some((code, msg.clone()))
            } else if let Some(info) = guard.tasks.get(&key)
                && info.state == CoordinatedTaskStateProto::TaskstateError as i32
            {
                Some((info.error_code, info.error_message.clone()))
            } else {
                None
            }
        };
        if let Some((status_code, message)) = active_err {
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method: "PollForError",
                task_id: key.task_id,
                status_code,
                message,
            });
        }
        Ok(PollForErrorResponseProto {})
    }
}

/// Serves a single HTTP/2 unary gRPC connection on `stream` against `state`.
///
/// Validates the 24-byte HTTP/2 connection preface, exchanges `SETTINGS` + `SETTINGS_ACK`,
/// reads request `HEADERS` and `DATA` frames on stream `1`, invokes [`CoordinationServiceState::dispatch_unary_rpc`],
/// and writes response `HEADERS` (`:status = 200`), `DATA` (5-byte gRPC envelope), and trailing
/// `HEADERS` (`grpc-status`, `grpc-message`).
///
/// # Arguments
///
/// * `stream` - Bidirectional client/server stream.
/// * `state` - Shared [`CoordinationServiceState`].
///
/// # Errors
///
/// Returns [`LibTpuError`] if the HTTP/2 preface, frame headers, HPACK headers, or gRPC framing is malformed.
pub fn serve_http2_unary_stream(
    stream: &mut TcpStream,
    state: &CoordinationServiceState,
) -> LibTpuResult<()> {
    let mut preface = [0_u8; 24];
    stream.read_exact(&mut preface)?;
    if &preface != HTTP2_CLIENT_PREFACE {
        return Err(LibTpuError::DeserializationFailed {
            reason: String::from("Invalid HTTP/2 client connection preface"),
        });
    }

    // Send server empty SETTINGS frame
    Http2Frame {
        frame_type: Http2FrameType::Settings,
        flags: 0,
        stream_id: 0,
        payload: Vec::new(),
    }
    .write_to(stream)?;

    let mut rpc_stream_id: Option<u32> = None;
    let mut req_headers = BTreeMap::new();
    let mut req_data = Vec::new();
    loop {
        let frame = Http2Frame::read_from(stream)?;
        match frame.frame_type {
            Http2FrameType::Settings => {
                if (frame.flags & HTTP2_FLAG_ACK) == 0 {
                    Http2Frame {
                        frame_type: Http2FrameType::Settings,
                        flags: HTTP2_FLAG_ACK,
                        stream_id: 0,
                        payload: Vec::new(),
                    }
                    .write_to(stream)?;
                }
            }
            Http2FrameType::WindowUpdate => {}
            Http2FrameType::Headers => {
                let _ = rpc_stream_id.insert(frame.stream_id);
                req_headers = HpackCodec::decode_headers(&frame.payload)?;
            }
            Http2FrameType::Data => {
                let _ = rpc_stream_id.insert(frame.stream_id);
                req_data.extend_from_slice(&frame.payload);
                if (frame.flags & HTTP2_FLAG_END_STREAM) != 0 {
                    break;
                }
            }
        }
    }
    let rpc_stream_id = rpc_stream_id.unwrap_or(1_u32);

    let path = req_headers.get(":path").map_or("", String::as_str);
    let compress_resp = req_headers
        .get("grpc-encoding")
        .is_some_and(|enc| enc == "brotli");
    let req_proto = GrpcProtoEnvelope::unframe(&req_data)?;
    let (grpc_status, grpc_msg, resp_proto) = state.dispatch_unary_rpc(path, &req_proto);

    let init_hdrs = HpackCodec::encode_headers(&[
        (":status", "200"),
        ("content-type", "application/grpc+proto"),
    ]);
    Http2Frame {
        frame_type: Http2FrameType::Headers,
        flags: HTTP2_FLAG_END_HEADERS,
        stream_id: rpc_stream_id,
        payload: init_hdrs,
    }
    .write_to(stream)?;

    if grpc_status == 0 {
        let framed_body = GrpcProtoEnvelope::frame(&resp_proto, compress_resp);
        Http2Frame {
            frame_type: Http2FrameType::Data,
            flags: 0,
            stream_id: rpc_stream_id,
            payload: framed_body,
        }
        .write_to(stream)?;
    }

    let status_str = grpc_status.to_string();
    let trailers =
        HpackCodec::encode_headers(&[("grpc-status", &status_str), ("grpc-message", &grpc_msg)]);
    Http2Frame {
        frame_type: Http2FrameType::Headers,
        flags: HTTP2_FLAG_END_HEADERS | HTTP2_FLAG_END_STREAM,
        stream_id: rpc_stream_id,
        payload: trailers,
    }
    .write_to(stream)?;
    stream.flush()?;
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let _ = stream.read_to_end(&mut Vec::new());
    Ok(())
}

/// Concurrent HTTP/2 + Protobuf gRPC server for `tensorflow.CoordinationService`.
#[derive(Debug)]
pub struct CoordinationGrpcServer {
    /// Bound local TCP socket address.
    local_addr: SocketAddr,
    /// Shared [`CoordinationServiceState`].
    pub state: Arc<CoordinationServiceState>,
    /// Shutdown flag signaled on drop or [`Self::shutdown`].
    shutdown_flag: Arc<AtomicBool>,
    /// Background accept-loop thread handle.
    accept_thread: Option<std::thread::JoinHandle<()>>,
}

impl CoordinationGrpcServer {
    /// Runs the TCP accept loop until `stop_flag` is set or `listener.accept()` returns an error.
    ///
    /// # Arguments
    ///
    /// * `listener` - Bound TCP listener socket.
    /// * `state` - Shared [`CoordinationServiceState`] instance.
    /// * `stop_flag` - Atomic shutdown signal flag.
    pub(crate) fn run_accept_loop(
        listener: &TcpListener,
        state: &Arc<CoordinationServiceState>,
        stop_flag: &AtomicBool,
    ) {
        while let Ok((mut stream, _)) = listener.accept() {
            if stop_flag.load(Ordering::Acquire) {
                break;
            }
            let st_conn = Arc::clone(state);
            std::thread::spawn(move || {
                let _ = serve_http2_unary_stream(&mut stream, &st_conn);
            });
        }
    }

    /// Binds a live HTTP/2 gRPC `CoordinationService` server on `bind_addr` (e.g., `"127.0.0.1:0"`)
    /// and starts a concurrent connection-dispatch accept loop supporting blocking multi-task barriers.
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Local socket address string to bind.
    /// * `expected_num_tasks` - Total cluster tasks expected by `WaitForAllTasks` and barriers.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if binding the TCP listener or querying its address fails.
    pub fn start(bind_addr: &str, expected_num_tasks: usize) -> LibTpuResult<Self> {
        let listener = TcpListener::bind(bind_addr)?;
        let local_addr = listener.local_addr()?;
        let state = Arc::new(CoordinationServiceState::new(
            expected_num_tasks,
            0xC00D_2026_0000_0001,
        ));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let st_clone = Arc::clone(&state);
        let stop_clone = Arc::clone(&shutdown_flag);

        let handle = std::thread::spawn(move || {
            Self::run_accept_loop(&listener, &st_clone, &stop_clone);
        });

        Ok(Self {
            local_addr,
            state,
            shutdown_flag,
            accept_thread: Some(handle),
        })
    }

    /// Returns the bound local [`SocketAddr`] of this gRPC server.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Stops the background accept loop cleanly.
    pub fn shutdown(&mut self) {
        if let Some(handle) = self.accept_thread.take() {
            self.shutdown_flag.store(true, Ordering::Release);
            let _ = TcpStream::connect(self.local_addr);
            let _ = handle.join();
        }
    }
}

impl Drop for CoordinationGrpcServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Wire-compatible HTTP/2 + Protobuf gRPC client for all 19 `tensorflow.CoordinationService` RPCs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinationGrpcClient {
    /// Target gRPC server socket address.
    pub endpoint: SocketAddr,
    /// Whether to Brotli-compress outgoing/incoming gRPC `DATA` frames (`grpc-encoding: brotli`).
    pub use_brotli_compression: bool,
}

impl CoordinationGrpcClient {
    /// Creates a new [`CoordinationGrpcClient`] targeting `endpoint`.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - Bound [`SocketAddr`] of a [`CoordinationGrpcServer`].
    #[must_use]
    pub const fn new(endpoint: SocketAddr) -> Self {
        Self {
            endpoint,
            use_brotli_compression: false,
        }
    }

    /// Enables or disables Brotli gRPC payload compression (`grpc-encoding: brotli`).
    ///
    /// # Arguments
    ///
    /// * `enabled` - `true` to compress gRPC `DATA` envelopes with Brotli.
    #[must_use]
    pub const fn with_brotli_compression(mut self, enabled: bool) -> Self {
        self.use_brotli_compression = enabled;
        self
    }

    /// Executes a unary HTTP/2 + Protobuf gRPC call on `/tensorflow.CoordinationService/<rpc_method>`.
    ///
    /// # Arguments
    ///
    /// * `rpc_method` - Static `CoordinationService` method name (e.g., `"RegisterTask"`).
    /// * `task_id` - Originating worker task index attached to any returned [`LibTpuError`].
    /// * `req` - Typed Protobuf request message.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP/HTTP/2 transport fails, `grpc-status != 0`, or Protobuf decoding fails.
    pub fn invoke_unary<Req, Resp>(
        &self,
        rpc_method: &'static str,
        task_id: i32,
        req: &Req,
    ) -> LibTpuResult<Resp>
    where
        Req: ProtoSchemaMessage,
        Resp: ProtoSchemaMessage,
    {
        let raw_resp = self.invoke_raw_http2_rpc(rpc_method, task_id, &req.encode_proto())?;
        Resp::decode_proto(&raw_resp)
    }

    /// Sends the HTTP/2 preface, `SETTINGS`, `WINDOW_UPDATE`, request `HEADERS`, and `DATA(END_STREAM)`.
    ///
    /// # Arguments
    ///
    /// * `stream` - Connected TCP stream to the gRPC server.
    /// * `rpc_method` - Static `CoordinationService` RPC name.
    /// * `req_proto_bytes` - Serialized request Protobuf message bytes.
    fn write_http2_unary_request(
        &self,
        stream: &mut TcpStream,
        rpc_method: &'static str,
        req_proto_bytes: &[u8],
    ) -> LibTpuResult<()> {
        stream.write_all(HTTP2_CLIENT_PREFACE)?;
        Http2Frame {
            frame_type: Http2FrameType::Settings,
            flags: 0,
            stream_id: 0,
            payload: Vec::new(),
        }
        .write_to(stream)?;
        Http2Frame {
            frame_type: Http2FrameType::WindowUpdate,
            flags: 0,
            stream_id: 0,
            payload: 1_048_576_u32.to_be_bytes().to_vec(),
        }
        .write_to(stream)?;

        let full_path = format!("/tensorflow.CoordinationService/{rpc_method}");
        let hdr_pairs: Vec<(&str, &str)> = if self.use_brotli_compression {
            vec![
                (":method", "POST"),
                (":scheme", "http"),
                (":path", &full_path),
                ("content-type", "application/grpc+proto"),
                ("te", "trailers"),
                ("grpc-encoding", "brotli"),
            ]
        } else {
            vec![
                (":method", "POST"),
                (":scheme", "http"),
                (":path", &full_path),
                ("content-type", "application/grpc+proto"),
                ("te", "trailers"),
            ]
        };
        Http2Frame {
            frame_type: Http2FrameType::Headers,
            flags: HTTP2_FLAG_END_HEADERS,
            stream_id: 1,
            payload: HpackCodec::encode_headers(&hdr_pairs),
        }
        .write_to(stream)?;
        Http2Frame {
            frame_type: Http2FrameType::Data,
            flags: HTTP2_FLAG_END_STREAM,
            stream_id: 1,
            payload: GrpcProtoEnvelope::frame(req_proto_bytes, self.use_brotli_compression),
        }
        .write_to(stream)?;
        stream.flush()?;
        Ok(())
    }

    /// Performs the concrete non-generic HTTP/2 connection preface, `SETTINGS`, `HEADERS`, `DATA`,
    /// and response/trailer exchange over TCP so all 19 RPC methods share a single transport body.
    ///
    /// # Arguments
    ///
    /// * `rpc_method` - Static `CoordinationService` RPC name.
    /// * `task_id` - Originating worker task index attached to any returned [`LibTpuError`].
    /// * `req_proto_bytes` - Serialized request Protobuf message bytes.
    fn invoke_raw_http2_rpc(
        &self,
        rpc_method: &'static str,
        task_id: i32,
        req_proto_bytes: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        let mut stream = TcpStream::connect(self.endpoint)?;
        self.write_http2_unary_request(&mut stream, rpc_method, req_proto_bytes)?;

        let mut resp_data = Vec::new();
        let mut trailers = BTreeMap::new();
        let mut saw_initial_headers = false;
        loop {
            let frame = Http2Frame::read_from(&mut stream)?;
            match frame.frame_type {
                Http2FrameType::Settings => {
                    if (frame.flags & HTTP2_FLAG_ACK) == 0 {
                        Http2Frame {
                            frame_type: Http2FrameType::Settings,
                            flags: HTTP2_FLAG_ACK,
                            stream_id: 0,
                            payload: Vec::new(),
                        }
                        .write_to(&mut stream)?;
                    }
                }
                Http2FrameType::WindowUpdate => {}
                Http2FrameType::Headers => {
                    let decoded = HpackCodec::decode_headers(&frame.payload)?;
                    if saw_initial_headers {
                        trailers = decoded;
                    } else {
                        saw_initial_headers = true;
                    }
                    if (frame.flags & HTTP2_FLAG_END_STREAM) != 0 {
                        break;
                    }
                }
                Http2FrameType::Data => {
                    resp_data.extend_from_slice(&frame.payload);
                    if (frame.flags & HTTP2_FLAG_END_STREAM) != 0 {
                        break;
                    }
                }
            }
        }

        let status_code = trailers
            .get("grpc-status")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        if status_code != 0 {
            let message = trailers.remove("grpc-message").unwrap_or_default();
            return Err(LibTpuError::CoordinationServiceError {
                rpc_method,
                task_id,
                status_code,
                message,
            });
        }
        if resp_data.is_empty() {
            return Ok(Vec::new());
        }
        GrpcProtoEnvelope::unframe(&resp_data)
    }

    /// Invokes `RegisterTask` (`tensorflow.CoordinationService/RegisterTask`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`RegisterTaskRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the RPC fails or the task is already connected with another incarnation.
    pub fn register_task(
        &self,
        req: &RegisterTaskRequestProto,
    ) -> LibTpuResult<RegisterTaskResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("RegisterTask", tid, req)
    }

    /// Invokes `WaitForAllTasks` (`tensorflow.CoordinationService/WaitForAllTasks`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`WaitForAllTasksRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if waiting times out or any cluster task reports an error.
    pub fn wait_for_all_tasks(
        &self,
        req: &WaitForAllTasksRequestProto,
    ) -> LibTpuResult<WaitForAllTasksResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("WaitForAllTasks", tid, req)
    }

    /// Invokes `ShutdownTask` (`tensorflow.CoordinationService/ShutdownTask`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`ShutdownTaskRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the task is unregistered or transport fails.
    pub fn shutdown_task(
        &self,
        req: &ShutdownTaskRequestProto,
    ) -> LibTpuResult<ShutdownTaskResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("ShutdownTask", tid, req)
    }

    /// Invokes `ResetTask` (`tensorflow.CoordinationService/ResetTask`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`ResetTaskRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn reset_task(&self, req: &ResetTaskRequestProto) -> LibTpuResult<ResetTaskResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("ResetTask", tid, req)
    }

    /// Invokes `Heartbeat` (`tensorflow.CoordinationService/Heartbeat`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`HeartbeatRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the task is unregistered or `incarnation` mismatches.
    pub fn heartbeat(&self, req: &HeartbeatRequestProto) -> LibTpuResult<HeartbeatResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("Heartbeat", tid, req)
    }

    /// Invokes `ReportErrorToTask` (`tensorflow.CoordinationService/ReportErrorToTask`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`ReportErrorToTaskRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn report_error_to_task(
        &self,
        req: &ReportErrorToTaskRequestProto,
    ) -> LibTpuResult<ReportErrorToTaskResponseProto> {
        self.invoke_unary("ReportErrorToTask", 0, req)
    }

    /// Invokes `ReportErrorToService` (`tensorflow.CoordinationService/ReportErrorToService`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`ReportErrorToServiceRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn report_error_to_service(
        &self,
        req: &ReportErrorToServiceRequestProto,
    ) -> LibTpuResult<ReportErrorToServiceResponseProto> {
        let tid = req.error_origin.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("ReportErrorToService", tid, req)
    }

    /// Invokes `GetTaskState` (`tensorflow.CoordinationService/GetTaskState`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetTaskStateRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn get_task_state(
        &self,
        req: &GetTaskStateRequestProto,
    ) -> LibTpuResult<GetTaskStateResponseProto> {
        self.invoke_unary("GetTaskState", 0, req)
    }

    /// Invokes `WatchJobState` (`tensorflow.CoordinationService/WatchJobState`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`WatchJobStateRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn watch_job_state(
        &self,
        req: &WatchJobStateRequestProto,
    ) -> LibTpuResult<WatchJobStateResponseProto> {
        self.invoke_unary("WatchJobState", 0, req)
    }

    /// Invokes `InsertKeyValue` (`tensorflow.CoordinationService/InsertKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`InsertKeyValueRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `kv` is missing or the key already exists when `!allow_overwrite`.
    pub fn insert_key_value(
        &self,
        req: &InsertKeyValueRequestProto,
    ) -> LibTpuResult<InsertKeyValueResponseProto> {
        self.invoke_unary("InsertKeyValue", 0, req)
    }

    /// Invokes `GetKeyValue` (`tensorflow.CoordinationService/GetKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetKeyValueRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the key is not found before timeout.
    pub fn get_key_value(
        &self,
        req: &GetKeyValueRequestProto,
    ) -> LibTpuResult<GetKeyValueResponseProto> {
        self.invoke_unary("GetKeyValue", 0, req)
    }

    /// Invokes `TryGetKeyValue` (`tensorflow.CoordinationService/TryGetKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`TryGetKeyValueRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] (`NotFound`) if `req.key` is absent.
    pub fn try_get_key_value(
        &self,
        req: &TryGetKeyValueRequestProto,
    ) -> LibTpuResult<TryGetKeyValueResponseProto> {
        self.invoke_unary("TryGetKeyValue", 0, req)
    }

    /// Invokes `IncrementKeyValue` (`tensorflow.CoordinationService/IncrementKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`IncrementKeyValueRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the existing value is not a valid signed 64-bit integer.
    pub fn increment_key_value(
        &self,
        req: &IncrementKeyValueRequestProto,
    ) -> LibTpuResult<IncrementKeyValueResponseProto> {
        self.invoke_unary("IncrementKeyValue", 0, req)
    }

    /// Invokes `GetKeyValueDir` (`tensorflow.CoordinationService/GetKeyValueDir`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetKeyValueDirRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn get_key_value_dir(
        &self,
        req: &GetKeyValueDirRequestProto,
    ) -> LibTpuResult<GetKeyValueDirResponseProto> {
        self.invoke_unary("GetKeyValueDir", 0, req)
    }

    /// Invokes `DeleteKeyValue` (`tensorflow.CoordinationService/DeleteKeyValue`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`DeleteKeyValueRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn delete_key_value(
        &self,
        req: &DeleteKeyValueRequestProto,
    ) -> LibTpuResult<DeleteKeyValueResponseProto> {
        self.invoke_unary("DeleteKeyValue", 0, req)
    }

    /// Invokes `Barrier` (`tensorflow.CoordinationService/Barrier`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`BarrierRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the barrier times out, is cancelled, or encounters a cluster error.
    pub fn barrier(&self, req: &BarrierRequestProto) -> LibTpuResult<BarrierResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("Barrier", tid, req)
    }

    /// Invokes `CancelBarrier` (`tensorflow.CoordinationService/CancelBarrier`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`CancelBarrierRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `barrier_id` is empty or transport fails.
    pub fn cancel_barrier(
        &self,
        req: &CancelBarrierRequestProto,
    ) -> LibTpuResult<CancelBarrierResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("CancelBarrier", tid, req)
    }

    /// Invokes `GetAliveTasks` (`tensorflow.CoordinationService/GetAliveTasks`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`GetAliveTasksRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] on HTTP/2 transport failure.
    pub fn get_alive_tasks(
        &self,
        req: &GetAliveTasksRequestProto,
    ) -> LibTpuResult<GetAliveTasksResponseProto> {
        let tid = req.requesting_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("GetAliveTasks", tid, req)
    }

    /// Invokes `PollForError` (`tensorflow.CoordinationService/PollForError`).
    ///
    /// # Arguments
    ///
    /// * `req` - [`PollForErrorRequestProto`] payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the task or service is in an error state.
    pub fn poll_for_error(
        &self,
        req: &PollForErrorRequestProto,
    ) -> LibTpuResult<PollForErrorResponseProto> {
        let tid = req.source_task.as_ref().map_or(0, |t| t.task_id);
        self.invoke_unary("PollForError", tid, req)
    }
}

/// `OpenXLA` (`xla/pjrt/distributed/`) multi-host `LocalTopologyProto` -> `GlobalTopologyProto`
/// cluster topology discovery handshake over HTTP/2 gRPC [`CoordinationGrpcClient`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MultiHostTopologyHandshake;

impl MultiHostTopologyHandshake {
    /// Executes the distributed PJRT topology exchange for `local_topology`.
    ///
    /// Performs the 4-stage `OpenXLA` coordination protocol:
    /// 1. `RegisterTask` & `WaitForAllTasks`
    /// 2. `InsertKeyValue("pjrt_topology/local/<process_id>", local_topology.encode())`
    /// 3. `Barrier("pjrt_topology_local_published")`
    /// 4. Leader (`process_id == 0`) reads `GetKeyValueDir("pjrt_topology/local/")`, assembles
    ///    [`GlobalTopologyProto`] via [`MultiSliceTopologyAssembler::assemble_global_topology`], and
    ///    publishes `"pjrt_topology/global"`; non-leaders await `"pjrt_topology/global"` via `GetKeyValue`
    ///    after `Barrier("pjrt_topology_global_ready")`.
    ///
    /// # Arguments
    ///
    /// * `client` - Connected [`CoordinationGrpcClient`].
    /// * `local_topology` - This host's [`LocalTopologyProto`].
    /// * `incarnation` - Worker process incarnation ID.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if registration, barrier synchronization, or topology decoding fails.
    pub fn exchange_cluster_topology(
        client: &CoordinationGrpcClient,
        local_topology: &LocalTopologyProto,
        incarnation: u64,
    ) -> LibTpuResult<GlobalTopologyProto> {
        let task = CoordinatedTaskProto {
            job_name: String::from("jax_worker"),
            task_id: local_topology.process_id,
            recoverable: false,
        };
        let _ = client.register_task(&RegisterTaskRequestProto {
            incarnation,
            source_task: Some(task.clone()),
        })?;
        let _ = client.wait_for_all_tasks(&WaitForAllTasksRequestProto {
            source_task: Some(task.clone()),
            device_info: None,
        })?;

        let local_key = format!("pjrt_topology/local/{:04}", local_topology.process_id);
        let _ = client.insert_key_value(&InsertKeyValueRequestProto {
            kv: Some(KeyValueEntryProto {
                key: local_key,
                value: local_topology.encode(),
            }),
            allow_overwrite: true,
        })?;

        let _ = client.barrier(&BarrierRequestProto {
            barrier_id: String::from("pjrt_topology_local_published"),
            barrier_timeout_in_ms: 2_000,
            tasks: Vec::new(),
            source_task: Some(task.clone()),
            counter: 0,
        })?;

        if local_topology.process_id == 0 {
            let dir_resp = client.get_key_value_dir(&GetKeyValueDirRequestProto {
                directory_key: String::from("pjrt_topology/local/"),
            })?;
            let mut locals = Vec::with_capacity(dir_resp.kv.len());
            for entry in &dir_resp.kv {
                locals.push(LocalTopologyProto::decode(&entry.value)?);
            }
            let global = MultiSliceTopologyAssembler::assemble_global_topology(&locals);
            let _ = client.insert_key_value(&InsertKeyValueRequestProto {
                kv: Some(KeyValueEntryProto {
                    key: String::from("pjrt_topology/global"),
                    value: global.encode(),
                }),
                allow_overwrite: true,
            })?;
            let _ = client.barrier(&BarrierRequestProto {
                barrier_id: String::from("pjrt_topology_global_ready"),
                barrier_timeout_in_ms: 2_000,
                tasks: Vec::new(),
                source_task: Some(task),
                counter: 0,
            })?;
            return Ok(global);
        }

        let _ = client.barrier(&BarrierRequestProto {
            barrier_id: String::from("pjrt_topology_global_ready"),
            barrier_timeout_in_ms: 2_000,
            tasks: Vec::new(),
            source_task: Some(task),
            counter: 0,
        })?;
        let kv_resp = client.get_key_value(&GetKeyValueRequestProto {
            key: String::from("pjrt_topology/global"),
        })?;
        let raw_bytes = kv_resp.kv.map(|e| e.value).unwrap_or_default();
        GlobalTopologyProto::decode(&raw_bytes)
    }
}
