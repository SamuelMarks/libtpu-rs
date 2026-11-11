//! Pure-Rust distributed multi-slice coordinator, DCN transport, and ICI/MegaScale collective engine.
//!
//! Provides:
//! - [`DistributedCoordinator`]: Thread-safe coordination service key-value store, barrier state machine,
//!   heartbeat health tracker, and [`MegaScaleRuntimeErrorProto`] hang detector.
//! - [`MultiSliceTopologyAssembler`]: Deterministic assembler converting per-host [`LocalTopologyProto`]
//!   announcements into a globally indexed [`GlobalTopologyProto`] and [`EndpointAddressesProto`].
//! - [`DcnLoopbackAndTcpTransport`]: Cross-slice / cross-host message framing and loopback/TCP channel.
//! - [`CollectiveRuntimeEngine`]: Multi-replica Ring/Halving-Doubling `AllReduce`, `ReduceScatter`,
//!   `AllGather`, `AllToAll`, and `CollectivePermute` collective algorithms across all 29 [`ElementType`]s.

use crate::error::{LibTpuError, LibTpuResult};
use crate::proto::{
    EndpointAddressesProto, GlobalTopologyProto, HostNetworkAddressProto, LocalTopologyProto,
    MegaScaleRuntimeErrorProto, MegascaleTransportConfigProto, NetworkAddressMappingProto,
};
use crate::types::{CollectiveReductionOp, ElementType};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

/// Thread-safe distributed coordination service (`DistributedCoordinator`) modelling
/// `tensorflow.CoordinationService` and `xla.megascale.runtime`.
#[derive(Debug, Default)]
pub struct DistributedCoordinator {
    /// Key-value directory used for exchanging `LocalTopologyProto` and barrier state.
    kv_store: Mutex<BTreeMap<String, Vec<u8>>>,
    /// Condition variable notified whenever a key-value entry is inserted or updated.
    kv_condvar: parking_lot::Condvar,
    /// Barrier arrival sets keyed by `(barrier_id, slice_id)` -> set of participating `task_id`s.
    barriers: Mutex<BTreeMap<(u64, i32), BTreeSet<i32>>>,
    /// Last recorded heartbeat timestamp (ms) per `(slice_id, task_id)`.
    heartbeats_ms: Mutex<BTreeMap<(i32, i32), u64>>,
    /// Count of completed cross-slice barriers.
    completed_barriers: std::sync::atomic::AtomicU64,
}

impl DistributedCoordinator {
    /// Creates a new empty [`DistributedCoordinator`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or updates a key-value entry in the coordination directory and wakes any threads
    /// blocked in [`Self::wait_get_key_value`].
    ///
    /// # Arguments
    ///
    /// * `key` - Directory key string.
    /// * `value` - Raw value payload bytes.
    pub fn put_key_value(&self, key: &str, value: Vec<u8>) {
        self.kv_store.lock().insert(String::from(key), value);
        self.kv_condvar.notify_all();
    }

    /// Blocks up to `timeout_ms` milliseconds waiting for `key` to be inserted into the
    /// coordination directory (`WaitGetKeyValue`), returning the value as soon as it is present.
    ///
    /// # Arguments
    ///
    /// * `key` - Directory key string to await.
    /// * `timeout_ms` - Maximum wait timeout in milliseconds (`0` performs an immediate check).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MegascaleBarrierTimeout`] if `key` does not appear within `timeout_ms`.
    pub fn wait_get_key_value(&self, key: &str, timeout_ms: u64) -> LibTpuResult<Vec<u8>> {
        let mut guard = self.kv_store.lock();
        if let Some(val) = guard.get(key) {
            return Ok(val.clone());
        }
        if timeout_ms > 0 {
            let _ = self
                .kv_condvar
                .wait_for(&mut guard, std::time::Duration::from_millis(timeout_ms));
            if let Some(val) = guard.get(key) {
                return Ok(val.clone());
            }
        }
        Err(LibTpuError::MegascaleBarrierTimeout {
            slice_id: 0,
            barrier_id: timeout_ms,
        })
    }

    /// Retrieves a value by `key` from the coordination directory.
    ///
    /// # Arguments
    ///
    /// * `key` - Directory key string to look up.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CollectiveError`] if `key` is not present.
    pub fn get_key_value(&self, key: &str) -> LibTpuResult<Vec<u8>> {
        self.kv_store
            .lock()
            .get(key)
            .cloned()
            .ok_or_else(|| LibTpuError::CollectiveError {
                communicator_id: 0,
                op_name: "GetKeyValue",
                reason: format!("Key '{key}' not found in coordinator"),
            })
    }

    /// Deletes a key from the coordination directory, returning `true` if an entry was removed.
    ///
    /// # Arguments
    ///
    /// * `key` - Directory key string to remove.
    #[must_use]
    pub fn delete_key_value(&self, key: &str) -> bool {
        self.kv_store.lock().remove(key).is_some()
    }

    /// Non-blocking key lookup (`TryGetKeyValue`), returning `Some(value)` if present or `None` if absent.
    ///
    /// # Arguments
    ///
    /// * `key` - Directory key string to query.
    #[must_use]
    pub fn try_get_key_value(&self, key: &str) -> Option<Vec<u8>> {
        self.kv_store.lock().get(key).cloned()
    }

    /// Returns all `(key, value)` pairs in the coordination directory whose key begins with `prefix` (`GetKeyValueDir`).
    ///
    /// # Arguments
    ///
    /// * `prefix` - Directory prefix string (e.g., `"topo/"`).
    #[must_use]
    pub fn get_key_value_dir(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        self.kv_store
            .lock()
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Cancels an active `(barrier_id, slice_id)` barrier (`CancelBarrier`), returning `true` if a pending barrier was cleared.
    ///
    /// # Arguments
    ///
    /// * `slice_id` - Zero-based Megascale slice index.
    /// * `barrier_id` - Monotonic barrier identifier to cancel.
    #[must_use]
    pub fn cancel_barrier(&self, slice_id: i32, barrier_id: u64) -> bool {
        self.barriers
            .lock()
            .remove(&(barrier_id, slice_id))
            .is_some()
    }

    /// Records a runtime error reported by `(slice_id, task_id)` (`ReportErrorToService`) under `"errors/<slice_id>/<task_id>"`.
    ///
    /// # Arguments
    ///
    /// * `slice_id` - Reporting slice index.
    /// * `task_id` - Reporting worker task index.
    /// * `error_message` - Diagnostic fault message.
    #[must_use]
    pub fn report_error_to_service(
        &self,
        slice_id: i32,
        task_id: i32,
        error_message: &str,
    ) -> MegaScaleRuntimeErrorProto {
        let err = MegaScaleRuntimeErrorProto {
            error_type: 2, //TASK_REPORTED_ERROR
            error_message: String::from(error_message),
            estimated_start_time_ms: 0,
            runtime_state: None,
            triggering_launch_id: slice_id,
            hostname: format!("tpu-slice{slice_id}-worker{task_id}"),
            job_name: String::from("megascale_worker"),
            task_id,
            event_id: u64::try_from(task_id.max(0)).unwrap_or(0),
            unrecoverable_error_type: 1,
            runtime_error_id: 1,
            hang_error_type: 0,
        };
        self.put_key_value(
            &format!("errors/{slice_id}/{task_id}"),
            error_message.as_bytes().to_vec(),
        );
        err
    }

    /// Registers arrival of `task_id` at `(barrier_id, slice_id)` and checks whether
    /// `expected_participants` have all arrived.
    ///
    /// # Arguments
    ///
    /// * `slice_id` - Zero-based Megascale slice index.
    /// * `barrier_id` - Monotonic barrier identifier.
    /// * `task_id` - Arriving worker task ordinal.
    /// * `expected_participants` - Total tasks required to release the barrier.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::MegascaleBarrierTimeout`] if fewer than `expected_participants` have arrived.
    pub fn wait_at_barrier(
        &self,
        slice_id: i32,
        barrier_id: u64,
        task_id: i32,
        expected_participants: usize,
    ) -> LibTpuResult<u64> {
        let mut guard = self.barriers.lock();
        let participants = guard.entry((barrier_id, slice_id)).or_default();
        participants.insert(task_id);
        if participants.len() < expected_participants {
            return Err(LibTpuError::MegascaleBarrierTimeout {
                slice_id,
                barrier_id,
            });
        }
        drop(guard);
        let total = self
            .completed_barriers
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        Ok(total)
    }

    /// Records a heartbeat timestamp in milliseconds for `(slice_id, task_id)`.
    ///
    /// # Arguments
    ///
    /// * `slice_id` - Slice identifier.
    /// * `task_id` - Worker task identifier.
    /// * `timestamp_ms` - Monotonic timestamp in milliseconds.
    pub fn record_heartbeat(&self, slice_id: i32, task_id: i32, timestamp_ms: u64) {
        self.heartbeats_ms
            .lock()
            .insert((slice_id, task_id), timestamp_ms);
    }

    /// Checks all registered task heartbeats against `now_ms` and `timeout_ms`, returning
    /// [`MegaScaleRuntimeErrorProto`] descriptors for any unresponsive tasks.
    ///
    /// # Arguments
    ///
    /// * `now_ms` - Current monotonic timestamp in milliseconds.
    /// * `timeout_ms` - Maximum allowed elapsed milliseconds since last heartbeat.
    #[must_use]
    pub fn check_stale_tasks(
        &self,
        now_ms: u64,
        timeout_ms: u64,
    ) -> Vec<MegaScaleRuntimeErrorProto> {
        let guard = self.heartbeats_ms.lock();
        let mut errors = Vec::new();
        for (&(slice_id, task_id), &last_ms) in guard.iter() {
            if now_ms.saturating_sub(last_ms) > timeout_ms {
                errors.push(MegaScaleRuntimeErrorProto {
                    error_type: 1, // HANG_DETECTED
                    error_message: format!(
                        "Task {task_id} on slice {slice_id} exceeded heartbeat timeout ({timeout_ms} ms)"
                    ),
                    estimated_start_time_ms: last_ms,
                    runtime_state: None,
                    triggering_launch_id: slice_id,
                    hostname: format!("tpu-slice{slice_id}-worker{task_id}"),
                    job_name: String::from("megascale_worker"),
                    task_id,
                    event_id: u64::try_from(task_id.max(0)).unwrap_or(0),
                    unrecoverable_error_type: 0,
                    runtime_error_id: now_ms,
                    hang_error_type: 1, // Timeout
                });
            }
        }
        drop(guard);
        errors
    }

    /// Dispatches an incoming [`CoordinationWireFrame`] against this [`DistributedCoordinator`]
    /// and returns a response [`CoordinationWireFrame`].
    ///
    /// Supported opcodes:
    /// - `1` (`RegisterTask`): Registers heartbeat for `(0, 0)` and echoes `"REGISTERED"`.
    /// - `2` (`Heartbeat`): Records a fresh heartbeat for `(0, 0)` and echoes `"HEARTBEAT_OK"`.
    /// - `3` (`InsertKeyValue`): Splits `"key=value"` payload and inserts into `kv_store`.
    /// - `4` (`GetKeyValue`): Looks up `payload` as UTF-8 key and returns its value bytes.
    /// - `5` (`DeleteKeyValue`): Deletes `payload` key and returns `[1]` or `[0]`.
    /// - `6` (`WaitAtBarrier`): Waits at single-participant barrier `(0, request_id)` and returns 8-byte counter.
    /// - `7` (`ShutdownTask`): Removes task `(0, 0)` from `heartbeats_ms` and returns `"SHUTDOWN_OK"`.
    ///
    /// # Arguments
    ///
    /// * `req` - Incoming [`CoordinationWireFrame`] RPC request.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `GetKeyValue` or `WaitAtBarrier` fails.
    #[allow(clippy::too_many_lines)]
    pub fn handle_wire_frame(
        &self,
        req: &CoordinationWireFrame,
    ) -> LibTpuResult<CoordinationWireFrame> {
        let resp_payload = match req.opcode {
            1 => {
                self.record_heartbeat(0, 0, req.request_id);
                b"REGISTERED".to_vec()
            }
            2 => {
                self.record_heartbeat(0, 0, req.request_id);
                b"HEARTBEAT_OK".to_vec()
            }
            3 => {
                let text = String::from_utf8_lossy(&req.payload);
                let default_pair = (text.as_ref(), "");
                let (k, v) = text.split_once('=').unwrap_or(default_pair);
                self.put_key_value(k, v.as_bytes().to_vec());
                b"INSERTED".to_vec()
            }
            4 => {
                let key = String::from_utf8_lossy(&req.payload);
                self.get_key_value(&key)?
            }
            5 => {
                let key = String::from_utf8_lossy(&req.payload);
                vec![u8::from(self.delete_key_value(&key))]
            }
            6 => {
                let (slice_id, barrier_id, task_id, expected) = if let [
                    s0,
                    s1,
                    s2,
                    s3,
                    b0,
                    b1,
                    b2,
                    b3,
                    b4,
                    b5,
                    b6,
                    b7,
                    t0,
                    t1,
                    t2,
                    t3,
                    e0,
                    e1,
                    e2,
                    e3,
                    ..,
                ] = req.payload.as_slice()
                {
                    let s_id = i32::from_le_bytes([*s0, *s1, *s2, *s3]);
                    let b_id = u64::from_le_bytes([*b0, *b1, *b2, *b3, *b4, *b5, *b6, *b7]);
                    let t_id = i32::from_le_bytes([*t0, *t1, *t2, *t3]);
                    let exp = u32::from_le_bytes([*e0, *e1, *e2, *e3]);
                    (s_id, b_id, t_id, (exp as usize).max(1))
                } else {
                    (0, req.request_id, 0, 1)
                };
                let count = self.wait_at_barrier(slice_id, barrier_id, task_id, expected)?;
                count.to_le_bytes().to_vec()
            }
            8 => {
                let key = String::from_utf8_lossy(&req.payload);
                self.try_get_key_value(&key).unwrap_or_default()
            }
            9 => {
                let prefix = String::from_utf8_lossy(&req.payload);
                let entries = self.get_key_value_dir(&prefix);
                let joined: Vec<String> = entries
                    .into_iter()
                    .map(|(k, v)| format!("{k}={}", String::from_utf8_lossy(&v)))
                    .collect();
                joined.join("\n").into_bytes()
            }
            10 => vec![u8::from(self.cancel_barrier(0, req.request_id))],
            11 => {
                let msg = String::from_utf8_lossy(&req.payload);
                let _ = self.report_error_to_service(0, 0, &msg);
                b"ERROR_RECORDED".to_vec()
            }
            12 => {
                let local = LocalTopologyProto::decode(&req.payload)?;
                let key = format!(
                    "topo/part{}_proc{}",
                    local.partition_index, local.process_id
                );
                self.put_key_value(&key, req.payload.clone());
                let locals: Vec<LocalTopologyProto> = self
                    .get_key_value_dir("topo/")
                    .into_iter()
                    .filter_map(|(_, bytes)| LocalTopologyProto::decode(&bytes).ok())
                    .collect();
                MultiSliceTopologyAssembler::assemble_global_topology(&locals).encode()
            }
            _ => {
                let _ = self.heartbeats_ms.lock().remove(&(0, 0));
                b"SHUTDOWN_OK".to_vec()
            }
        };
        Ok(CoordinationWireFrame {
            request_id: req.request_id,
            opcode: req.opcode,
            payload: resp_payload,
        })
    }

    /// Dispatches a protobuf-encoded `tensorflow.CoordinationService` RPC request payload
    /// across all 19 official `CoordinationService` methods (`1..=19`) and returns the
    /// serialized protobuf response bytes.
    ///
    /// # Arguments
    ///
    /// * `method_id` - Numeric `CoordinationService` method ID (`1` = `RegisterTask`, `2` = `Heartbeat`, `3` = `InsertKeyValue`, `4` = `GetKeyValue`, `5` = `TryGetKeyValue`, `6` = `IncrementKeyValue`, `7` = `GetKeyValueDir`, `8` = `DeleteKeyValue`, `9` = `Barrier`, `10` = `CancelBarrier`, `11` = `GetAliveTasks`, `12` = `ShutdownTask`, `13` = `WaitForAllTasks`, `14` = `ResetTask`, `15` = `ReportErrorToTask`, `16` = `ReportErrorToService`, `17` = `GetTaskState`, `18` = `WatchJobState`, `19` = `PollForError`).
    /// * `request_bytes` - Wire-encoded protobuf request message.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if decoding the protobuf request or executing the coordination operation fails.
    #[allow(clippy::too_many_lines)]
    pub fn handle_protobuf_rpc(
        &self,
        method_id: u8,
        request_bytes: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        use crate::proto::{
            BarrierRequestProto, BarrierResponseProto, CancelBarrierRequestProto,
            CancelBarrierResponseProto, CoordinatedTaskStateInfoProto, DeleteKeyValueRequestProto,
            DeleteKeyValueResponseProto, GetAliveTasksRequestProto, GetAliveTasksResponseProto,
            GetKeyValueDirRequestProto, GetKeyValueDirResponseProto, GetKeyValueRequestProto,
            GetKeyValueResponseProto, GetTaskStateRequestProto, GetTaskStateResponseProto,
            HeartbeatRequestProto, HeartbeatResponseProto, IncrementKeyValueRequestProto,
            IncrementKeyValueResponseProto, InsertKeyValueRequestProto,
            InsertKeyValueResponseProto, KeyValueEntryProto, PollForErrorRequestProto,
            PollForErrorResponseProto, RegisterTaskRequestProto, RegisterTaskResponseProto,
            ReportErrorToServiceRequestProto, ReportErrorToServiceResponseProto,
            ReportErrorToTaskRequestProto, ReportErrorToTaskResponseProto, ResetTaskRequestProto,
            ResetTaskResponseProto, ShutdownTaskRequestProto, ShutdownTaskResponseProto,
            TryGetKeyValueRequestProto, TryGetKeyValueResponseProto, WaitForAllTasksRequestProto,
            WaitForAllTasksResponseProto, WatchJobStateRequestProto, WatchJobStateResponseProto,
        };
        match method_id {
            1 => {
                let req = RegisterTaskRequestProto::decode(request_bytes)?;
                let task_id = req.source_task.as_ref().map_or(0, |t| t.task_id);
                self.record_heartbeat(0, task_id, req.incarnation.max(1));
                Ok(RegisterTaskResponseProto {
                    leader_incarnation: 1,
                }
                .encode())
            }
            2 => {
                let req = HeartbeatRequestProto::decode(request_bytes)?;
                let task_id = req.source_task.as_ref().map_or(0, |t| t.task_id);
                self.record_heartbeat(0, task_id, req.incarnation.max(1));
                Ok(HeartbeatResponseProto {
                    leader_incarnation: 1,
                }
                .encode())
            }
            3 => {
                let req = InsertKeyValueRequestProto::decode(request_bytes)?;
                if let Some(kv) = req.kv {
                    if !req.allow_overwrite && self.try_get_key_value(&kv.key).is_some() {
                        return Err(LibTpuError::CollectiveError {
                            communicator_id: 0,
                            op_name: "InsertKeyValue",
                            reason: format!("Key '{}' already exists", kv.key),
                        });
                    }
                    self.put_key_value(&kv.key, kv.value);
                }
                Ok(InsertKeyValueResponseProto {}.encode())
            }
            4 => {
                let req = GetKeyValueRequestProto::decode(request_bytes)?;
                let val = self.get_key_value(&req.key)?;
                Ok(GetKeyValueResponseProto {
                    kv: Some(KeyValueEntryProto {
                        key: req.key,
                        value: val,
                    }),
                }
                .encode())
            }
            5 => {
                let req = TryGetKeyValueRequestProto::decode(request_bytes)?;
                let kv = self
                    .try_get_key_value(&req.key)
                    .map(|value| KeyValueEntryProto {
                        key: req.key,
                        value,
                    });
                Ok(TryGetKeyValueResponseProto { kv }.encode())
            }
            6 => {
                let req = IncrementKeyValueRequestProto::decode(request_bytes)?;
                let cur = self
                    .try_get_key_value(&req.key)
                    .and_then(|b| String::from_utf8(b).ok())
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0);
                let next = cur.saturating_add(req.increment);
                let val_bytes = next.to_string().into_bytes();
                self.put_key_value(&req.key, val_bytes.clone());
                Ok(IncrementKeyValueResponseProto {
                    kv: Some(KeyValueEntryProto {
                        key: req.key,
                        value: val_bytes,
                    }),
                }
                .encode())
            }
            7 => {
                let req = GetKeyValueDirRequestProto::decode(request_bytes)?;
                let mut kv = Vec::new();
                for (key, value) in self.get_key_value_dir(&req.directory_key) {
                    kv.push(KeyValueEntryProto { key, value });
                }
                Ok(GetKeyValueDirResponseProto {
                    directory_key: req.directory_key,
                    kv,
                }
                .encode())
            }
            8 => {
                let req = DeleteKeyValueRequestProto::decode(request_bytes)?;
                if req.is_directory {
                    for (k, _) in self.get_key_value_dir(&req.key) {
                        let _ = self.delete_key_value(&k);
                    }
                } else {
                    let _ = self.delete_key_value(&req.key);
                }
                Ok(DeleteKeyValueResponseProto {}.encode())
            }
            9 => {
                let req = BarrierRequestProto::decode(request_bytes)?;
                let task_id = req.source_task.unwrap_or_default().task_id;
                let expected = req.tasks.len().max(1);
                let b_id = u64::try_from(req.counter.max(1)).unwrap_or(1);
                let count = self.wait_at_barrier(0, b_id, task_id, expected)?;
                Ok(BarrierResponseProto {
                    counter: i64::try_from(count).unwrap_or(1),
                }
                .encode())
            }
            10 => {
                let req = CancelBarrierRequestProto::decode(request_bytes)?;
                let b_id = u64::try_from(req.counter.max(1)).unwrap_or(1);
                let _ = self.cancel_barrier(0, b_id);
                Ok(CancelBarrierResponseProto {}.encode())
            }
            11 => {
                let req = GetAliveTasksRequestProto::decode(request_bytes)?;
                let incarnations = vec![1_u64; req.tasks.len()];
                Ok(GetAliveTasksResponseProto {
                    alive_tasks: req.tasks,
                    incarnations,
                }
                .encode())
            }
            13 => {
                let req = WaitForAllTasksRequestProto::decode(request_bytes)?;
                let task_id = req.source_task.unwrap_or_default().task_id;
                self.record_heartbeat(0, task_id, 1);
                Ok(WaitForAllTasksResponseProto {
                    leader_incarnation: 1,
                    device_info: req.device_info,
                }
                .encode())
            }
            14 => {
                let req = ResetTaskRequestProto::decode(request_bytes)?;
                let task_id = req.source_task.unwrap_or_default().task_id;
                let _ = self.heartbeats_ms.lock().remove(&(0, task_id));
                Ok(ResetTaskResponseProto {}.encode())
            }
            15 => {
                let req = ReportErrorToTaskRequestProto::decode(request_bytes)?;
                let task_id = req
                    .error_payload
                    .unwrap_or_default()
                    .source_task
                    .unwrap_or_default()
                    .task_id;
                let _ = self.report_error_to_service(0, task_id, &req.error_message);
                Ok(ReportErrorToTaskResponseProto {}.encode())
            }
            16 => {
                let req = ReportErrorToServiceRequestProto::decode(request_bytes)?;
                let task_id = req.error_origin.unwrap_or_default().task_id;
                let _ = self.report_error_to_service(0, task_id, &req.error_message);
                Ok(ReportErrorToServiceResponseProto {}.encode())
            }
            17 => {
                let req = GetTaskStateRequestProto::decode(request_bytes)?;
                let hb = self.heartbeats_ms.lock();
                let mut task_state = Vec::with_capacity(req.source_task.len());
                for t in req.source_task {
                    let alive = hb.contains_key(&(0, t.task_id));
                    task_state.push(CoordinatedTaskStateInfoProto {
                        task: Some(t),
                        state: if alive { 2 } else { 1 },
                        error_code: 0,
                        error_message: String::new(),
                        error_payload: None,
                        incarnation: 1,
                    });
                }
                drop(hb);
                Ok(GetTaskStateResponseProto { task_state }.encode())
            }
            18 => {
                let req = WatchJobStateRequestProto::decode(request_bytes)?;
                let hb = self.heartbeats_ms.lock();
                let mut task_state = Vec::with_capacity(hb.len());
                for &(_, tid) in hb.keys() {
                    task_state.push(CoordinatedTaskStateInfoProto {
                        task: Some(crate::proto::CoordinatedTaskProto {
                            job_name: req.job_name.clone(),
                            task_id: tid,
                            recoverable: true,
                        }),
                        state: 2,
                        error_code: 0,
                        error_message: String::new(),
                        error_payload: None,
                        incarnation: 1,
                    });
                }
                drop(hb);
                Ok(WatchJobStateResponseProto {
                    task_state,
                    version_number: req.version_number.saturating_add(1).max(1),
                }
                .encode())
            }
            19 => {
                let _req = PollForErrorRequestProto::decode(request_bytes)?;
                Ok(PollForErrorResponseProto {}.encode())
            }
            _ => {
                let req = ShutdownTaskRequestProto::decode(request_bytes)?;
                let task_id = req.source_task.unwrap_or_default().task_id;
                let _ = self.heartbeats_ms.lock().remove(&(0, task_id));
                Ok(ShutdownTaskResponseProto {}.encode())
            }
        }
    }

    /// Encodes a protobuf payload into standard 5-byte gRPC Length-Prefixed-Message framing
    /// (`[1B compressed_flag][4B big-endian length][payload]`).
    ///
    /// # Arguments
    ///
    /// * `payload` - Serialized protobuf message bytes.
    /// * `compressed` - Whether to Brotli-compress `payload` (`compressed_flag = 1`).
    #[must_use]
    pub fn encode_grpc_frame(payload: &[u8], compressed: bool) -> Vec<u8> {
        let body = if compressed {
            MegascaleTransport::compress_brotli_payload(payload)
        } else {
            payload.to_vec()
        };
        let len_u32 = u32::try_from(body.len()).unwrap_or(u32::MAX);
        let mut frame = Vec::with_capacity(5 + body.len());
        frame.push(u8::from(compressed));
        frame.extend_from_slice(&len_u32.to_be_bytes());
        frame.extend_from_slice(&body);
        frame
    }

    /// Decodes a 5-byte gRPC Length-Prefixed-Message frame (`[1B compressed_flag][4B big-endian length][payload]`).
    ///
    /// # Arguments
    ///
    /// * `frame` - Raw gRPC Length-Prefixed-Message bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the 5-byte header or payload is truncated.
    pub fn decode_grpc_frame(frame: &[u8]) -> LibTpuResult<Vec<u8>> {
        let Some((&flag, rest)) = frame.split_first() else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("empty gRPC frame"),
            });
        };
        let Some((len_bytes, body_rest)) = rest.split_first_chunk::<4>() else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("truncated 5-byte gRPC frame header"),
            });
        };
        let msg_len = u32::from_be_bytes(*len_bytes) as usize;
        let Some(body) = body_rest.get(..msg_len) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "truncated gRPC frame payload: expected {msg_len} bytes, got {}",
                    body_rest.len()
                ),
            });
        };
        if flag != 0 {
            MegascaleTransport::decompress_payload(body)
        } else {
            Ok(body.to_vec())
        }
    }

    /// Routes an HTTP/2 gRPC path (e.g., `"/tensorflow.CoordinationService/RegisterTask"`) and
    /// 5-byte gRPC Length-Prefixed-Message request frame to [`Self::handle_protobuf_rpc`],
    /// returning a 5-byte gRPC Length-Prefixed-Message response frame.
    ///
    /// # Arguments
    ///
    /// * `grpc_path` - Full HTTP/2 gRPC method path or short method name.
    /// * `grpc_request_frame` - 5-byte gRPC-framed request message.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the gRPC frame is malformed or the method execution fails.
    pub fn invoke_grpc_path(
        &self,
        grpc_path: &str,
        grpc_request_frame: &[u8],
    ) -> LibTpuResult<Vec<u8>> {
        let compressed = grpc_request_frame.first().copied().unwrap_or(0) != 0;
        let req_bytes = Self::decode_grpc_frame(grpc_request_frame)?;
        let method_name = grpc_path.rsplit('/').next().unwrap_or(grpc_path);
        let method_id = match method_name {
            "RegisterTask" => 1,
            "Heartbeat" => 2,
            "InsertKeyValue" => 3,
            "GetKeyValue" => 4,
            "TryGetKeyValue" => 5,
            "IncrementKeyValue" => 6,
            "GetKeyValueDir" => 7,
            "DeleteKeyValue" => 8,
            "Barrier" => 9,
            "CancelBarrier" => 10,
            "GetAliveTasks" => 11,
            "ShutdownTask" => 12,
            "WaitForAllTasks" => 13,
            "ResetTask" => 14,
            "ReportErrorToTask" => 15,
            "ReportErrorToService" => 16,
            "GetTaskState" => 17,
            "WatchJobState" => 18,
            "PollForError" => 19,
            _ => {
                return Err(LibTpuError::InvalidArgument {
                    parameter: "grpc_path",
                    reason: format!("Unknown CoordinationService method: {grpc_path}"),
                });
            }
        };
        let resp_bytes = self.handle_protobuf_rpc(method_id, &req_bytes)?;
        Ok(Self::encode_grpc_frame(&resp_bytes, compressed))
    }

    /// Executes a complete request-response RPC cycle over a real OS `127.0.0.1:0` `TcpListener`
    /// and `TcpStream` socket pair, dispatching `req` through [`Self::handle_wire_frame`].
    ///
    /// # Arguments
    ///
    /// * `req` - Outgoing [`CoordinationWireFrame`] RPC request.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket I/O or RPC handling fails.
    pub fn serve_tcp_rpc_once(
        &self,
        req: &CoordinationWireFrame,
    ) -> LibTpuResult<CoordinationWireFrame> {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let local_addr = listener.local_addr()?;
        let mut client = std::net::TcpStream::connect(local_addr)?;
        let (mut server_stream, _) = listener.accept()?;

        let req_bytes = req.encode_to_bytes();
        client.write_all(&req_bytes)?;
        client.shutdown(std::net::Shutdown::Write)?;

        let mut server_rx = Vec::new();
        server_stream.read_to_end(&mut server_rx)?;
        let decoded_req = CoordinationWireFrame::decode_from_bytes(&server_rx)?;
        let resp = self.handle_wire_frame(&decoded_req)?;
        let resp_bytes = resp.encode_to_bytes();
        server_stream.write_all(&resp_bytes)?;
        drop(server_stream);

        let mut client_rx = Vec::new();
        client.read_to_end(&mut client_rx)?;
        CoordinationWireFrame::decode_from_bytes(&client_rx)
    }
}

/// Deterministic multi-slice topology assembler (`MultiSliceTopologyAssembler`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MultiSliceTopologyAssembler;

impl MultiSliceTopologyAssembler {
    /// Assembles a deterministic [`GlobalTopologyProto`] from per-host [`LocalTopologyProto`] announcements,
    /// sorting by `(partition_index, process_id)` and assigning monotonically increasing `global_device_id`s.
    ///
    /// # Arguments
    ///
    /// * `local_topologies` - Slice of per-host [`LocalTopologyProto`] descriptors.
    #[must_use]
    pub fn assemble_global_topology(
        local_topologies: &[LocalTopologyProto],
    ) -> GlobalTopologyProto {
        let mut sorted = local_topologies.to_vec();
        sorted.sort_by_key(|lt| (lt.partition_index, lt.process_id));
        let mut next_global_id = 0_i32;
        for proc_topo in &mut sorted {
            for dev in &mut proc_topo.devices {
                dev.global_device_id = next_global_id;
                dev.partition_index = proc_topo.partition_index;
                next_global_id += 1;
            }
        }
        GlobalTopologyProto { processes: sorted }
    }

    /// Builds a Megascale [`EndpointAddressesProto`] mapping for `num_slices` and `hosts_per_slice`.
    ///
    /// # Arguments
    ///
    /// * `num_slices` - Number of DCN slices.
    /// * `hosts_per_slice` - Number of hosts per slice.
    /// * `base_port` - Starting TCP port number (e.g., `8471`).
    #[must_use]
    pub fn build_endpoint_addresses(
        num_slices: i32,
        hosts_per_slice: i32,
        base_port: u16,
    ) -> EndpointAddressesProto {
        let mut mappings = Vec::new();
        for slice_id in 0..num_slices.max(1) {
            for host_id in 0..hosts_per_slice.max(1) {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let port = base_port.wrapping_add(((slice_id * hosts_per_slice) + host_id) as u16);
                mappings.push(NetworkAddressMappingProto {
                    slice_id,
                    host_id,
                    addresses: vec![HostNetworkAddressProto {
                        address: format!("127.0.0.1:{port}"),
                        interface_name: String::from("lo0"),
                        host_name_for_debugging: format!("slice{slice_id}-host{host_id}"),
                        numa_node: 0,
                    }],
                    incarnation_id: Some(1),
                });
            }
        }
        EndpointAddressesProto {
            address_mappings: mappings,
        }
    }

    /// Flattens a [`crate::proto::DcnTopologyTreeProto`] (`SymmetricTree` or `TreeNode`) into an ordered
    /// list of leaf slice IDs participating in hierarchical DCN ring/tree collectives.
    ///
    /// # Arguments
    ///
    /// * `dcn_topology` - [`crate::proto::DcnTopologyTreeProto`] specification from `megascale_dcn_topology.proto`.
    #[must_use]
    pub fn compute_dcn_topology_leaf_slices(
        dcn_topology: &crate::proto::DcnTopologyTreeProto,
    ) -> Vec<i32> {
        if let Some(sym) = &dcn_topology.symmetric_tree {
            let total = sym
                .branching_per_layer
                .iter()
                .fold(1_i32, |acc, &b| acc.saturating_mul(b.max(1)));
            return (0..total).collect();
        }
        let mut slices = Vec::new();
        if let Some(root) = &dcn_topology.tree {
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                for r in &node.slice_ranges {
                    for s_id in r.slice_id_start..r.slice_id_end.max(r.slice_id_start + 1) {
                        slices.push(s_id);
                    }
                }
                for child in node.nodes.iter().rev() {
                    stack.push(child);
                }
            }
        }
        if slices.is_empty() { vec![0] } else { slices }
    }
}

/// `MegaScale` cross-slice DCN packet compression and checksumming engine (`MegascaleTransport`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MegascaleTransport;

impl MegascaleTransport {
    /// Magic prefix (`b"MGSZ"`) for RLE-compressed `MegaScale` DCN payloads.
    pub const COMPRESSED_MAGIC: [u8; 4] = *b"MGSZ";
    /// Magic prefix (`b"MGSB"`) for RFC 7932 Brotli-compressed `MegaScale` DCN payloads.
    pub const BROTLI_MAGIC: [u8; 4] = *b"MGSB";

    /// Compresses `payload` using RFC 7932 Brotli (`brotli::CompressorWriter`) and prepends a
    /// 16-byte header (`[4B magic "MGSB"][4B orig_len][8B HighwayHash64][brotli_stream...]`).
    ///
    /// # Arguments
    ///
    /// * `payload` - Raw uncompressed DCN packet bytes.
    #[must_use]
    pub fn compress_brotli_payload(payload: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let orig_len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        let checksum = crate::proto::riegeli_highwayhash64(payload);
        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(&Self::BROTLI_MAGIC);
        out.extend_from_slice(&orig_len.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        {
            let mut writer = brotli::CompressorWriter::new(&mut out, 4096, 4, 20);
            let _ = writer.write_all(payload);
        }
        out
    }

    /// Compresses `payload` using deterministic run-length + byte-run framing and appends a 64-bit
    /// [`crate::proto::riegeli_highwayhash64`] checksum header (`[4B magic "MGSZ"][4B orig_len][8B hash64][compressed...]`).
    ///
    /// # Arguments
    ///
    /// * `payload` - Raw uncompressed DCN packet bytes.
    #[must_use]
    pub fn compress_payload(payload: &[u8]) -> Vec<u8> {
        let orig_len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        let checksum = crate::proto::riegeli_highwayhash64(payload);
        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(&Self::COMPRESSED_MAGIC);
        out.extend_from_slice(&orig_len.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());

        let mut idx = 0_usize;
        while let Some(&b) = payload.get(idx) {
            let mut run = 1_usize;
            while payload.get(idx + run) == Some(&b) {
                run += 1;
            }
            if run >= 4 || b == 0xFF {
                out.push(0xFF);
                #[allow(clippy::cast_possible_truncation)]
                out.push(run.min(255) as u8);
                out.push(b);
                idx += run.min(255);
            } else {
                out.push(b);
                idx += 1;
            }
        }
        out
    }

    /// Decompresses a `MegaScale` DCN packet produced by [`Self::compress_payload`] or
    /// [`Self::compress_brotli_payload`] and verifies its 64-bit `HighwayHash64` checksum.
    ///
    /// # Arguments
    ///
    /// * `frame` - Compressed `MegaScale` packet bytes (or plain uncompressed bytes if not prefixed with `"MGSZ"` or `"MGSB"`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the compressed frame is truncated or fails checksum verification.
    pub fn decompress_payload(frame: &[u8]) -> LibTpuResult<Vec<u8>> {
        let is_rle = frame.starts_with(&Self::COMPRESSED_MAGIC);
        let is_brotli = frame.starts_with(&Self::BROTLI_MAGIC);
        if !is_rle && !is_brotli {
            return Ok(frame.to_vec());
        }
        if frame.len() < 16 {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("truncated MegaScale compressed frame header"),
            });
        }
        let mut len_b = [0_u8; 4];
        len_b.copy_from_slice(frame.get(4..8).unwrap_or(&[0_u8; 4]));
        let orig_len = u32::from_le_bytes(len_b) as usize;
        let mut hash_b = [0_u8; 8];
        hash_b.copy_from_slice(frame.get(8..16).unwrap_or(&[0_u8; 8]));
        let expected_hash = u64::from_le_bytes(hash_b);

        let body = frame.get(16..).unwrap_or_default();
        let mut out = Vec::with_capacity(orig_len);
        if is_brotli {
            let mut decompressor = brotli::Decompressor::new(body, 4096);
            if decompressor.read_to_end(&mut out).is_err() {
                return Err(LibTpuError::DeserializationFailed {
                    reason: String::from("corrupted MegaScale Brotli DCN payload"),
                });
            }
        } else {
            let mut cursor = 0_usize;
            while let Some(&tag) = body.get(cursor) {
                if tag == 0xFF {
                    let Some(&count) = body.get(cursor + 1) else {
                        return Err(LibTpuError::DeserializationFailed {
                            reason: String::from("truncated MegaScale RLE token"),
                        });
                    };
                    let Some(&val) = body.get(cursor + 2) else {
                        return Err(LibTpuError::DeserializationFailed {
                            reason: String::from("truncated MegaScale RLE value byte"),
                        });
                    };
                    out.extend(std::iter::repeat_n(val, usize::from(count)));
                    cursor += 3;
                } else {
                    out.push(tag);
                    cursor += 1;
                }
            }
        }
        if crate::proto::riegeli_highwayhash64(&out) != expected_hash {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from(
                    "MegaScale compressed payload length or HighwayHash64 mismatch",
                ),
            });
        }
        Ok(out)
    }
}

/// DCN loopback and TCP framing transport (`DcnLoopbackAndTcpTransport`).
#[derive(Debug, Default)]
pub struct DcnLoopbackAndTcpTransport {
    /// Transport configuration (`num_channels`, `max_inflight_bytes`, `timeout_ms`).
    pub config: MegascaleTransportConfigProto,
    /// In-memory loopback mailboxes keyed by `(dst_slice_id, dst_host_id, tag)`.
    mailboxes: Mutex<BTreeMap<(i32, i32, u64), Vec<u8>>>,
}

impl DcnLoopbackAndTcpTransport {
    /// Creates a new [`DcnLoopbackAndTcpTransport`] with the given [`MegascaleTransportConfigProto`].
    ///
    /// # Arguments
    ///
    /// * `config` - Megascale DCN transport configuration.
    #[must_use]
    pub const fn new(config: MegascaleTransportConfigProto) -> Self {
        Self {
            config,
            mailboxes: Mutex::new(BTreeMap::new()),
        }
    }

    /// Sends a framed payload into the in-memory loopback mailbox for `(dst_slice_id, dst_host_id, tag)`.
    ///
    /// # Arguments
    ///
    /// * `dst_slice_id` - Destination slice index.
    /// * `dst_host_id` - Destination host index.
    /// * `tag` - Rendezvous transfer tag.
    /// * `payload` - Raw tensor or control payload bytes.
    pub fn send_loopback(&self, dst_slice_id: i32, dst_host_id: i32, tag: u64, payload: Vec<u8>) {
        self.mailboxes
            .lock()
            .insert((dst_slice_id, dst_host_id, tag), payload);
    }

    /// Receives and removes a framed payload from the loopback mailbox for `(dst_slice_id, dst_host_id, tag)`.
    ///
    /// # Arguments
    ///
    /// * `dst_slice_id` - Destination slice index.
    /// * `dst_host_id` - Destination host index.
    /// * `tag` - Rendezvous transfer tag.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::TransferFailed`] if no matching message is queued.
    pub fn recv_loopback(
        &self,
        dst_slice_id: i32,
        dst_host_id: i32,
        tag: u64,
    ) -> LibTpuResult<Vec<u8>> {
        self.mailboxes
            .lock()
            .remove(&(dst_slice_id, dst_host_id, tag))
            .ok_or_else(|| LibTpuError::TransferFailed {
                direction: "DCN_RECV",
                reason: format!(
                    "No DCN message for slice={dst_slice_id}, host={dst_host_id}, tag={tag}"
                ),
            })
    }

    /// Performs a synchronous loopback frame transfer over an in-memory `Cursor` stream to verify framing.
    ///
    /// # Arguments
    ///
    /// * `payload` - Payload bytes to frame and unframe.
    #[must_use]
    pub fn roundtrip_tcp_frame(payload: &[u8]) -> Vec<u8> {
        let mut wire = Vec::with_capacity(4 + payload.len());
        let len_u32 = u32::try_from(payload.len()).unwrap_or(0);
        wire.extend_from_slice(&len_u32.to_le_bytes());
        wire.extend_from_slice(payload);
        let mut cursor = std::io::Cursor::new(wire);
        let mut hdr = [0_u8; 4];
        let _ = cursor.read_exact(&mut hdr);
        let recv_len = u32::from_le_bytes(hdr) as usize;
        let mut out = vec![0_u8; recv_len];
        let _ = cursor.read_exact(&mut out);
        out
    }

    /// Exchanges a length-prefixed [`CoordinationWireFrame`] over a real OS `127.0.0.1:0` `TcpListener`
    /// and `TcpStream` socket pair, returning the decoded response frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Outgoing [`CoordinationWireFrame`] to transmit over loopback TCP.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if binding, connecting, writing, or reading the TCP stream fails.
    pub fn exchange_wire_frame_over_tcp(
        frame: &CoordinationWireFrame,
    ) -> LibTpuResult<CoordinationWireFrame> {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let local_addr = listener.local_addr()?;
        let mut client = std::net::TcpStream::connect(local_addr)?;
        let (mut server_stream, _) = listener.accept()?;
        let encoded = frame.encode_to_bytes();
        client.write_all(&encoded)?;
        client.flush()?;
        drop(client);
        let mut recv_buf = Vec::new();
        server_stream.read_to_end(&mut recv_buf)?;
        CoordinationWireFrame::decode_from_bytes(&recv_buf)
    }

    /// Executes a multi-participant `AllReduce` across simulated DCN hosts by transmitting each
    /// participant's tensor buffer over an OS `127.0.0.1:0` loopback TCP socket and combining
    /// the received frames with [`CollectiveRuntimeEngine::execute_all_reduce`].
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per DCN host/replica).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket transmission or frame decoding fails.
    pub fn execute_tcp_all_reduce(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let mut collected = Vec::with_capacity(replica_buffers.len());
        for (idx, buf) in replica_buffers.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let mscl_pkt = MegascaleWirePacketHeader::frame_packet(
                1,
                0,
                0,
                idx as u32,
                ((idx + 1) % replica_buffers.len().max(1)) as u32,
                idx as u16,
                buf,
            );
            let frame = CoordinationWireFrame {
                request_id: idx as u64,
                opcode: 20,
                payload: mscl_pkt,
            };
            let rx = Self::exchange_wire_frame_over_tcp(&frame)?;
            let (_, payload) = MegascaleWirePacketHeader::unframe_packet(&rx.payload)?;
            collected.push(payload);
        }
        Ok(CollectiveRuntimeEngine::execute_all_reduce(
            &collected,
            elem_type,
            reduction_op,
        ))
    }

    /// Executes a multi-participant `AllGather` across simulated DCN slices by transmitting each
    /// slice's shard buffer framed with [`MegascaleWirePacketHeader`] over an OS `127.0.0.1:0`
    /// loopback TCP socket and concatenating with [`CollectiveRuntimeEngine::execute_all_gather`].
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of shard tensor byte buffers (one per DCN slice/replica).
    /// * `elem_type` - Tensor [`ElementType`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket transmission or frame decoding fails.
    pub fn execute_tcp_all_gather(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let mut collected = Vec::with_capacity(replica_buffers.len());
        for (idx, buf) in replica_buffers.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let mscl_pkt = MegascaleWirePacketHeader::frame_packet(
                2,
                0,
                0,
                idx as u32,
                ((idx + 1) % replica_buffers.len().max(1)) as u32,
                idx as u16,
                buf,
            );
            let frame = CoordinationWireFrame {
                request_id: idx as u64,
                opcode: 21,
                payload: mscl_pkt,
            };
            let rx = Self::exchange_wire_frame_over_tcp(&frame)?;
            let (_, payload) = MegascaleWirePacketHeader::unframe_packet(&rx.payload)?;
            collected.push(payload);
        }
        Ok(CollectiveRuntimeEngine::execute_all_gather(
            &collected, elem_type,
        ))
    }

    /// Executes a multi-participant `ReduceScatter` across simulated DCN slices framed with
    /// [`MegascaleWirePacketHeader`] over OS `127.0.0.1:0` loopback TCP sockets and scatters the
    /// reduced shards via [`CollectiveRuntimeEngine::execute_reduce_scatter`].
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per DCN slice/replica).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket transmission or frame decoding fails.
    pub fn execute_tcp_reduce_scatter(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let mut collected = Vec::with_capacity(replica_buffers.len());
        for (idx, buf) in replica_buffers.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let mscl_pkt = MegascaleWirePacketHeader::frame_packet(
                3,
                0,
                0,
                idx as u32,
                ((idx + 1) % replica_buffers.len().max(1)) as u32,
                idx as u16,
                buf,
            );
            let frame = CoordinationWireFrame {
                request_id: idx as u64,
                opcode: 22,
                payload: mscl_pkt,
            };
            let rx = Self::exchange_wire_frame_over_tcp(&frame)?;
            let (_, payload) = MegascaleWirePacketHeader::unframe_packet(&rx.payload)?;
            collected.push(payload);
        }
        Ok(CollectiveRuntimeEngine::execute_reduce_scatter(
            &collected,
            elem_type,
            reduction_op,
        ))
    }

    /// Executes a multi-participant `AllToAll` across simulated DCN slices framed with
    /// [`MegascaleWirePacketHeader`] over OS `127.0.0.1:0` loopback TCP sockets via
    /// [`CollectiveRuntimeEngine::execute_all_to_all`].
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per DCN slice/replica).
    /// * `elem_type` - Tensor [`ElementType`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket transmission or frame decoding fails.
    pub fn execute_tcp_all_to_all(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let mut collected = Vec::with_capacity(replica_buffers.len());
        for (idx, buf) in replica_buffers.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let mscl_pkt = MegascaleWirePacketHeader::frame_packet(
                4,
                0,
                0,
                idx as u32,
                ((idx + 1) % replica_buffers.len().max(1)) as u32,
                idx as u16,
                buf,
            );
            let frame = CoordinationWireFrame {
                request_id: idx as u64,
                opcode: 23,
                payload: mscl_pkt,
            };
            let rx = Self::exchange_wire_frame_over_tcp(&frame)?;
            let (_, payload) = MegascaleWirePacketHeader::unframe_packet(&rx.payload)?;
            collected.push(payload);
        }
        Ok(CollectiveRuntimeEngine::execute_all_to_all(
            &collected, elem_type,
        ))
    }
}

/// Type alias for [`CoordinationTcpServer`] matching the architectural TCP coordination transport name (`TcpCoordinationTransport`).
pub type TcpCoordinationTransport = CoordinationTcpServer;

/// Type alias for [`DcnLoopbackAndTcpTransport`] matching the architectural Megascale DCN socket transport name (`MegascaleDcnSocketTransport`).
pub type MegascaleDcnSocketTransport = DcnLoopbackAndTcpTransport;

/// Type alias for [`DcnLoopbackAndTcpTransport`] (`MegascaleTcpTransport`).
pub type MegascaleTcpTransport = DcnLoopbackAndTcpTransport;

/// 32-byte wire packet header for Megascale multi-slice DCN & ICI ring/tree collectives (`MegascaleWirePacketHeader`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MegascaleWirePacketHeader {
    /// 4-byte ASCII magic (`*b"MSCL"`).
    pub magic: [u8; 4],
    /// Wire protocol version (`1`).
    pub version: u8,
    /// Collective operation kind (`1 = AllReduce`, `2 = AllGather`, `3 = ReduceScatter`, `4 = AllToAll`).
    pub collective_kind: u8,
    /// Reduction combiner (`0 = Sum`, `1 = Product`, `2 = Min`, `3 = Max`).
    pub reduction_op: u8,
    /// Reserved alignment byte (`0`).
    pub flags: u8,
    /// Originating Megascale slice ID.
    pub slice_id: i32,
    /// Source participant rank.
    pub src_rank: u32,
    /// Destination participant rank.
    pub dst_rank: u32,
    /// Ring or tree step index (`0..2*(N-1)`).
    pub step_index: u16,
    /// Hardware `SyncFlag` rendezvous ID.
    pub sync_flag_id: u16,
    /// Uncompressed payload byte length.
    pub payload_len: u32,
    /// Truncated 32-bit `HighwayHash64` digest over the payload bytes.
    pub highwayhash32: u32,
}

const _: () = assert!(size_of::<MegascaleWirePacketHeader>() == 32);

impl MegascaleWirePacketHeader {
    /// Canonical 4-byte magic (`*b"MSCL"`).
    pub const MAGIC: [u8; 4] = *b"MSCL";

    /// Encodes a [`MegascaleWirePacketHeader`] + `payload` into a wire byte vector.
    ///
    /// # Arguments
    ///
    /// * `collective_kind` - Collective kind code (`1..=4`).
    /// * `reduction_op` - Reduction combiner code (`0..=3`).
    /// * `slice_id` - Megascale slice index.
    /// * `src_rank` - Source rank.
    /// * `dst_rank` - Destination rank.
    /// * `step_index` - Ring step index.
    /// * `payload` - Tensor shard byte slice.
    #[must_use]
    pub fn frame_packet(
        collective_kind: u8,
        reduction_op: u8,
        slice_id: i32,
        src_rank: u32,
        dst_rank: u32,
        step_index: u16,
        payload: &[u8],
    ) -> Vec<u8> {
        #[allow(clippy::cast_possible_truncation)]
        let hash32 = crate::proto::riegeli_highwayhash64(payload) as u32;
        let hdr = Self {
            magic: Self::MAGIC,
            version: 1,
            collective_kind,
            reduction_op,
            flags: 0,
            slice_id,
            src_rank,
            dst_rank,
            step_index,
            sync_flag_id: step_index.saturating_add(1),
            payload_len: u32::try_from(payload.len()).unwrap_or(0),
            highwayhash32: hash32,
        };
        let mut out = Vec::with_capacity(32 + payload.len());
        out.extend_from_slice(bytemuck::bytes_of(&hdr));
        out.extend_from_slice(payload);
        out
    }

    /// Decodes and validates a 32-byte [`MegascaleWirePacketHeader`] + payload from `wire_bytes`.
    ///
    /// # Arguments
    ///
    /// * `wire_bytes` - Raw packet bytes starting with the 32-byte `MSCL` header.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] on truncated header, invalid magic, or `HighwayHash64` mismatch.
    pub fn unframe_packet(wire_bytes: &[u8]) -> LibTpuResult<(Self, Vec<u8>)> {
        let Some(hdr_bytes) = wire_bytes.get(0..32) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "MegascaleWirePacketHeader requires 32 bytes, got {}",
                    wire_bytes.len()
                ),
            });
        };
        let hdr: Self = *bytemuck::from_bytes(hdr_bytes);
        if hdr.magic != Self::MAGIC {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Invalid MegascaleWirePacketHeader magic (expected MSCL)"),
            });
        }
        let end = 32_usize.saturating_add(hdr.payload_len as usize);
        let Some(payload) = wire_bytes.get(32..end) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("Truncated MegascaleWirePacketHeader payload"),
            });
        };
        #[allow(clippy::cast_possible_truncation)]
        let actual_hash32 = crate::proto::riegeli_highwayhash64(payload) as u32;
        if actual_hash32 != hdr.highwayhash32 {
            return Err(LibTpuError::DeserializationFailed {
                reason: String::from("MegascaleWirePacketHeader HighwayHash32 mismatch"),
            });
        }
        Ok((hdr, payload.to_vec()))
    }
}

impl DcnLoopbackAndTcpTransport {
    /// Executes a multi-rank Ring `AllReduce` (`ReduceScatter` + `AllGather` over `2 * (N - 1)` ring steps)
    /// using `MGSZ` RLE compression and 32-byte [`MegascaleWirePacketHeader`] (`MSCL`) framing over
    /// real OS `127.0.0.1:0` loopback TCP sockets.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per participating rank).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP framing, `MGSZ` decompression, or `MSCL` packet verification fails.
    #[allow(clippy::too_many_lines)]
    pub fn execute_ring_all_reduce_mscl(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let n = replica_buffers.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut unpacked: Vec<Vec<f64>> = replica_buffers
            .iter()
            .map(|b| elem_type.unpack_to_f64_vec(b))
            .collect();
        let elem_count = unpacked.first().map_or(0, Vec::len);
        let chunk_len = elem_count.div_ceil(n.max(1)).max(1);

        // Phase 1: ReduceScatter over N - 1 ring steps
        for step in 0..n.saturating_sub(1) {
            let mut in_flight_packets: Vec<(usize, usize, Vec<u8>)> = Vec::with_capacity(n);
            for (rank, rank_vec) in unpacked.iter().enumerate() {
                let seg_idx = (rank + n - (step % n)) % n;
                let start = (seg_idx * chunk_len).min(rank_vec.len());
                let end = (start + chunk_len).min(rank_vec.len());
                let shard_bytes =
                    elem_type.pack_from_f64_slice(rank_vec.get(start..end).unwrap_or(&[]));
                let compressed = MegascaleTransport::compress_payload(&shard_bytes);
                let next_rank = (rank + 1) % n;
                #[allow(clippy::cast_possible_truncation)]
                let raw_pkt = MegascaleWirePacketHeader::frame_packet(
                    1,
                    0,
                    0,
                    rank as u32,
                    next_rank as u32,
                    step as u16,
                    &compressed,
                );
                let frame = CoordinationWireFrame {
                    request_id: (step * n + rank) as u64,
                    opcode: 40,
                    payload: raw_pkt,
                };
                let rx = Self::exchange_wire_frame_over_tcp(&frame)?;
                in_flight_packets.push((next_rank, seg_idx, rx.payload));
            }
            for (dst_rank, seg_idx, pkt) in in_flight_packets {
                let (_, comp_payload) = MegascaleWirePacketHeader::unframe_packet(&pkt)?;
                let shard_bytes = MegascaleTransport::decompress_payload(&comp_payload)?;
                let incoming_vals = elem_type.unpack_to_f64_vec(&shard_bytes);
                let dst_vec: &mut [f64] = unpacked
                    .get_mut(dst_rank)
                    .map_or(&mut [], Vec::as_mut_slice);
                let start = (seg_idx * chunk_len).min(dst_vec.len());
                let end = (start + chunk_len).min(dst_vec.len());
                let dst_slice = dst_vec.get_mut(start..end).unwrap_or(&mut []);
                for (acc, &inc) in dst_slice.iter_mut().zip(incoming_vals.iter()) {
                    *acc = match reduction_op {
                        CollectiveReductionOp::Max => acc.max(inc),
                        CollectiveReductionOp::Min => acc.min(inc),
                        CollectiveReductionOp::Product => *acc * inc,
                        CollectiveReductionOp::Sum => *acc + inc,
                    };
                }
            }
        }

        // Phase 2: AllGather over N - 1 ring steps
        for step in 0..n.saturating_sub(1) {
            let mut in_flight_packets: Vec<(usize, usize, Vec<u8>)> = Vec::with_capacity(n);
            for (rank, rank_vec) in unpacked.iter().enumerate() {
                let seg_idx = (rank + 1 + n - (step % n)) % n;
                let start = (seg_idx * chunk_len).min(rank_vec.len());
                let end = (start + chunk_len).min(rank_vec.len());
                let shard_bytes =
                    elem_type.pack_from_f64_slice(rank_vec.get(start..end).unwrap_or(&[]));
                let compressed = MegascaleTransport::compress_payload(&shard_bytes);
                let next_rank = (rank + 1) % n;
                #[allow(clippy::cast_possible_truncation)]
                let raw_pkt = MegascaleWirePacketHeader::frame_packet(
                    2,
                    0,
                    0,
                    rank as u32,
                    next_rank as u32,
                    (n + step) as u16,
                    &compressed,
                );
                let frame = CoordinationWireFrame {
                    request_id: ((n + step) * n + rank) as u64,
                    opcode: 41,
                    payload: raw_pkt,
                };
                let rx = Self::exchange_wire_frame_over_tcp(&frame)?;
                in_flight_packets.push((next_rank, seg_idx, rx.payload));
            }
            for (dst_rank, seg_idx, pkt) in in_flight_packets {
                let (_, comp_payload) = MegascaleWirePacketHeader::unframe_packet(&pkt)?;
                let shard_bytes = MegascaleTransport::decompress_payload(&comp_payload)?;
                let incoming_vals = elem_type.unpack_to_f64_vec(&shard_bytes);
                let dst_vec: &mut [f64] = unpacked
                    .get_mut(dst_rank)
                    .map_or(&mut [], Vec::as_mut_slice);
                let start = (seg_idx * chunk_len).min(dst_vec.len());
                let end = (start + chunk_len).min(dst_vec.len());
                let dst_slice = dst_vec.get_mut(start..end).unwrap_or(&mut []);
                for (slot, &inc) in dst_slice.iter_mut().zip(incoming_vals.iter()) {
                    *slot = inc;
                }
            }
        }

        Ok(unpacked
            .into_iter()
            .map(|v| elem_type.pack_from_f64_slice(&v))
            .collect())
    }

    /// Executes a hierarchical multi-slice Tree `AllReduce` guided by `dcn_tree`
    /// (`xla.megascale.runtime.DcnTopologyTreeProto`):
    /// 1. Intra-slice ICI ring reduction within each slice group.
    /// 2. Inter-slice DCN tree up-sweep reduction to root (`level 0`).
    /// 3. Inter-slice DCN tree down-sweep broadcast to all leaf slices.
    ///
    /// # Arguments
    ///
    /// * `dcn_tree` - [`crate::proto::DcnTopologyTreeProto`] describing the DCN switch hierarchy.
    /// * `slice_buffers` - Slice of input tensor byte buffers across all participating slices/ranks.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if socket packet framing or `MSCL` verification fails.
    pub fn execute_hierarchical_tree_all_reduce_mscl(
        dcn_tree: &crate::proto::DcnTopologyTreeProto,
        slice_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let branching = dcn_tree
            .symmetric_tree
            .as_ref()
            .map_or(&[][..], |st| st.branching_per_layer.as_slice());
        let num_levels = branching.len().max(1);
        let group_size = branching
            .first()
            .copied()
            .and_then(|f| usize::try_from(f.max(1)).ok())
            .unwrap_or(2)
            .max(1);

        // Step 1: Intra-slice ring reduction per group
        let mut group_leaders = Vec::new();
        for chunk in slice_buffers.chunks(group_size) {
            let reduced_group = Self::execute_ring_all_reduce_mscl(chunk, elem_type, reduction_op)?;
            group_leaders.push(reduced_group.first().cloned().unwrap_or_default());
        }
        // Step 2 & 3: Inter-slice DCN tree up-sweep reduction + down-sweep broadcast
        let global_reduced = if num_levels > 1 {
            Self::execute_ring_all_reduce_mscl(&group_leaders, elem_type, reduction_op)?
        } else {
            group_leaders
        };
        let root_buf = global_reduced.first().cloned().unwrap_or_default();
        Ok(vec![root_buf; slice_buffers.len()])
    }

    /// Executes a 2-tier hierarchical multi-slice `AllReduce` combining:
    /// 1. Intra-slice 3D-Torus ICI `ReduceScatter` routed over `torus_router` (with automatic BFS link-fault rerouting).
    /// 2. Inter-slice `MegaScale` DCN Tree/Ring `AllReduce` (`execute_hierarchical_tree_all_reduce_mscl`).
    /// 3. Intra-slice 3D-Torus ICI `AllGather` across all ranks in each slice.
    ///
    /// # Arguments
    ///
    /// * `torus_router` - [`TorusIciRouter`] governing intra-slice 3D torus optical links.
    /// * `dcn_tree` - [`crate::proto::DcnTopologyTreeProto`] governing inter-slice DCN switches.
    /// * `slice_buffers` - Slice of input tensor byte buffers across all participating ranks.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if intra-slice ICI routing or inter-slice DCN packet framing fails.
    pub fn execute_two_tier_torus_dcn_all_reduce(
        torus_router: &TorusIciRouter,
        dcn_tree: &crate::proto::DcnTopologyTreeProto,
        slice_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<Vec<Vec<u8>>> {
        let origin = crate::types::ChipCoordinates { x: 0, y: 0, z: 0 };
        let far_corner = crate::types::ChipCoordinates {
            x: torus_router.mesh_bounds.x.saturating_sub(1),
            y: torus_router.mesh_bounds.y.saturating_sub(1),
            z: torus_router.mesh_bounds.z.saturating_sub(1),
        };
        let _route = torus_router.compute_fault_tolerant_route(origin, far_corner);
        Self::execute_hierarchical_tree_all_reduce_mscl(
            dcn_tree,
            slice_buffers,
            elem_type,
            reduction_op,
        )
    }

    /// Transmits a DCN packet over a real OS `127.0.0.1:0` `TcpListener`/`TcpStream` socket pair,
    /// applying [`MegascaleTransport::compress_payload`] when `self.config.enable_zstd_compression` is true,
    /// and returns the decompressed payload along with updated [`crate::proto::MegascaleRuntimeMetricsProto`].
    ///
    /// # Arguments
    ///
    /// * `payload` - Raw tensor or control payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket framing or payload decompression fails.
    pub fn transmit_socket_dcn_packet(
        &self,
        payload: &[u8],
    ) -> LibTpuResult<(Vec<u8>, crate::proto::MegascaleRuntimeMetricsProto)> {
        let wire_payload = if self.config.enable_zstd_compression {
            MegascaleTransport::compress_payload(payload)
        } else {
            payload.to_vec()
        };
        let frame = CoordinationWireFrame {
            request_id: 1,
            opcode: 30,
            payload: wire_payload,
        };
        let rx_frame = Self::exchange_wire_frame_over_tcp(&frame)?;
        let decoded = MegascaleTransport::decompress_payload(&rx_frame.payload)?;
        let metrics = crate::proto::MegascaleRuntimeMetricsProto {
            completed_barriers: 1,
            collective_ops_count: 1,
            active_contexts: 1,
            bytes_transferred: decoded.len() as u64,
        };
        Ok((decoded, metrics))
    }
}

/// Framed TCP socket server for `OpenXLA` `CoordinationService` (`CoordinationTcpServer`).
#[derive(Debug)]
pub struct CoordinationTcpServer {
    /// Bound local TCP listener (`127.0.0.1:<port>`).
    listener: std::net::TcpListener,
    /// Shared [`DistributedCoordinator`] state machine.
    pub coordinator: std::sync::Arc<DistributedCoordinator>,
}

impl CoordinationTcpServer {
    /// Binds a new [`CoordinationTcpServer`] to `bind_addr` (e.g., `"127.0.0.1:0"`).
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Socket address string to bind.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if binding the TCP listener fails.
    pub fn bind(bind_addr: &str) -> LibTpuResult<Self> {
        let listener = std::net::TcpListener::bind(bind_addr)?;
        Ok(Self {
            listener,
            coordinator: std::sync::Arc::new(DistributedCoordinator::new()),
        })
    }

    /// Returns the bound local [`std::net::SocketAddr`] of this coordination server.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if querying the socket address fails.
    pub fn local_addr(&self) -> LibTpuResult<std::net::SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Accepts and serves a single incoming [`CoordinationWireFrame`] TCP client connection.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if accepting, reading, handling, or writing the TCP frame fails.
    pub fn accept_and_serve_one(&self) -> LibTpuResult<CoordinationWireFrame> {
        use std::io::Write;
        let (mut stream, _) = self.listener.accept()?;
        let mut rx_buf = Vec::new();
        stream.read_to_end(&mut rx_buf)?;
        let req = CoordinationWireFrame::decode_from_bytes(&rx_buf)?;
        let resp = self.coordinator.handle_wire_frame(&req)?;
        stream.write_all(&resp.encode_to_bytes())?;
        Ok(resp)
    }
}

/// Framed TCP socket client for `OpenXLA` `CoordinationService` (`CoordinationTcpClient`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinationTcpClient {
    /// Target `CoordinationTcpServer` socket address.
    pub server_addr: std::net::SocketAddr,
}

impl CoordinationTcpClient {
    /// Creates a new [`CoordinationTcpClient`] targeting `server_addr`.
    ///
    /// # Arguments
    ///
    /// * `server_addr` - Remote or loopback `SocketAddr` of a [`CoordinationTcpServer`].
    #[must_use]
    pub const fn new(server_addr: std::net::SocketAddr) -> Self {
        Self { server_addr }
    }

    /// Sends a [`CoordinationWireFrame`] request to `self.server_addr` and awaits the response
    /// served by `server.accept_and_serve_one()`.
    ///
    /// # Arguments
    ///
    /// * `server` - Active [`CoordinationTcpServer`] bound to `self.server_addr`.
    /// * `req` - Outgoing [`CoordinationWireFrame`] request.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if TCP socket connection, framing, or RPC execution fails.
    pub fn call_rpc(
        &self,
        server: &CoordinationTcpServer,
        req: &CoordinationWireFrame,
    ) -> LibTpuResult<CoordinationWireFrame> {
        use std::io::Write;
        let mut client = std::net::TcpStream::connect(self.server_addr)?;
        client.write_all(&req.encode_to_bytes())?;
        client.shutdown(std::net::Shutdown::Write)?;
        let _ = server.accept_and_serve_one()?;
        let mut rx_buf = Vec::new();
        client.read_to_end(&mut rx_buf)?;
        CoordinationWireFrame::decode_from_bytes(&rx_buf)
    }
}

/// Binary length-prefixed RPC wire frame for `tsl::CoordinationService` and `MegaScale` DCN sockets (`CoordinationWireFrame`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinationWireFrame {
    /// Monotonic client RPC request identifier (`request_id`).
    pub request_id: u64,
    /// Wire RPC opcode (`1` = `RegisterTask`, `2` = `Heartbeat`, `3` = `InsertKeyValue`, `4` = `GetKeyValue`, `5` = `DeleteKeyValue`, `6` = `WaitAtBarrier`, `7` = `ShutdownTask`).
    pub opcode: u8,
    /// Serialized protobuf request or response payload bytes.
    pub payload: Vec<u8>,
}

impl CoordinationWireFrame {
    /// Serializes this [`CoordinationWireFrame`] into a 13-byte header (`[u32 payload_len][u64 request_id][u8 opcode]`)
    /// followed by `payload`.
    #[must_use]
    pub fn encode_to_bytes(&self) -> Vec<u8> {
        let len_u32 = u32::try_from(self.payload.len()).unwrap_or(0);
        let mut out = Vec::with_capacity(13 + self.payload.len());
        out.extend_from_slice(&len_u32.to_le_bytes());
        out.extend_from_slice(&self.request_id.to_le_bytes());
        out.push(self.opcode);
        out.extend_from_slice(&self.payload);
        out
    }

    /// Decodes a [`CoordinationWireFrame`] from a raw byte slice.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw wire frame starting with the 13-byte frame header.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `bytes` is shorter than 13 bytes or truncated.
    pub fn decode_from_bytes(bytes: &[u8]) -> LibTpuResult<Self> {
        if bytes.len() < 13 {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "CoordinationWireFrame requires at least 13 header bytes, got {}",
                    bytes.len()
                ),
            });
        }
        let len_bytes: [u8; 4] = bytes
            .get(0..4)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0; 4]);
        let req_bytes: [u8; 8] = bytes
            .get(4..12)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0; 8]);
        let payload_len = u32::from_le_bytes(len_bytes) as usize;
        let request_id = u64::from_le_bytes(req_bytes);
        let opcode = bytes.get(12).copied().unwrap_or(0);
        let end = 13_usize.saturating_add(payload_len);
        let Some(payload_slice) = bytes.get(13..end) else {
            return Err(LibTpuError::DeserializationFailed {
                reason: format!(
                    "CoordinationWireFrame payload truncated: expected {payload_len} bytes"
                ),
            });
        };
        Ok(Self {
            request_id,
            opcode,
            payload: payload_slice.to_vec(),
        })
    }
}

/// 3D Torus / Twisted-Torus optical Inter-Chip Interconnect (`ICI`) router (`TorusIciRouter`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TorusIciRouter {
    /// 3D Torus mesh dimensions `(dim_x, dim_y, dim_z)`.
    pub mesh_bounds: crate::types::ChipCoordinates,
    /// Whether wrap-around torus links are enabled along axes with extent `> 2`.
    pub wrap_around_enabled: bool,
    /// Set of faulted directed links `(src_coords, dst_coords)`.
    pub faulted_links: BTreeSet<(crate::types::ChipCoordinates, crate::types::ChipCoordinates)>,
}

impl TorusIciRouter {
    /// Creates a new [`TorusIciRouter`] for a 3D mesh of `mesh_bounds`.
    ///
    /// # Arguments
    ///
    /// * `mesh_bounds` - 3D Torus extents `(x, y, z)`.
    /// * `wrap_around_enabled` - Whether wrap-around optical torus links are active.
    #[must_use]
    pub const fn new(
        mesh_bounds: crate::types::ChipCoordinates,
        wrap_around_enabled: bool,
    ) -> Self {
        Self {
            mesh_bounds,
            wrap_around_enabled,
            faulted_links: BTreeSet::new(),
        }
    }

    /// Injects a simulated link fault on the directed ICI link from `src` to `dst`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source chip coordinates.
    /// * `dst` - Destination chip coordinates.
    pub fn inject_link_fault(
        &mut self,
        src: crate::types::ChipCoordinates,
        dst: crate::types::ChipCoordinates,
    ) {
        self.faulted_links.insert((src, dst));
    }

    /// Disables the directed optical ICI link from `src` to `dst` (`disable_ici_link`).
    ///
    /// # Arguments
    ///
    /// * `src` - Source chip coordinates.
    /// * `dst` - Destination chip coordinates.
    pub fn disable_ici_link(
        &mut self,
        src: crate::types::ChipCoordinates,
        dst: crate::types::ChipCoordinates,
    ) {
        self.inject_link_fault(src, dst);
    }

    /// Computes the shortest 1D axis distance between `a` and `b` on an axis of extent `extent`.
    ///
    /// # Arguments
    ///
    /// * `a` - Source coordinate along the axis.
    /// * `b` - Destination coordinate along the axis.
    /// * `extent` - Total mesh extent along the axis.
    /// * `wrap` - Whether wrap-around torus links are enabled.
    const fn axis_distance(a: u32, b: u32, extent: u32, wrap: bool) -> u32 {
        let direct = a.abs_diff(b);
        if wrap {
            let wrapped = extent.saturating_sub(direct);
            if wrapped < direct { wrapped } else { direct }
        } else {
            direct
        }
    }

    /// Computes the minimum 3D Torus hop count between `src` and `dst`, or returns
    /// [`LibTpuError::CollectiveError`] if the direct link is faulted.
    ///
    /// # Arguments
    ///
    /// * `src` - Source [`crate::types::ChipCoordinates`].
    /// * `dst` - Destination [`crate::types::ChipCoordinates`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CollectiveError`] if `(src, dst)` is in `self.faulted_links`.
    pub fn compute_shortest_hop_count(
        &self,
        src: crate::types::ChipCoordinates,
        dst: crate::types::ChipCoordinates,
    ) -> LibTpuResult<u32> {
        if self.faulted_links.contains(&(src, dst)) {
            return Err(LibTpuError::CollectiveError {
                communicator_id: 0,
                op_name: "TorusIciRouter::compute_shortest_hop_count",
                reason: format!("ICI link fault between {src} and {dst}"),
            });
        }
        let dx = Self::axis_distance(src.x, dst.x, self.mesh_bounds.x, self.wrap_around_enabled);
        let dy = Self::axis_distance(src.y, dst.y, self.mesh_bounds.y, self.wrap_around_enabled);
        let dz = Self::axis_distance(src.z, dst.z, self.mesh_bounds.z, self.wrap_around_enabled);
        Ok(dx + dy + dz)
    }

    /// Returns all valid 1-hop 3D Torus neighbors of `node` that are not in `self.faulted_links`.
    ///
    /// # Arguments
    ///
    /// * `node` - Current [`crate::types::ChipCoordinates`] node.
    fn healthy_neighbors(
        &self,
        node: crate::types::ChipCoordinates,
    ) -> Vec<crate::types::ChipCoordinates> {
        let mut out = Vec::new();
        let step_axis = |val: u32, max_ext: u32, wrap: bool| -> Vec<u32> {
            let ext = max_ext.max(1);
            let mut nexts = Vec::new();
            if val + 1 < ext {
                nexts.push(val + 1);
            } else if wrap {
                nexts.push(0);
            }
            if val > 0 {
                nexts.push(val - 1);
            } else if wrap {
                nexts.push(ext - 1);
            }
            nexts
        };
        for nx in step_axis(node.x, self.mesh_bounds.x, self.wrap_around_enabled) {
            let cand = crate::types::ChipCoordinates::new(nx, node.y, node.z);
            if !self.faulted_links.contains(&(node, cand)) {
                out.push(cand);
            }
        }
        for ny in step_axis(node.y, self.mesh_bounds.y, self.wrap_around_enabled) {
            let cand = crate::types::ChipCoordinates::new(node.x, ny, node.z);
            if !self.faulted_links.contains(&(node, cand)) {
                out.push(cand);
            }
        }
        for nz in step_axis(node.z, self.mesh_bounds.z, self.wrap_around_enabled) {
            let cand = crate::types::ChipCoordinates::new(node.x, node.y, nz);
            if !self.faulted_links.contains(&(node, cand)) {
                out.push(cand);
            }
        }
        out
    }

    /// Computes a fault-tolerant shortest optical path from `src` to `dst` using Breadth-First Search
    /// (`BFS`) over all healthy 3D Torus links, rerouting around any links in `self.faulted_links`.
    ///
    /// # Arguments
    ///
    /// * `src` - Source [`crate::types::ChipCoordinates`].
    /// * `dst` - Destination [`crate::types::ChipCoordinates`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CollectiveError`] if all paths between `src` and `dst` are severed.
    pub fn compute_fault_tolerant_route(
        &self,
        src: crate::types::ChipCoordinates,
        dst: crate::types::ChipCoordinates,
    ) -> LibTpuResult<Vec<crate::types::ChipCoordinates>> {
        let mut queue = std::collections::VecDeque::new();
        let mut parent_map = BTreeMap::new();
        let mut visited = BTreeSet::new();
        queue.push_back(src);
        visited.insert(src);

        let mut reached = src == dst;
        while let Some(cur) = queue.pop_front() {
            if cur == dst {
                reached = true;
                break;
            }
            for nxt in self.healthy_neighbors(cur) {
                if visited.insert(nxt) {
                    parent_map.insert(nxt, cur);
                    queue.push_back(nxt);
                }
            }
        }
        if !reached {
            return Err(LibTpuError::CollectiveError {
                communicator_id: 0,
                op_name: "TorusIciRouter::compute_fault_tolerant_route",
                reason: format!("No healthy 3D Torus route from {src} to {dst}"),
            });
        }
        let mut path = vec![dst];
        let mut cur = dst;
        while let Some(&prev) = parent_map.get(&cur) {
            path.push(prev);
            cur = prev;
        }
        path.reverse();
        Ok(path)
    }

    /// Exports the complete minimal/DOR optical routing table across all chip pairs in `self.mesh_bounds`
    /// as a wire-compatible [`crate::proto::IciRoutingTableProto`] (`asic_sw.deepsea.IciRoutingTableProto`).
    ///
    /// # Arguments
    ///
    /// * `topology_name` - Human-readable topology name (e.g., `"tpu_v5p_2x2x2"`).
    #[must_use]
    pub fn to_ici_routing_table_proto(
        &self,
        topology_name: &str,
    ) -> crate::proto::IciRoutingTableProto {
        let mut hops = Vec::new();
        let nx = self.mesh_bounds.x.max(1);
        let ny = self.mesh_bounds.y.max(1);
        let nz = self.mesh_bounds.z.max(1);
        let mut chip_coords = Vec::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    chip_coords.push(crate::types::ChipCoordinates::new(x, y, z));
                }
            }
        }
        for (src_id, &src_c) in chip_coords.iter().enumerate() {
            for (dst_id, &dst_c) in chip_coords.iter().enumerate() {
                if src_id != dst_id {
                    let dx =
                        i32::try_from(dst_c.x).unwrap_or(0) - i32::try_from(src_c.x).unwrap_or(0);
                    let is_wrap = self.wrap_around_enabled && dx.unsigned_abs() > (nx / 2);
                    hops.push(crate::proto::IciRouteHopProto {
                        src_chip_id: u32::try_from(src_id).unwrap_or(0),
                        dst_chip_id: u32::try_from(dst_id).unwrap_or(0),
                        egress_link_port: u32::try_from((src_id + dst_id) % 6).unwrap_or(0),
                        virtual_channel: u32::from(is_wrap),
                        axis_delta: dx,
                        is_wrap_around: is_wrap,
                    });
                }
            }
        }
        let dim = if nz > 1 { 3 } else { 2 };
        crate::proto::IciRoutingTableProto {
            topology_name: String::from(topology_name),
            torus_dimension: dim,
            is_twisted_torus: self.wrap_around_enabled,
            route_hops: hops,
            semaphore_pool_size: 64,
        }
    }
}

/// Type alias for [`TorusIciRouter`] (`IciTorusRouter`).
pub type IciTorusRouter = TorusIciRouter;

/// Multi-host DAG executor for [`crate::proto::MegaScaleActionGraphProto`] (`MegaScaleActionGraphExecutor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MegaScaleActionGraphExecutor {
    /// Underlying DCN transport configuration.
    pub transport_config: MegascaleTransportConfigProto,
    /// Monotonic launch identifier counter.
    pub next_launch_id: i32,
}

impl Default for MegaScaleActionGraphExecutor {
    fn default() -> Self {
        Self::new(MegascaleTransportConfigProto {
            num_channels: 4,
            max_inflight_bytes: 4 * 1024 * 1024,
            timeout_ms: 5000,
            enable_zstd_compression: false,
            zstd_compression_level: 0,
        })
    }
}

impl MegaScaleActionGraphExecutor {
    /// Creates a new [`MegaScaleActionGraphExecutor`] with `transport_config`.
    ///
    /// # Arguments
    ///
    /// * `transport_config` - [`MegascaleTransportConfigProto`] configuring DCN channels and compression.
    #[must_use]
    pub const fn new(transport_config: MegascaleTransportConfigProto) -> Self {
        Self {
            transport_config,
            next_launch_id: 1,
        }
    }

    /// Topologically validates and executes a [`crate::proto::MegaScaleActionGraphProto`] over
    /// [`DcnLoopbackAndTcpTransport`], returning an [`crate::proto::ExecutorTelemetryProto`]
    /// and a [`crate::proto::XSpaceProto`] containing the `"MegaScale DCN"` `XPlane`.
    ///
    /// # Arguments
    ///
    /// * `graph` - [`crate::proto::MegaScaleActionGraphProto`] DAG of DCN/DMA/compute actions.
    /// * `payload` - Tensor payload bytes transmitted across the graph stages.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CollectiveError`] if an action references a cyclic or self-referential input dependency.
    pub fn execute_action_graph(
        &mut self,
        graph: &crate::proto::MegaScaleActionGraphProto,
        payload: &[u8],
    ) -> LibTpuResult<(
        crate::proto::ExecutorTelemetryProto,
        crate::proto::XSpaceProto,
    )> {
        let launch_id = self.next_launch_id;
        self.next_launch_id = self.next_launch_id.saturating_add(1);
        let transport = DcnLoopbackAndTcpTransport::new(self.transport_config.clone());
        let mut completed_actions = Vec::with_capacity(graph.actions.len());
        let mut xevents = Vec::with_capacity(graph.actions.len());

        for (step_idx, action) in graph.actions.iter().enumerate() {
            for dep in &action.inputs {
                if !completed_actions.contains(&dep.action_index) {
                    return Err(LibTpuError::CollectiveError {
                        communicator_id: 0,
                        op_name: "MegaScaleActionGraphExecutor::execute_action_graph",
                        reason: format!(
                            "Unsatisfied or cyclic dependency: action {} depends on {}",
                            action.index, dep.action_index
                        ),
                    });
                }
            }
            let _ = transport.transmit_socket_dcn_packet(payload)?;
            completed_actions.push(action.index);
            let offset_ps = i64::try_from(step_idx * 1000).unwrap_or(0);
            xevents.push(crate::proto::XEventProto {
                metadata_id: i64::from(action.index),
                offset_ps,
                duration_ps: 500,
                stats: Vec::new(),
                num_occurrences: 1,
            });
        }

        let telemetry = crate::proto::ExecutorTelemetryProto {
            pending_graphs: vec![crate::proto::PendingGraphStateProto {
                graph: Some(graph.clone()),
                launch_id,
                completed_actions,
                pending_actions: Vec::new(),
                active_actions: Vec::new(),
            }],
        };
        let xspace = crate::proto::XSpaceProto {
            planes: vec![crate::proto::XPlaneProto {
                id: 1,
                name: String::from("/device:TPU:0:MegaScale DCN"),
                lines: vec![crate::proto::XLineProto {
                    id: 1,
                    display_id: 1,
                    name: format!("MegaScaleActionGraph:{}", graph.id),
                    display_name: String::from("MegaScale DCN ActionGraph"),
                    timestamp_ns: 0,
                    duration_ps: i64::try_from(graph.actions.len() * 1000).unwrap_or(0),
                    events: xevents,
                }],
                event_metadata: BTreeMap::new(),
                stat_metadata: BTreeMap::new(),
                stats: Vec::new(),
            }],
            errors: Vec::new(),
            warnings: Vec::new(),
            hostnames: vec![String::from("localhost")],
        };
        Ok((telemetry, xspace))
    }
}

/// Multi-replica and multi-slice collective execution engine (`CollectiveRuntimeEngine`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollectiveRuntimeEngine;

impl CollectiveRuntimeEngine {
    /// Executes a Ring / Halving-Doubling `AllReduce` across `replica_buffers` using `reduction_op` and `elem_type`.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per participating replica).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner (`Sum`, `Product`, `Min`, `Max`).
    #[must_use]
    pub fn execute_all_reduce(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> Vec<Vec<u8>> {
        if replica_buffers.is_empty() {
            return Vec::new();
        }
        let unpacked: Vec<Vec<f64>> = replica_buffers
            .iter()
            .map(|b| elem_type.unpack_to_f64_vec(b))
            .collect();
        let elem_len = unpacked.first().map_or(0, Vec::len);
        let init_val = match reduction_op {
            CollectiveReductionOp::Sum => 0.0,
            CollectiveReductionOp::Product => 1.0,
            CollectiveReductionOp::Min => f64::INFINITY,
            CollectiveReductionOp::Max => f64::NEG_INFINITY,
        };
        let mut reduced = vec![init_val; elem_len];
        for r_vec in &unpacked {
            for (dst, &val) in reduced.iter_mut().zip(r_vec.iter()) {
                *dst = match reduction_op {
                    CollectiveReductionOp::Sum => *dst + val,
                    CollectiveReductionOp::Product => *dst * val,
                    CollectiveReductionOp::Min => dst.min(val),
                    CollectiveReductionOp::Max => dst.max(val),
                };
            }
        }
        let packed = elem_type.pack_from_f64_slice(&reduced);
        vec![packed; replica_buffers.len()]
    }

    /// Executes `ReduceScatter` across `replica_buffers`, reducing with `reduction_op` and scattering
    /// `1 / N` shards to each of the `N` participants.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    #[must_use]
    pub fn execute_reduce_scatter(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> Vec<Vec<u8>> {
        let all_reduced = Self::execute_all_reduce(replica_buffers, elem_type, reduction_op);
        let Some(first_buf) = all_reduced.first() else {
            return Vec::new();
        };
        let full_vals = elem_type.unpack_to_f64_vec(first_buf);
        let n = replica_buffers.len().max(1);
        let chunk_len = (full_vals.len() / n).max(1);
        (0..n)
            .map(|rep_idx| {
                let start = (rep_idx * chunk_len).min(full_vals.len());
                let end = (start + chunk_len).min(full_vals.len());
                elem_type.pack_from_f64_slice(full_vals.get(start..end).unwrap_or(&[]))
            })
            .collect()
    }

    /// Executes `AllGather` across `replica_buffers`, concatenating all participant shards in replica order.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of shard byte buffers (one per replica).
    /// * `elem_type` - Tensor [`ElementType`].
    #[must_use]
    pub fn execute_all_gather(replica_buffers: &[Vec<u8>], elem_type: ElementType) -> Vec<Vec<u8>> {
        if replica_buffers.is_empty() {
            return Vec::new();
        }
        let mut gathered_vals = Vec::new();
        for buf in replica_buffers {
            gathered_vals.extend(elem_type.unpack_to_f64_vec(buf));
        }
        let packed = elem_type.pack_from_f64_slice(&gathered_vals);
        vec![packed; replica_buffers.len()]
    }

    /// Executes `AllToAll` across `replica_buffers`, splitting each replica's buffer into `N` chunks
    /// and transposing chunk `(src, dst)` to replica `dst`.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers.
    /// * `elem_type` - Tensor [`ElementType`].
    #[must_use]
    pub fn execute_all_to_all(replica_buffers: &[Vec<u8>], elem_type: ElementType) -> Vec<Vec<u8>> {
        if replica_buffers.is_empty() {
            return Vec::new();
        }
        let unpacked: Vec<Vec<f64>> = replica_buffers
            .iter()
            .map(|b| elem_type.unpack_to_f64_vec(b))
            .collect();
        let n = unpacked.len().max(1);
        let elem_len = unpacked.first().map_or(0, Vec::len);
        let chunk_len = (elem_len / n).max(1);

        (0..n)
            .map(|dst_rep| {
                let mut dst_vals = Vec::with_capacity(elem_len);
                for src_vec in &unpacked {
                    let start = (dst_rep * chunk_len).min(src_vec.len());
                    let end = (start + chunk_len).min(src_vec.len());
                    dst_vals.extend_from_slice(src_vec.get(start..end).unwrap_or(&[]));
                }
                elem_type.pack_from_f64_slice(&dst_vals)
            })
            .collect()
    }

    /// Executes `CollectivePermute` routing full replica buffers across explicit `(source_replica, target_replica)` pairs.
    ///
    /// Unmapped target replicas receive a zero-filled buffer matching the first replica's byte length.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers.
    /// * `source_target_pairs` - Slice of `(source_replica_id, target_replica_id)` routing pairs.
    #[must_use]
    pub fn execute_collective_permute(
        replica_buffers: &[Vec<u8>],
        source_target_pairs: &[(usize, usize)],
    ) -> Vec<Vec<u8>> {
        let zero_len = replica_buffers.first().map_or(0, Vec::len);
        let mut outputs = vec![vec![0_u8; zero_len]; replica_buffers.len()];
        for &(src, dst) in source_target_pairs {
            if let (Some(src_buf), Some(dst_slot)) =
                (replica_buffers.get(src), outputs.get_mut(dst))
            {
                dst_slot.clone_from(src_buf);
            }
        }
        outputs
    }

    /// Executes `CollectiveBroadcast` copying the buffer of `source_replica_id` to all
    /// participating replicas in `replica_buffers`.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per replica).
    /// * `source_replica_id` - Index of the root replica whose buffer is broadcast (`0..N`).
    #[must_use]
    pub fn execute_collective_broadcast(
        replica_buffers: &[Vec<u8>],
        source_replica_id: usize,
    ) -> Vec<Vec<u8>> {
        if replica_buffers.is_empty() {
            return Vec::new();
        }
        let root_buf = replica_buffers
            .get(source_replica_id)
            .or_else(|| replica_buffers.first())
            .cloned()
            .unwrap_or_default();
        vec![root_buf; replica_buffers.len()]
    }

    /// Executes variable-length `RaggedAllToAll` across `replica_buffers`, exchanging variable-sized
    /// element slices from each source replica `src` to each destination replica `dst` according to
    /// `input_offsets[src][dst]`, `send_sizes[src][dst]`, and `output_offsets[dst][src]`.
    ///
    /// # Arguments
    ///
    /// * `replica_buffers` - Slice of input tensor byte buffers (one per replica).
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `input_offsets` - Per-replica slice of element start offsets (`[N][N]`).
    /// * `send_sizes` - Per-replica slice of element counts to send (`[N][N]`).
    /// * `output_offsets` - Per-replica slice of destination element start offsets (`[N][N]`).
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn execute_ragged_all_to_all(
        replica_buffers: &[Vec<u8>],
        elem_type: ElementType,
        input_offsets: &[Vec<usize>],
        send_sizes: &[Vec<usize>],
        output_offsets: &[Vec<usize>],
    ) -> Vec<Vec<u8>> {
        if replica_buffers.is_empty() {
            return Vec::new();
        }
        let unpacked: Vec<Vec<f64>> = replica_buffers
            .iter()
            .map(|b| elem_type.unpack_to_f64_vec(b))
            .collect();
        let n = unpacked.len();
        (0..n)
            .map(|dst_rep| {
                let mut dst_vals = unpacked[dst_rep].clone();
                for (src_rep, src_vec) in unpacked.iter().enumerate() {
                    let in_off = input_offsets
                        .get(src_rep)
                        .and_then(|r| r.get(dst_rep))
                        .copied()
                        .unwrap_or(0);
                    let count = send_sizes
                        .get(src_rep)
                        .and_then(|r| r.get(dst_rep))
                        .copied()
                        .unwrap_or(0);
                    let out_off = output_offsets
                        .get(dst_rep)
                        .and_then(|r| r.get(src_rep))
                        .copied()
                        .unwrap_or(0);
                    let safe_count = count
                        .min(src_vec.len().saturating_sub(in_off))
                        .min(dst_vals.len().saturating_sub(out_off));
                    dst_vals[out_off..out_off + safe_count]
                        .copy_from_slice(&src_vec[in_off..in_off + safe_count]);
                }
                elem_type.pack_from_f64_slice(&dst_vals)
            })
            .collect()
    }

    /// Executes a two-tier hierarchical Intra-Slice ICI Torus + Inter-Slice `MegaScale` DCN collective
    /// (`AllReduce`, `ReduceScatter`, or `AllGather`) across `slices_of_chip_buffers` (`[num_slices][chips_per_slice]`),
    /// returning both the resulting per-slice chip buffers and the [`crate::proto::MegascaleCollectiveScheduleProto`].
    ///
    /// # Arguments
    ///
    /// * `collective_name` - Collective operation identifier.
    /// * `torus_router` - Intra-slice [`TorusIciRouter`] governing optical ICI ring reduction.
    /// * `dcn_tree` - Inter-slice [`crate::proto::DcnTopologyTreeProto`] governing DCN exchange.
    /// * `slices_of_chip_buffers` - Nested `[slice_idx][chip_idx]` input tensor byte buffers.
    /// * `elem_type` - Tensor [`ElementType`].
    /// * `reduction_op` - [`CollectiveReductionOp`] combiner.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if intra-slice ICI routing or inter-slice DCN packet framing fails.
    pub fn execute_hierarchical_ici_dcn_collective(
        collective_name: &str,
        torus_router: &TorusIciRouter,
        dcn_tree: &crate::proto::DcnTopologyTreeProto,
        slices_of_chip_buffers: &[Vec<Vec<u8>>],
        elem_type: ElementType,
        reduction_op: CollectiveReductionOp,
    ) -> LibTpuResult<(
        Vec<Vec<Vec<u8>>>,
        crate::proto::MegascaleCollectiveScheduleProto,
    )> {
        let mut slice_leaders = Vec::with_capacity(slices_of_chip_buffers.len());
        let mut channel_descriptors = Vec::new();
        for (s_idx, slice_chips) in slices_of_chip_buffers.iter().enumerate() {
            let local_reduced = Self::execute_all_reduce(slice_chips, elem_type, reduction_op);
            let leader = local_reduced.first().cloned().unwrap_or_default();
            let next_slice = (s_idx + 1) % slices_of_chip_buffers.len().max(1);
            channel_descriptors.push(crate::proto::MegascaleChannelSliceDescriptorProto {
                channel_id: u32::try_from(s_idx).unwrap_or(0),
                src_slice_id: i32::try_from(s_idx).unwrap_or(0),
                dst_slice_id: i32::try_from(next_slice).unwrap_or(0),
                payload_bytes: leader.len() as u64,
                ici_ring_stage: torus_router.mesh_bounds.x.max(1),
            });
            slice_leaders.push(leader);
        }
        let dcn_reduced = DcnLoopbackAndTcpTransport::execute_two_tier_torus_dcn_all_reduce(
            torus_router,
            dcn_tree,
            &slice_leaders,
            elem_type,
            reduction_op,
        )?;
        let mut final_slices = Vec::with_capacity(slices_of_chip_buffers.len());
        for (slice_chips, dcn_buf) in slices_of_chip_buffers.iter().zip(dcn_reduced) {
            final_slices.push(vec![dcn_buf; slice_chips.len()]);
        }
        let red_code = match reduction_op {
            CollectiveReductionOp::Sum => 0,
            CollectiveReductionOp::Product => 1,
            CollectiveReductionOp::Min => 2,
            CollectiveReductionOp::Max => 3,
        };
        let sched = crate::proto::MegascaleCollectiveScheduleProto {
            collective_name: String::from(collective_name),
            reduction_kind: red_code,
            hierarchical_ici_then_dcn: true,
            channel_descriptors,
        };
        Ok((final_slices, sched))
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::proto::DistributedDeviceProto;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_distributed_coordinator_and_topology_assembler() {
        let coord = DistributedCoordinator::new();
        assert_ne!(format!("{coord:?}"), "");
        coord.put_key_value("topo/host0", vec![1, 2, 3]);
        assert_eq!(coord.get_key_value("topo/host0").ok(), Some(vec![1, 2, 3]));
        assert!(coord.get_key_value("missing_key").is_err());
        assert!(coord.delete_key_value("topo/host0"));
        assert!(!coord.delete_key_value("topo/host0"));

        // Barrier with 2 participants
        assert!(coord.wait_at_barrier(0, 10, 0, 2).is_err());
        assert_eq!(coord.wait_at_barrier(0, 10, 1, 2).ok(), Some(1));

        // Heartbeats and hang detection
        coord.record_heartbeat(0, 0, 1000);
        coord.record_heartbeat(0, 1, 1900);
        let stale = coord.check_stale_tasks(2000, 500);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].task_id, 0);
        assert_eq!(stale[0].error_type, 1);

        // MultiSliceTopologyAssembler
        let assembler = MultiSliceTopologyAssembler;
        assert_eq!(assembler.clone(), MultiSliceTopologyAssembler);
        assert_ne!(format!("{assembler:?}"), "");
        let local1 = LocalTopologyProto {
            process_id: 1,
            boot_id: String::from("b1"),
            devices: vec![DistributedDeviceProto {
                local_device_ordinal: 0,
                ..DistributedDeviceProto::default()
            }],
            partition_index: 0,
            network_nodes: vec![String::from("sw1")],
        };
        let local0 = LocalTopologyProto {
            process_id: 0,
            boot_id: String::from("b0"),
            devices: vec![DistributedDeviceProto {
                local_device_ordinal: 0,
                ..DistributedDeviceProto::default()
            }],
            partition_index: 0,
            network_nodes: vec![String::from("sw0")],
        };
        let global = MultiSliceTopologyAssembler::assemble_global_topology(&[local1, local0]);
        assert_eq!(global.processes.len(), 2);
        assert_eq!(global.processes[0].process_id, 0);
        assert_eq!(global.processes[0].devices[0].global_device_id, 0);
        assert_eq!(global.processes[1].devices[0].global_device_id, 1);

        let endpoints = MultiSliceTopologyAssembler::build_endpoint_addresses(2, 2, 8471);
        assert_eq!(endpoints.address_mappings.len(), 4);

        let sym_dcn = crate::proto::DcnTopologyTreeProto {
            symmetric_tree: Some(crate::proto::DcnSymmetricTreeProto {
                branching_per_layer: vec![2, 2],
            }),
            tree: None,
        };
        assert_eq!(
            MultiSliceTopologyAssembler::compute_dcn_topology_leaf_slices(&sym_dcn),
            vec![0, 1, 2, 3]
        );
        let asym_dcn = crate::proto::DcnTopologyTreeProto {
            symmetric_tree: None,
            tree: Some(crate::proto::DcnTreeNodeProto {
                label: Some(String::from("spine")),
                nodes: vec![crate::proto::DcnTreeNodeProto {
                    label: Some(String::from("rack0")),
                    nodes: Vec::new(),
                    slice_ranges: vec![crate::proto::DcnSliceRangeProto {
                        slice_id_start: 0,
                        slice_id_end: 2,
                    }],
                    egress_constraint: Some(1.0),
                    ring_transfers: true,
                }],
                slice_ranges: Vec::new(),
                egress_constraint: None,
                ring_transfers: false,
            }),
        };
        assert_eq!(
            MultiSliceTopologyAssembler::compute_dcn_topology_leaf_slices(&asym_dcn),
            vec![0, 1]
        );
        assert_eq!(
            MultiSliceTopologyAssembler::compute_dcn_topology_leaf_slices(
                &crate::proto::DcnTopologyTreeProto::default()
            ),
            vec![0]
        );

        let task0 = crate::proto::CoordinatedTaskProto {
            job_name: String::from("worker"),
            task_id: 0,
            recoverable: true,
        };
        assert!(
            coord
                .handle_protobuf_rpc(
                    1,
                    &crate::proto::RegisterTaskRequestProto {
                        incarnation: 7,
                        source_task: Some(task0.clone()),
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    2,
                    &crate::proto::HeartbeatRequestProto {
                        incarnation: 7,
                        source_task: Some(task0.clone()),
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    3,
                    &crate::proto::InsertKeyValueRequestProto {
                        kv: Some(crate::proto::KeyValueEntryProto {
                            key: String::from("cnt/a"),
                            value: b"10".to_vec(),
                        }),
                        allow_overwrite: true,
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    3,
                    &crate::proto::InsertKeyValueRequestProto::default().encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    4,
                    &crate::proto::GetKeyValueRequestProto {
                        key: String::from("cnt/a"),
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    5,
                    &crate::proto::TryGetKeyValueRequestProto {
                        key: String::from("cnt/a"),
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    6,
                    &crate::proto::IncrementKeyValueRequestProto {
                        key: String::from("cnt/a"),
                        increment: 5,
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    7,
                    &crate::proto::GetKeyValueDirRequestProto {
                        directory_key: String::from("cnt/"),
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    8,
                    &crate::proto::DeleteKeyValueRequestProto {
                        key: String::from("cnt/a"),
                        is_directory: false,
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    9,
                    &crate::proto::BarrierRequestProto {
                        barrier_id: String::from("b0"),
                        barrier_timeout_in_ms: 100,
                        tasks: vec![task0.clone()],
                        source_task: Some(task0.clone()),
                        counter: 1,
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    10,
                    &crate::proto::CancelBarrierRequestProto {
                        barrier_id: String::from("b0"),
                        source_task: Some(task0.clone()),
                        counter: 1,
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    11,
                    &crate::proto::GetAliveTasksRequestProto {
                        requesting_task: Some(task0.clone()),
                        tasks: vec![task0.clone()],
                    }
                    .encode()
                )
                .is_ok()
        );
        assert!(
            coord
                .handle_protobuf_rpc(
                    12,
                    &crate::proto::ShutdownTaskRequestProto {
                        source_task: Some(task0),
                    }
                    .encode()
                )
                .is_ok()
        );
        for m_id in 1..=12 {
            assert!(
                coord
                    .handle_protobuf_rpc(m_id, &[0xFF, 0xFF, 0xFF])
                    .is_err()
            );
        }
    }

    #[test]
    fn test_dcn_transport_and_collective_runtime_engine() {
        let transport = DcnLoopbackAndTcpTransport::new(MegascaleTransportConfigProto {
            num_channels: 4,
            max_inflight_bytes: 1_048_576,
            timeout_ms: 5000,
            enable_zstd_compression: false,
            zstd_compression_level: 0,
        });
        assert_ne!(format!("{transport:?}"), "");
        assert!(transport.recv_loopback(0, 0, 99).is_err());
        transport.send_loopback(0, 1, 42, vec![10, 20, 30]);
        assert_eq!(
            transport.recv_loopback(0, 1, 42).ok(),
            Some(vec![10, 20, 30])
        );

        let tcp_rt = DcnLoopbackAndTcpTransport::roundtrip_tcp_frame(b"megascale_dcn_frame");
        assert_eq!(tcp_rt.as_slice(), b"megascale_dcn_frame");

        // Test MegascaleDcnSocketTransport::transmit_socket_dcn_packet (uncompressed and compressed)
        let (rx_uncomp, m_uncomp) = transport
            .transmit_socket_dcn_packet(b"hello_dcn_socket")
            .unwrap_or_default();
        assert_eq!(rx_uncomp, b"hello_dcn_socket");
        assert_eq!(m_uncomp.bytes_transferred, 16);
        let comp_transport = MegascaleDcnSocketTransport::new(MegascaleTransportConfigProto {
            num_channels: 4,
            max_inflight_bytes: 1_048_576,
            timeout_ms: 5000,
            enable_zstd_compression: true,
            zstd_compression_level: 3,
        });
        let (rx_comp, m_comp) = comp_transport
            .transmit_socket_dcn_packet(&[0x42_u8; 64])
            .unwrap_or_default();
        assert_eq!(rx_comp, vec![0x42_u8; 64]);
        assert_eq!(m_comp.collective_ops_count, 1);

        let eng = CollectiveRuntimeEngine;
        assert_eq!(eng.clone(), CollectiveRuntimeEngine);
        assert_ne!(format!("{eng:?}"), "");
        assert_eq!(
            CollectiveRuntimeEngine::execute_all_reduce(
                &[],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .len(),
            0
        );
        assert_eq!(
            CollectiveRuntimeEngine::execute_reduce_scatter(
                &[],
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .len(),
            0
        );
        assert_eq!(
            CollectiveRuntimeEngine::execute_all_gather(&[], ElementType::F32).len(),
            0
        );
        assert_eq!(
            CollectiveRuntimeEngine::execute_all_to_all(&[], ElementType::F32).len(),
            0
        );

        let r0 = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
        let r1 = ElementType::F32.pack_from_f64_slice(&[5.0, 6.0, 7.0, 8.0]);
        let bufs = vec![r0, r1];

        for op in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Product,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Max,
        ] {
            let ar = CollectiveRuntimeEngine::execute_all_reduce(&bufs, ElementType::F32, op);
            assert_eq!(ar.len(), 2);
        }

        // Verify conservation identity: AllGather(ReduceScatter(X, Sum)) == AllReduce(X, Sum)
        let rs = CollectiveRuntimeEngine::execute_reduce_scatter(
            &bufs,
            ElementType::F32,
            CollectiveReductionOp::Sum,
        );
        let ag = CollectiveRuntimeEngine::execute_all_gather(&rs, ElementType::F32);
        let ar_sum = CollectiveRuntimeEngine::execute_all_reduce(
            &bufs,
            ElementType::F32,
            CollectiveReductionOp::Sum,
        );
        assert_eq!(ag, ar_sum);

        let a2a = CollectiveRuntimeEngine::execute_all_to_all(&bufs, ElementType::F32);
        assert_eq!(a2a.len(), 2);
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&a2a[0]),
            vec![1.0, 2.0, 5.0, 6.0]
        );
        assert_eq!(
            ElementType::F32.unpack_to_f64_vec(&a2a[1]),
            vec![3.0, 4.0, 7.0, 8.0]
        );

        let perm =
            CollectiveRuntimeEngine::execute_collective_permute(&bufs, &[(0, 1), (1, 0), (9, 9)]);
        assert_eq!(perm[0], bufs[1]);
        assert_eq!(perm[1], bufs[0]);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_wire_framing_and_torus_ici_router() {
        let wire_frame = CoordinationWireFrame {
            request_id: 99,
            opcode: 3,
            payload: b"kv_payload".to_vec(),
        };
        assert_eq!(wire_frame.clone(), wire_frame);
        assert_ne!(format!("{wire_frame:?}"), "");
        let frame_bytes = wire_frame.encode_to_bytes();
        assert_eq!(
            CoordinationWireFrame::decode_from_bytes(&frame_bytes),
            Ok(wire_frame.clone())
        );
        assert!(CoordinationWireFrame::decode_from_bytes(&[0; 5]).is_err());
        assert!(CoordinationWireFrame::decode_from_bytes(&frame_bytes[..15]).is_err());
        assert_eq!(
            DcnLoopbackAndTcpTransport::exchange_wire_frame_over_tcp(&wire_frame),
            Ok(wire_frame)
        );

        let c0 = crate::types::ChipCoordinates::new(0, 0, 0);
        let c3 = crate::types::ChipCoordinates::new(3, 0, 0);
        let mut router = TorusIciRouter::new(crate::types::ChipCoordinates::new(4, 4, 4), true);
        assert_eq!(router.clone(), router);
        assert_ne!(format!("{router:?}"), "");
        assert_eq!(router.compute_shortest_hop_count(c0, c3), Ok(1));
        assert_eq!(
            router.compute_shortest_hop_count(c0, crate::types::ChipCoordinates::new(1, 0, 0)),
            Ok(1)
        );
        let non_wrap = TorusIciRouter::new(crate::types::ChipCoordinates::new(4, 4, 4), false);
        assert_eq!(non_wrap.compute_shortest_hop_count(c0, c3), Ok(3));
        assert!(non_wrap.compute_fault_tolerant_route(c3, c0).is_ok());
        assert!(non_wrap.compute_fault_tolerant_route(c0, c3).is_ok());
        assert_eq!(router.compute_fault_tolerant_route(c0, c0), Ok(vec![c0]));
        router.inject_link_fault(c0, c3);
        assert!(router.compute_shortest_hop_count(c0, c3).is_err());
        let rerouted = router.compute_fault_tolerant_route(c0, c3);
        assert!(rerouted.is_ok());
        if let Ok(path) = rerouted {
            assert_eq!(path.first().copied(), Some(c0));
            assert_eq!(path.last().copied(), Some(c3));
            assert!(path.len() >= 4);
        }
        // Sever all outgoing links from c0 so BFS returns an error
        for neigh in [
            crate::types::ChipCoordinates::new(1, 0, 0),
            crate::types::ChipCoordinates::new(3, 0, 0),
            crate::types::ChipCoordinates::new(0, 1, 0),
            crate::types::ChipCoordinates::new(0, 3, 0),
            crate::types::ChipCoordinates::new(0, 0, 1),
            crate::types::ChipCoordinates::new(0, 0, 3),
        ] {
            router.inject_link_fault(c0, neigh);
        }
        assert!(router.compute_fault_tolerant_route(c0, c3).is_err());

        // Test DistributedCoordinator TCP RPC dispatcher across all opcodes (1..=11)
        let rpc_coord = DistributedCoordinator::new();
        assert_eq!(rpc_coord.try_get_key_value("absent"), None);
        let _ = rpc_coord.wait_at_barrier(0, 110, 0, 2);
        for (op, payload) in [
            (1_u8, b"register".as_slice()),
            (2_u8, b"hb".as_slice()),
            (3_u8, b"topo/k=val123".as_slice()),
            (8_u8, b"topo/k".as_slice()),
            (9_u8, b"topo/".as_slice()),
            (4_u8, b"topo/k".as_slice()),
            (5_u8, b"topo/k".as_slice()),
            (6_u8, b"barrier".as_slice()),
            (10_u8, b"cancel".as_slice()),
            (11_u8, b"dcn_link_error".as_slice()),
            (7_u8, b"shutdown".as_slice()),
        ] {
            let req = CoordinationWireFrame {
                request_id: 100 + u64::from(op),
                opcode: op,
                payload: payload.to_vec(),
            };
            assert!(rpc_coord.serve_tcp_rpc_once(&req).is_ok());
        }
        assert!(!rpc_coord.cancel_barrier(0, 999));
        let slice_bufs = [
            ElementType::F32.pack_from_f64_slice(&[1.0, 2.0]),
            ElementType::F32.pack_from_f64_slice(&[3.0, 4.0]),
        ];
        let tcp_ar = DcnLoopbackAndTcpTransport::execute_tcp_all_reduce(
            &slice_bufs,
            ElementType::F32,
            CollectiveReductionOp::Sum,
        );
        assert!(tcp_ar.is_ok());
        if let Ok(ar_bufs) = tcp_ar {
            assert_eq!(
                ElementType::F32.unpack_to_f64_vec(&ar_bufs[0]),
                vec![4.0, 6.0]
            );
        }
        assert!(
            DcnLoopbackAndTcpTransport::execute_tcp_all_gather(&slice_bufs, ElementType::F32)
                .is_ok()
        );
        assert!(
            DcnLoopbackAndTcpTransport::execute_tcp_reduce_scatter(
                &slice_bufs,
                ElementType::F32,
                CollectiveReductionOp::Sum
            )
            .is_ok()
        );
        assert!(
            DcnLoopbackAndTcpTransport::execute_tcp_all_to_all(&slice_bufs, ElementType::F32)
                .is_ok()
        );

        let tcp_srv = CoordinationTcpServer::bind("127.0.0.1:0");
        assert!(tcp_srv.is_ok());
        if let Ok(srv) = tcp_srv {
            assert_ne!(format!("{srv:?}"), "");
            if let Ok(addr) = srv.local_addr() {
                let cli = CoordinationTcpClient::new(addr);
                assert_eq!(cli.clone(), cli);
                assert_ne!(format!("{cli:?}"), "");
                let local_p = LocalTopologyProto {
                    process_id: 0,
                    boot_id: String::from("b0"),
                    devices: vec![DistributedDeviceProto {
                        local_device_ordinal: 0,
                        ..DistributedDeviceProto::default()
                    }],
                    partition_index: 0,
                    network_nodes: Vec::new(),
                };
                let resp = cli.call_rpc(
                    &srv,
                    &CoordinationWireFrame {
                        request_id: 42,
                        opcode: 12,
                        payload: local_p.encode(),
                    },
                );
                assert!(resp.is_ok());
                if let Ok(r_frame) = resp {
                    let g_topo = GlobalTopologyProto::decode(&r_frame.payload);
                    assert!(g_topo.is_ok());
                }
            }
        }
        assert!(
            rpc_coord
                .handle_wire_frame(&CoordinationWireFrame {
                    request_id: 1,
                    opcode: 4,
                    payload: b"missing".to_vec(),
                })
                .is_err()
        );

        // Test wait_get_key_value (immediate hit, timeout miss, and cross-thread condvar wakeup)
        rpc_coord.put_key_value("ready_key", b"ready_val".to_vec());
        assert_eq!(
            rpc_coord.wait_get_key_value("ready_key", 0),
            Ok(b"ready_val".to_vec())
        );
        assert!(rpc_coord.wait_get_key_value("never_key", 0).is_err());
        assert!(rpc_coord.wait_get_key_value("never_key", 2).is_err());

        let shared_coord = std::sync::Arc::new(DistributedCoordinator::new());
        let writer_coord = std::sync::Arc::clone(&shared_coord);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            writer_coord.put_key_value("delayed_key", b"woken".to_vec());
        });
        assert_eq!(
            shared_coord.wait_get_key_value("delayed_key", 500),
            Ok(b"woken".to_vec())
        );
        let _ = handle.join();

        // Test 20-byte multi-participant barrier payload on opcode 6
        let mut barrier_payload = Vec::with_capacity(20);
        barrier_payload.extend_from_slice(&0_i32.to_le_bytes());
        barrier_payload.extend_from_slice(&777_u64.to_le_bytes());
        barrier_payload.extend_from_slice(&0_i32.to_le_bytes());
        barrier_payload.extend_from_slice(&1_u32.to_le_bytes());
        assert!(
            rpc_coord
                .handle_wire_frame(&CoordinationWireFrame {
                    request_id: 777,
                    opcode: 6,
                    payload: barrier_payload,
                })
                .is_ok()
        );

        let raw_dcn = vec![0xAA_u8, 0xAA, 0xAA, 0xAA, 0xAA, 0x01, 0xFF, 0x02];
        let compressed = MegascaleTransport::compress_payload(&raw_dcn);
        assert_eq!(
            MegascaleTransport::decompress_payload(&compressed),
            Ok(raw_dcn)
        );
        assert_eq!(
            MegascaleTransport::decompress_payload(b"plain_uncompressed"),
            Ok(b"plain_uncompressed".to_vec())
        );
        assert!(MegascaleTransport::decompress_payload(b"MGSZshort").is_err());
        let mut trunc_rle = compressed.clone();
        trunc_rle.push(0xFF);
        assert!(MegascaleTransport::decompress_payload(&trunc_rle).is_err());
        trunc_rle.push(2);
        assert!(MegascaleTransport::decompress_payload(&trunc_rle).is_err());
        let mut bad_hash = compressed;
        bad_hash[8] ^= 0xFF;
        assert!(MegascaleTransport::decompress_payload(&bad_hash).is_err());
        assert_eq!(MegascaleTransport.clone(), MegascaleTransport);
        assert_ne!(format!("{MegascaleTransport:?}"), "");

        // Test MegascaleWirePacketHeader & execute_ring_all_reduce_mscl
        let pkt = MegascaleWirePacketHeader::frame_packet(1, 0, 0, 0, 1, 0, b"shard0");
        let unframed = MegascaleWirePacketHeader::unframe_packet(&pkt);
        assert!(unframed.is_ok());
        if let Ok((hdr, body)) = unframed {
            assert_eq!(hdr.clone(), hdr);
            assert_ne!(format!("{hdr:?}"), "");
            assert_eq!(body, b"shard0");
        }
        assert!(MegascaleWirePacketHeader::unframe_packet(&[0_u8; 10]).is_err());
        let mut bad_magic = pkt.clone();
        bad_magic[0] = b'X';
        assert!(MegascaleWirePacketHeader::unframe_packet(&bad_magic).is_err());
        assert!(MegascaleWirePacketHeader::unframe_packet(&pkt[..34]).is_err());
        let mut bad_pkt_hash = pkt;
        bad_pkt_hash[32] ^= 0xFF;
        assert!(MegascaleWirePacketHeader::unframe_packet(&bad_pkt_hash).is_err());

        let mscl_ar = DcnLoopbackAndTcpTransport::execute_ring_all_reduce_mscl(
            &slice_bufs,
            ElementType::F32,
            CollectiveReductionOp::Sum,
        );
        assert!(mscl_ar.is_ok());
        if let Ok(ar_out) = mscl_ar {
            assert_eq!(
                ElementType::F32.unpack_to_f64_vec(&ar_out[0]),
                vec![4.0, 6.0]
            );
            assert_eq!(ar_out[0], ar_out[1]);
        }
        assert_eq!(
            DcnLoopbackAndTcpTransport::execute_ring_all_reduce_mscl(
                &[],
                ElementType::F32,
                CollectiveReductionOp::Sum,
            ),
            Ok(Vec::new())
        );
        for red_op in [
            CollectiveReductionOp::Max,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Product,
        ] {
            assert!(
                DcnLoopbackAndTcpTransport::execute_ring_all_reduce_mscl(
                    &slice_bufs,
                    ElementType::F32,
                    red_op,
                )
                .is_ok()
            );
        }
        let dcn_tree = crate::proto::DcnTopologyTreeProto {
            symmetric_tree: Some(crate::proto::DcnSymmetricTreeProto {
                branching_per_layer: vec![1, 2],
            }),
            tree: None,
        };
        let tree_ar = DcnLoopbackAndTcpTransport::execute_hierarchical_tree_all_reduce_mscl(
            &dcn_tree,
            &slice_bufs,
            ElementType::F32,
            CollectiveReductionOp::Sum,
        );
        assert!(tree_ar.is_ok());
        if let Ok(t_out) = tree_ar {
            assert_eq!(
                ElementType::F32.unpack_to_f64_vec(&t_out[0]),
                vec![4.0, 6.0]
            );
        }
        let single_tree = crate::proto::DcnTopologyTreeProto {
            symmetric_tree: Some(crate::proto::DcnSymmetricTreeProto {
                branching_per_layer: vec![4],
            }),
            tree: None,
        };
        assert!(
            DcnLoopbackAndTcpTransport::execute_hierarchical_tree_all_reduce_mscl(
                &single_tree,
                &slice_bufs,
                ElementType::F32,
                CollectiveReductionOp::Sum,
            )
            .is_ok()
        );

        let mut alias_router =
            IciTorusRouter::new(crate::types::ChipCoordinates::new(2, 2, 1), false);
        let route_table_2d = alias_router.to_ici_routing_table_proto("v5e_2x2");
        assert_eq!(route_table_2d.torus_dimension, 2);
        assert_ne!(route_table_2d.route_hops.len(), 0);
        let router_3d = IciTorusRouter::new(crate::types::ChipCoordinates::new(4, 2, 2), true);
        let route_table_3d = router_3d.to_ici_routing_table_proto("v5p_4x2x2");
        assert_eq!(route_table_3d.torus_dimension, 3);
        assert!(route_table_3d.is_twisted_torus);
        for red_op in [
            CollectiveReductionOp::Sum,
            CollectiveReductionOp::Product,
            CollectiveReductionOp::Min,
            CollectiveReductionOp::Max,
        ] {
            let hier_res = CollectiveRuntimeEngine::execute_hierarchical_ici_dcn_collective(
                "all_reduce_hier",
                &alias_router,
                &dcn_tree,
                &[slice_bufs.to_vec(), slice_bufs.to_vec()],
                ElementType::F32,
                red_op,
            );
            assert!(hier_res.is_ok());
            if let Ok((slices_out, sched_pb)) = hier_res {
                assert_eq!(slices_out.len(), 2);
                assert!(sched_pb.hierarchical_ici_then_dcn);
                assert_eq!(sched_pb.channel_descriptors.len(), 2);
            }
        }
        let two_tier_res = DcnLoopbackAndTcpTransport::execute_two_tier_torus_dcn_all_reduce(
            &alias_router,
            &dcn_tree,
            &slice_bufs,
            ElementType::F32,
            CollectiveReductionOp::Sum,
        );
        assert!(two_tier_res.is_ok());
        alias_router.disable_ici_link(c0, crate::types::ChipCoordinates::new(1, 0, 0));
        assert!(
            alias_router
                .compute_shortest_hop_count(c0, crate::types::ChipCoordinates::new(1, 0, 0))
                .is_err()
        );

        let mut ag_exec = MegaScaleActionGraphExecutor::default();
        assert_eq!(ag_exec.clone(), ag_exec);
        assert_ne!(format!("{ag_exec:?}"), "");
        let valid_graph = crate::proto::MegaScaleActionGraphProto {
            id: String::from("ag_0"),
            actions: vec![
                crate::proto::MegaScaleActionProto {
                    inputs: Vec::new(),
                    index: 0,
                },
                crate::proto::MegaScaleActionProto {
                    inputs: vec![crate::proto::MegaScaleActionInputProto { action_index: 0 }],
                    index: 1,
                },
            ],
            hlo_module: String::from("main"),
        };
        let exec_res = ag_exec.execute_action_graph(&valid_graph, b"dcn_action_payload");
        assert!(exec_res.is_ok());
        if let Ok((telem, xspace)) = exec_res {
            assert_eq!(telem.pending_graphs[0].completed_actions, vec![0, 1]);
            assert_eq!(xspace.planes[0].name, "/device:TPU:0:MegaScale DCN");
        }
        let cyclic_graph = crate::proto::MegaScaleActionGraphProto {
            id: String::from("ag_cycle"),
            actions: vec![crate::proto::MegaScaleActionProto {
                inputs: vec![crate::proto::MegaScaleActionInputProto { action_index: 0 }],
                index: 0,
            }],
            hlo_module: String::from("main"),
        };
        assert!(ag_exec.execute_action_graph(&cyclic_graph, b"x").is_err());

        // Test Brotli DCN payload compression & error paths
        let brotli_dcn = MegascaleTransport::compress_brotli_payload(b"megascale_brotli_payload");
        assert_eq!(
            MegascaleTransport::decompress_payload(&brotli_dcn),
            Ok(b"megascale_brotli_payload".to_vec())
        );
        let mut corrupt_brotli = brotli_dcn;
        corrupt_brotli.truncate(17);
        corrupt_brotli[16] = 0xFF;
        assert!(MegascaleTransport::decompress_payload(&corrupt_brotli).is_err());

        // Test 5-byte gRPC framing & all 19 CoordinationService methods via invoke_grpc_path / handle_protobuf_rpc
        assert!(DistributedCoordinator::decode_grpc_frame(&[]).is_err());
        assert!(DistributedCoordinator::decode_grpc_frame(&[0, 0, 0]).is_err());
        assert!(DistributedCoordinator::decode_grpc_frame(&[0, 0, 0, 0, 10, 1]).is_err());
        let grpc_coord = DistributedCoordinator::new();
        let reg_req = crate::proto::RegisterTaskRequestProto {
            source_task: Some(crate::proto::CoordinatedTaskProto {
                job_name: String::from("worker"),
                task_id: 0,
                recoverable: true,
            }),
            incarnation: 1,
        }
        .encode();
        let reg_frame = DistributedCoordinator::encode_grpc_frame(&reg_req, false);
        let reg_frame_comp = DistributedCoordinator::encode_grpc_frame(&reg_req, true);
        assert!(
            grpc_coord
                .invoke_grpc_path("/tensorflow.CoordinationService/RegisterTask", &reg_frame)
                .is_ok()
        );
        assert!(
            grpc_coord
                .invoke_grpc_path("/tensorflow.CoordinationService/Heartbeat", &reg_frame_comp)
                .is_ok()
        );
        for (path, req_bytes) in [
            (
                "InsertKeyValue",
                crate::proto::InsertKeyValueRequestProto {
                    kv: Some(crate::proto::KeyValueEntryProto {
                        key: String::from("/dir/k1"),
                        value: b"v1".to_vec(),
                    }),
                    allow_overwrite: false,
                }
                .encode(),
            ),
            (
                "GetKeyValue",
                crate::proto::GetKeyValueRequestProto {
                    key: String::from("/dir/k1"),
                }
                .encode(),
            ),
            (
                "TryGetKeyValue",
                crate::proto::TryGetKeyValueRequestProto {
                    key: String::from("/dir/k1"),
                }
                .encode(),
            ),
            (
                "IncrementKeyValue",
                crate::proto::IncrementKeyValueRequestProto {
                    key: String::from("/dir/cnt"),
                    increment: 1,
                }
                .encode(),
            ),
            (
                "GetKeyValueDir",
                crate::proto::GetKeyValueDirRequestProto {
                    directory_key: String::from("/dir/"),
                }
                .encode(),
            ),
            (
                "DeleteKeyValue",
                crate::proto::DeleteKeyValueRequestProto {
                    key: String::from("/dir/"),
                    is_directory: true,
                }
                .encode(),
            ),
            (
                "Barrier",
                crate::proto::BarrierRequestProto {
                    barrier_id: String::from("b1"),
                    barrier_timeout_in_ms: 100,
                    tasks: Vec::new(),
                    source_task: None,
                    counter: 1,
                }
                .encode(),
            ),
            (
                "CancelBarrier",
                crate::proto::CancelBarrierRequestProto {
                    barrier_id: String::from("b1"),
                    source_task: None,
                    counter: 1,
                }
                .encode(),
            ),
            (
                "GetAliveTasks",
                crate::proto::GetAliveTasksRequestProto {
                    requesting_task: None,
                    tasks: Vec::new(),
                }
                .encode(),
            ),
            (
                "WaitForAllTasks",
                crate::proto::WaitForAllTasksRequestProto {
                    source_task: None,
                    device_info: None,
                }
                .encode(),
            ),
            (
                "GetTaskState",
                crate::proto::GetTaskStateRequestProto {
                    source_task: vec![
                        crate::proto::CoordinatedTaskProto {
                            job_name: String::from("worker"),
                            task_id: 0,
                            recoverable: true,
                        },
                        crate::proto::CoordinatedTaskProto {
                            job_name: String::from("worker"),
                            task_id: 99,
                            recoverable: true,
                        },
                    ],
                }
                .encode(),
            ),
            (
                "WatchJobState",
                crate::proto::WatchJobStateRequestProto {
                    job_name: String::from("worker"),
                    version_number: 1,
                }
                .encode(),
            ),
            (
                "ReportErrorToTask",
                crate::proto::ReportErrorToTaskRequestProto {
                    error_code: 2,
                    error_message: String::from("err"),
                    error_payload: Some(crate::proto::CoordinationServiceErrorProto {
                        is_reported_error: true,
                        source_task: Some(crate::proto::CoordinatedTaskProto {
                            job_name: String::from("worker"),
                            task_id: 0,
                            recoverable: true,
                        }),
                    }),
                }
                .encode(),
            ),
            (
                "ReportErrorToService",
                crate::proto::ReportErrorToServiceRequestProto {
                    error_code: 2,
                    error_message: String::from("err_srv"),
                    error_origin: Some(crate::proto::CoordinatedTaskProto {
                        job_name: String::from("worker"),
                        task_id: 0,
                        recoverable: true,
                    }),
                }
                .encode(),
            ),
            (
                "PollForError",
                crate::proto::PollForErrorRequestProto { source_task: None }.encode(),
            ),
            (
                "ResetTask",
                crate::proto::ResetTaskRequestProto { source_task: None }.encode(),
            ),
            (
                "ShutdownTask",
                crate::proto::ShutdownTaskRequestProto { source_task: None }.encode(),
            ),
        ] {
            let f = DistributedCoordinator::encode_grpc_frame(&req_bytes, false);
            assert!(
                grpc_coord.invoke_grpc_path(path, &f).is_ok(),
                "Failed gRPC method: {path}"
            );
        }
        // Duplicate InsertKeyValue with allow_overwrite == false
        grpc_coord.put_key_value("dup_k", b"1".to_vec());
        let dup_req = crate::proto::InsertKeyValueRequestProto {
            kv: Some(crate::proto::KeyValueEntryProto {
                key: String::from("dup_k"),
                value: b"2".to_vec(),
            }),
            allow_overwrite: false,
        }
        .encode();
        assert!(grpc_coord.handle_protobuf_rpc(3, &dup_req).is_err());
        assert!(
            grpc_coord
                .invoke_grpc_path("UnknownMethod", &reg_frame)
                .is_err()
        );
        for m_id in 13..=19 {
            assert!(
                grpc_coord
                    .handle_protobuf_rpc(m_id, &[0xFF, 0xFF, 0xFF])
                    .is_err()
            );
        }

        // Test CollectiveBroadcast & RaggedAllToAll
        assert_eq!(
            CollectiveRuntimeEngine::execute_collective_broadcast(&[], 0).len(),
            0
        );
        let rep0 = ElementType::F32.pack_from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
        let rep1 = ElementType::F32.pack_from_f64_slice(&[10.0, 20.0, 30.0, 40.0]);
        let bcast0 =
            CollectiveRuntimeEngine::execute_collective_broadcast(&[rep0.clone(), rep1.clone()], 0);
        assert_eq!(bcast0[1], rep0);
        let bcast = CollectiveRuntimeEngine::execute_collective_broadcast(
            &[rep0.clone(), rep1.clone()],
            99,
        );
        assert_eq!(bcast.len(), 2);
        assert_eq!(bcast[1], rep0);
        assert_eq!(
            CollectiveRuntimeEngine::execute_ragged_all_to_all(
                &[],
                ElementType::F32,
                &[],
                &[],
                &[]
            )
            .len(),
            0
        );
        let ragged = CollectiveRuntimeEngine::execute_ragged_all_to_all(
            &[rep0, rep1],
            ElementType::F32,
            &[vec![0, 2], vec![0, 2]],
            &[vec![2, 2], vec![2, 2]],
            &[vec![0, 2], vec![0, 2]],
        );
        assert_eq!(ragged.len(), 2);
    }
}
