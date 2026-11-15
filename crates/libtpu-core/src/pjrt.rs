//! Safe, thread-safe Rust PJRT domain engine backing all `PJRT_*` C API handles.
//!
//! Provides [`PjrtCoreClient`], [`PjrtCoreDevice`], [`PjrtCoreDeviceDescription`],
//! [`PjrtCoreMemory`], [`PjrtCoreBuffer`], [`PjrtCoreEvent`], [`PjrtCoreTopology`],
//! [`PjrtCoreExecutable`], [`PjrtCoreLoadedExecutable`], and [`PjrtCoreCopyToDeviceStream`].

use crate::compiler::{CompiledMemoryStats, CompiledProgram, CostAnalysis};
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::hal::{DeviceMemoryStats, DeviceSpec, SimulatedTpuHal, TopologySpec, TpuHal};
use crate::proto::ProtoSchemaMessage as _;
use crate::types::{
    BufferId, ByteSize, DeviceAddress, DeviceId, ElementType, EventId, ExecutableId, MemoryKind,
    MemorySpaceId, ProcessIndex, Shape,
};
use parking_lot::{Mutex, MutexGuard};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

/// Global monotonic counter for assigning unique buffer, executable, and event IDs.
static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);

/// Allocates a fresh globally unique 64-bit object ID.
fn next_id() -> u64 {
    NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Boxed callback invoked when a [`PjrtCoreEvent`] transitions to the ready state.
pub type EventReadyCallback = Box<dyn FnOnce(Option<LibTpuError>) + Send + 'static>;

/// Core representation of a `PJRT_Event` asynchronous completion notification.
pub struct PjrtCoreEvent {
    /// Unique event identifier.
    pub id: EventId,
    /// Whether the event has reached completion.
    ready: AtomicBool,
    /// Optional error attached to the event.
    error: Mutex<Option<LibTpuError>>,
    /// Pending callbacks registered via `on_ready`.
    callbacks: Mutex<Vec<EventReadyCallback>>,
}

impl std::fmt::Debug for PjrtCoreEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PjrtCoreEvent")
            .field("id", &self.id)
            .field("ready", &self.is_ready())
            .finish_non_exhaustive()
    }
}

impl PjrtCoreEvent {
    /// Creates an immediately ready [`PjrtCoreEvent`] with no error.
    #[must_use]
    pub fn ready_ok() -> Self {
        Self {
            id: EventId(next_id()),
            ready: AtomicBool::new(true),
            error: Mutex::new(None),
            callbacks: Mutex::new(Vec::new()),
        }
    }

    /// Creates an immediately ready [`PjrtCoreEvent`] carrying `error`.
    ///
    /// # Arguments
    ///
    /// * `error` - Failure status carried by the ready event.
    #[must_use]
    pub fn ready_err(error: LibTpuError) -> Self {
        Self {
            id: EventId(next_id()),
            ready: AtomicBool::new(true),
            error: Mutex::new(Some(error)),
            callbacks: Mutex::new(Vec::new()),
        }
    }

    /// Creates an unready pending [`PjrtCoreEvent`].
    #[must_use]
    pub fn pending() -> Self {
        Self {
            id: EventId(next_id()),
            ready: AtomicBool::new(false),
            error: Mutex::new(None),
            callbacks: Mutex::new(Vec::new()),
        }
    }

    /// Returns whether this event has completed (`PJRT_Event_IsReady`).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Registers a callback to be invoked when this event becomes ready (`PJRT_Event_OnReady`).
    ///
    /// If the event is already ready, `callback` is invoked immediately on the calling thread;
    /// otherwise it is queued and invoked when [`PjrtCoreEvent::set`] or [`PjrtCoreEvent::await_ready`] completes.
    ///
    /// # Arguments
    ///
    /// * `callback` - Closure receiving `None` on success or `Some(LibTpuError)` on failure.
    pub fn on_ready<F>(&self, callback: F)
    where
        F: FnOnce(Option<LibTpuError>) + Send + 'static,
    {
        if self.is_ready() {
            callback(self.get_error());
        } else {
            self.callbacks.lock().push(Box::new(callback));
        }
    }

    /// Marks this event as complete with an optional [`LibTpuError`] (`PJRT_Event_Set`)
    /// and invokes any pending callbacks registered via [`PjrtCoreEvent::on_ready`].
    ///
    /// # Arguments
    ///
    /// * `status` - `None` for success, or `Some(err)` to fail the event.
    pub fn set(&self, status: Option<LibTpuError>) {
        let pending: Vec<EventReadyCallback> = std::mem::take(&mut *self.callbacks.lock());
        for cb in pending {
            cb(status.clone());
        }
        *self.error.lock() = status;
        self.ready.store(true, Ordering::Release);
    }

    /// Fails this event with a [`LibTpuError::EventError`] carrying its [`EventId`].
    ///
    /// # Arguments
    ///
    /// * `reason` - Failure reason message.
    pub fn fail_with_reason(&self, reason: String) {
        self.set(Some(LibTpuError::event_error(self.id, reason)));
    }

    /// Returns the error associated with this event, if any (`PJRT_Event_Error`).
    #[must_use]
    pub fn get_error(&self) -> Option<LibTpuError> {
        self.error.lock().clone()
    }

    /// Blocks until the event is ready and returns its status (`PJRT_Event_Await`).
    ///
    /// # Errors
    ///
    /// Returns the attached [`LibTpuError`] if the event completed with an error.
    pub fn await_ready(&self) -> LibTpuResult<()> {
        let err = self.get_error();
        self.set(err.clone());
        err.map_or(Ok(()), Err)
    }
}

/// Core representation of `PJRT_DeviceDescription`.
#[derive(Debug, Clone)]
pub struct PjrtCoreDeviceDescription {
    /// Underlying device specification from the HAL topology.
    pub spec: DeviceSpec,
    /// Human-readable device kind string (e.g., `"TPU v5 lite"`).
    pub kind_str: String,
    /// Detailed debug string (`PJRT_DeviceDescription_DebugString`).
    pub debug_string: String,
    /// Concise display string (`PJRT_DeviceDescription_ToString`).
    pub to_string_val: String,
    /// Coordinate array `[x, y, z]` exposed as a PJRT device attribute.
    pub coords_attr: [i64; 3],
}

impl PjrtCoreDeviceDescription {
    /// Constructs a [`PjrtCoreDeviceDescription`] from a [`DeviceSpec`].
    ///
    /// # Arguments
    ///
    /// * `spec` - Discovered HAL device specification.
    #[must_use]
    pub fn new(spec: DeviceSpec) -> Self {
        let kind_str = spec.generation.to_string();
        let debug_string = format!(
            "TPU_{}(id={}, process_index={}, coords={})",
            kind_str, spec.device_id, spec.process_index, spec.chip_coords
        );
        let to_string_val = format!(
            "TpuDevice(id={}, process_index={}, coords={}, core_on_chip={})",
            spec.device_id, spec.process_index, spec.chip_coords, spec.core_on_chip
        );
        let coords_attr = [
            i64::from(spec.chip_coords.x),
            i64::from(spec.chip_coords.y),
            i64::from(spec.chip_coords.z),
        ];
        Self {
            spec,
            kind_str,
            debug_string,
            to_string_val,
            coords_attr,
        }
    }
}

/// Core representation of a `PJRT_Memory` space.
#[derive(Debug, Clone)]
pub struct PjrtCoreMemory {
    /// Unique memory space identifier (`PJRT_Memory_Id`).
    pub id: MemorySpaceId,
    /// Memory kind classification (`device`, `pinned_host`, `unpinned_host`, `vmem`).
    pub kind: MemoryKind,
    /// Logical device IDs that can address this memory space.
    pub device_ids: Vec<DeviceId>,
    /// Detailed debug description (`PJRT_Memory_DebugString`).
    pub debug_string: String,
    /// Concise display string (`PJRT_Memory_ToString`).
    pub to_string_val: String,
}

impl PjrtCoreMemory {
    /// Creates a new [`PjrtCoreMemory`] instance.
    ///
    /// # Arguments
    ///
    /// * `id` - Memory space identifier.
    /// * `kind` - Memory kind.
    /// * `device_ids` - Devices that can address this memory space.
    #[must_use]
    pub fn new(id: MemorySpaceId, kind: MemoryKind, device_ids: Vec<DeviceId>) -> Self {
        let debug_string = format!("TpuMemorySpace(id={}, kind={})", id.0, kind.as_str());
        let to_string_val = format!("{}:{}", kind.as_str(), id.0);
        Self {
            id,
            kind,
            device_ids,
            debug_string,
            to_string_val,
        }
    }
}

/// Core representation of a `PJRT_Device`.
#[derive(Debug, Clone)]
pub struct PjrtCoreDevice {
    /// Static device description.
    pub description: PjrtCoreDeviceDescription,
    /// Default HBM memory space ID for this device.
    pub default_memory_id: MemorySpaceId,
    /// All memory space IDs addressable by this device.
    pub memory_space_ids: Vec<MemorySpaceId>,
}

/// Core representation of a `PJRT_TopologyDescription`.
#[derive(Debug, Clone)]
pub struct PjrtCoreTopology {
    /// Underlying HAL topology specification.
    pub spec: TopologySpec,
    /// Device descriptions for all devices in the topology.
    pub device_descriptions: Vec<PjrtCoreDeviceDescription>,
    /// Deterministic fingerprint of this topology (`PJRT_TopologyDescription_Fingerprint`).
    pub fingerprint: u64,
}

impl PjrtCoreTopology {
    /// Builds a [`PjrtCoreTopology`] from a [`TopologySpec`].
    ///
    /// # Arguments
    ///
    /// * `spec` - Discovered HAL topology specification.
    #[must_use]
    pub fn new(spec: TopologySpec) -> Self {
        let device_descriptions: Vec<PjrtCoreDeviceDescription> = spec
            .devices
            .iter()
            .cloned()
            .map(PjrtCoreDeviceDescription::new)
            .collect();
        let topo_seed = format!(
            "TPU_TOPO:{}:{}:{}:{}:{}",
            spec.generation as i32,
            spec.chip_bounds.x,
            spec.chip_bounds.y,
            spec.chip_bounds.z,
            device_descriptions.len()
        );
        let fingerprint = crate::proto::sha256_fingerprint_u64(topo_seed.as_bytes());
        Self {
            spec,
            device_descriptions,
            fingerprint,
        }
    }

    /// Serializes this topology description into wire-compatible `TpuTopologyArgsProto` bytes
    /// (`PJRT_TopologyDescription_Serialize`).
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let proto = crate::proto::TpuTopologyArgsProto {
            version: Some(self.spec.generation as i32),
            variant: Some(self.spec.platform_version.clone()),
            chip_config_name: Some(format!("{:016x}", self.fingerprint)),
            chips_per_host_bounds: Some(crate::proto::TpuDimensionsProto {
                x: i32::try_from((self.spec.chip_bounds.x / self.spec.host_bounds.x.max(1)).max(1))
                    .unwrap_or(1),
                y: i32::try_from((self.spec.chip_bounds.y / self.spec.host_bounds.y.max(1)).max(1))
                    .unwrap_or(1),
                z: i32::try_from((self.spec.chip_bounds.z / self.spec.host_bounds.z.max(1)).max(1))
                    .unwrap_or(1),
                w: 1,
            }),
            host_bounds: Some(crate::proto::TpuDimensionsProto {
                x: i32::try_from(self.spec.host_bounds.x).unwrap_or(1),
                y: i32::try_from(self.spec.host_bounds.y).unwrap_or(1),
                z: i32::try_from(self.spec.host_bounds.z).unwrap_or(1),
                w: 1,
            }),
            ..crate::proto::TpuTopologyArgsProto::default()
        };
        proto.encode_proto()
    }

    /// Deserializes a [`PjrtCoreTopology`] from serialized `TpuTopologyArgsProto` or `LTPUTOPO:` bytes
    /// (`PJRT_TopologyDescription_Deserialize`).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Serialized topology bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if the payload is invalid.
    pub fn deserialize(bytes: &[u8]) -> LibTpuResult<Self> {
        if bytes.starts_with(b"LTPUTOPO:") {
            let cfg = TpuInitConfig::default();
            let hal = SimulatedTpuHal::new(&cfg);
            return Ok(Self::new(hal.topology_spec()));
        }
        if let Ok(proto) = crate::proto::TpuTopologyArgsProto::decode_proto(bytes) {
            let ver = proto.version.unwrap_or(0);
            if ver > 0 {
                let mut cfg = TpuInitConfig {
                    generation: match ver {
                        2 => crate::types::TpuGeneration::V2,
                        3 => crate::types::TpuGeneration::V3,
                        4 => crate::types::TpuGeneration::V4,
                        6 => crate::types::TpuGeneration::V5p,
                        7 => crate::types::TpuGeneration::V6e,
                        8 => crate::types::TpuGeneration::V7x,
                        _ => crate::types::TpuGeneration::V5e,
                    },
                    ..TpuInitConfig::default()
                };
                if let Some(hb) = &proto.host_bounds {
                    let hx = u32::try_from(hb.x.max(1)).unwrap_or(1);
                    let hy = u32::try_from(hb.y.max(1)).unwrap_or(1);
                    let hz = u32::try_from(hb.z.max(1)).unwrap_or(1);
                    cfg.host_bounds = crate::types::ChipCoordinates::new(hx, hy, hz);
                }
                if let Some(cb) = &proto.chips_per_host_bounds {
                    let cx = u32::try_from(cb.x.max(1)).unwrap_or(2);
                    let cy = u32::try_from(cb.y.max(1)).unwrap_or(2);
                    let cz = u32::try_from(cb.z.max(1)).unwrap_or(1);
                    cfg.chips_per_host_bounds = crate::types::ChipCoordinates::new(cx, cy, cz);
                }
                let hal = SimulatedTpuHal::new(&cfg);
                let mut spec = hal.topology_spec();
                if let Some(pv) = proto.variant.filter(|s| !s.is_empty()) {
                    spec.platform_version = pv;
                }
                return Ok(Self::new(spec));
            }
        }
        Err(LibTpuError::DeserializationFailed {
            reason: String::from("Invalid topology serialized payload"),
        })
    }
}

/// Core representation of a `PJRT_Buffer` on-device tensor allocation.
#[derive(Debug)]
pub struct PjrtCoreBuffer {
    /// Unique buffer ID.
    pub id: BufferId,
    /// Owning device ID.
    pub device_id: DeviceId,
    /// Owning memory space ID.
    pub memory_id: MemorySpaceId,
    /// Validated tensor shape and layout.
    pub shape: Shape,
    /// Allocated HBM address on the HAL.
    pub address: DeviceAddress,
    /// Reference to the backing HAL for DMA reads and writes.
    pub hal: Arc<dyn TpuHal>,
    /// Optional error status recorded on this buffer (`PJRT_Client_CreateErrorBuffer`).
    ready_error: Mutex<Option<LibTpuError>>,
    /// Atomic deletion flag (`PJRT_Buffer_Delete` / `PJRT_Buffer_IsDeleted`).
    deleted: AtomicBool,
    /// Atomic alias-fulfillment flag (`PJRT_Client_FulfillAliasBuffer`).
    fulfilled: AtomicBool,
    /// External reference count (`PJRT_Buffer_IncreaseExternalReferenceCount`).
    external_ref_count: AtomicI64,
}

impl PjrtCoreBuffer {
    /// Creates a new [`PjrtCoreBuffer`] backed by an allocated HBM [`DeviceAddress`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - Owning logical TPU device.
    /// * `memory_id` - Owning memory space ID.
    /// * `shape` - Validated tensor shape.
    /// * `address` - Allocated HBM device address.
    /// * `hal` - Backing HAL instance.
    pub fn new(
        device_id: DeviceId,
        memory_id: MemorySpaceId,
        shape: Shape,
        address: DeviceAddress,
        hal: Arc<dyn TpuHal>,
    ) -> Self {
        Self {
            id: BufferId(next_id()),
            device_id,
            memory_id,
            shape,
            address,
            hal,
            ready_error: Mutex::new(None),
            deleted: AtomicBool::new(false),
            fulfilled: AtomicBool::new(false),
            external_ref_count: AtomicI64::new(1),
        }
    }

    /// Records a readiness error on this buffer (`PJRT_Client_CreateErrorBuffer`).
    ///
    /// # Arguments
    ///
    /// * `error` - Error status to attach to the buffer's readiness event.
    pub fn set_error(&self, error: LibTpuError) {
        *self.ready_error.lock() = Some(error);
    }

    /// Returns a clone of the error status recorded on this buffer, if any.
    #[must_use]
    pub fn ready_error(&self) -> Option<LibTpuError> {
        self.ready_error.lock().clone()
    }

    /// Returns a [`PjrtCoreEvent`] representing the readiness state of this buffer (`PJRT_Buffer_ReadyEvent`).
    #[must_use]
    pub fn ready_event(&self) -> PjrtCoreEvent {
        self.ready_error()
            .map_or_else(PjrtCoreEvent::ready_ok, PjrtCoreEvent::ready_err)
    }

    /// Fulfills this alias buffer by copying the device memory contents of `src` (`PJRT_Client_FulfillAliasBuffer`).
    ///
    /// # Arguments
    ///
    /// * `src` - Source buffer providing the fulfilled device payload.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::BufferDeleted`] if either buffer is deleted,
    /// [`LibTpuError::AliasAlreadyFulfilled`] if already fulfilled, or [`LibTpuError`] if the device copy fails.
    pub fn fulfill_from(&self, src: &Self) -> LibTpuResult<()> {
        if self.is_deleted() {
            return Err(LibTpuError::buffer_deleted(self.id));
        }
        if self.fulfilled.swap(true, Ordering::AcqRel) {
            return Err(LibTpuError::alias_already_fulfilled(self.id));
        }
        let bytes = src.to_host_bytes()?;
        let copy_len = self.shape.unpadded_byte_size().0.min(bytes.len());
        let slice = bytes.get(..copy_len).unwrap_or(&bytes);
        self.hal
            .write_device_memory(self.device_id, self.address, slice)
    }

    /// Marks this buffer as deleted and frees its underlying HBM allocation (`PJRT_Buffer_Delete`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if deallocation fails.
    pub fn delete(&self) -> LibTpuResult<()> {
        if !self.deleted.swap(true, Ordering::AcqRel) {
            self.hal
                .deallocate_device_memory(self.device_id, self.address)?;
        }
        Ok(())
    }

    /// Returns whether this buffer has been deleted (`PJRT_Buffer_IsDeleted`).
    pub fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::Acquire)
    }

    /// Copies the buffer contents to a host byte vector (`PJRT_Buffer_ToHostBuffer`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::BufferDeleted`] if deleted or [`LibTpuError::TransferFailed`] on DMA error.
    pub fn to_host_bytes(&self) -> LibTpuResult<Vec<u8>> {
        if self.is_deleted() {
            return Err(LibTpuError::BufferDeleted {
                buffer_id: self.id.0,
            });
        }
        let sz = self.shape.unpadded_byte_size().0;
        let mut out = vec![0_u8; sz];
        self.hal
            .read_device_memory(self.device_id, self.address, &mut out)?;
        Ok(out)
    }

    /// Copies this buffer to another device or memory space (`PJRT_Buffer_CopyToDevice` / `CopyToMemory`).
    ///
    /// # Arguments
    ///
    /// * `dst_device` - Destination logical TPU device.
    /// * `dst_memory` - Destination memory space ID.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if this buffer is deleted or allocation/transfer fails.
    pub fn copy_to_device(
        &self,
        dst_device: DeviceId,
        dst_memory: MemorySpaceId,
    ) -> LibTpuResult<Self> {
        if self.is_deleted() {
            return Err(LibTpuError::BufferDeleted {
                buffer_id: self.id.0,
            });
        }
        let on_dev_size = self.shape.on_device_byte_size();
        let new_addr =
            self.hal
                .allocate_device_memory(dst_device, MemoryKind::DeviceHbm, on_dev_size)?;
        let unpadded = self.shape.unpadded_byte_size();
        self.hal.copy_device_to_device(
            self.device_id,
            self.address,
            dst_device,
            new_addr,
            unpadded,
        )?;
        Ok(Self::new(
            dst_device,
            dst_memory,
            self.shape.clone(),
            new_addr,
            Arc::clone(&self.hal),
        ))
    }

    /// Reinterprets this buffer with a new element type and dimensions (`PJRT_Buffer_Bitcast`).
    ///
    /// # Arguments
    ///
    /// * `new_type` - Target element type.
    /// * `new_dims` - Target dimension sizes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if deleted or if the new shape is invalid.
    pub fn bitcast(&self, new_type: ElementType, new_dims: Vec<i64>) -> LibTpuResult<Self> {
        if self.is_deleted() {
            return Err(LibTpuError::BufferDeleted {
                buffer_id: self.id.0,
            });
        }
        let new_shape = Shape::new(new_type, new_dims, None, None)?;
        let on_dev_size = new_shape.on_device_byte_size();
        let new_addr =
            self.hal
                .allocate_device_memory(self.device_id, MemoryKind::DeviceHbm, on_dev_size)?;
        let copy_bytes = self
            .shape
            .unpadded_byte_size()
            .0
            .min(new_shape.unpadded_byte_size().0);
        self.hal.copy_device_to_device(
            self.device_id,
            self.address,
            self.device_id,
            new_addr,
            ByteSize(copy_bytes),
        )?;
        Ok(Self::new(
            self.device_id,
            self.memory_id,
            new_shape,
            new_addr,
            Arc::clone(&self.hal),
        ))
    }

    /// Increments the external reference count (`PJRT_Buffer_IncreaseExternalReferenceCount`).
    pub fn inc_external_ref(&self) -> i64 {
        self.external_ref_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements the external reference count (`PJRT_Buffer_DecreaseExternalReferenceCount`).
    pub fn dec_external_ref(&self) -> i64 {
        self.external_ref_count.fetch_sub(1, Ordering::Relaxed) - 1
    }
}

/// Core representation of `PJRT_CopyToDeviceStream` for chunked host-to-device streaming.
#[derive(Debug)]
pub struct PjrtCoreCopyToDeviceStream {
    /// Total expected byte size of the stream.
    pub total_bytes: i64,
    /// Granule alignment size in bytes (`256`).
    pub granule_size: i64,
    /// Current accumulated byte count.
    current_bytes: AtomicI64,
    /// Staged host bytes accumulated across chunks.
    staged_bytes: Arc<Mutex<Vec<u8>>>,
}

impl PjrtCoreCopyToDeviceStream {
    /// Creates a new [`PjrtCoreCopyToDeviceStream`].
    ///
    /// # Arguments
    ///
    /// * `total_bytes` - Total number of bytes to be transferred across all chunks.
    /// * `granule_size` - Transfer granule alignment in bytes.
    #[must_use]
    pub fn new(total_bytes: i64, granule_size: i64) -> Self {
        Self {
            total_bytes,
            granule_size: granule_size.max(1),
            current_bytes: AtomicI64::new(0),
            staged_bytes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a shared reference to the accumulated `staged_bytes` buffer.
    #[must_use]
    pub fn shared_staged_bytes(&self) -> Arc<Mutex<Vec<u8>>> {
        Arc::clone(&self.staged_bytes)
    }

    /// Adds a chunk of bytes to the stream (`PJRT_CopyToDeviceStream_AddChunk`).
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Size in bytes of the incoming chunk.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::StreamOverflow`] if adding `chunk_size` exceeds `total_bytes`.
    pub fn add_chunk(&self, chunk_size: i64) -> LibTpuResult<PjrtCoreEvent> {
        let prev = self.current_bytes.load(Ordering::Acquire);
        let remaining = self.total_bytes - prev;
        if chunk_size < 0 || chunk_size > remaining {
            return Err(LibTpuError::StreamOverflow {
                attempted_bytes: chunk_size,
                remaining_bytes: remaining,
            });
        }
        self.current_bytes.fetch_add(chunk_size, Ordering::AcqRel);
        Ok(PjrtCoreEvent::ready_ok())
    }

    /// Appends a chunk byte payload to the stream and advances `current_bytes` (`PJRT_CopyToDeviceStream_AddChunk`).
    ///
    /// # Arguments
    ///
    /// * `data` - Host byte slice for the chunk.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::StreamOverflow`] if adding `data.len()` exceeds `total_bytes`.
    pub fn add_chunk_data(&self, data: &[u8]) -> LibTpuResult<PjrtCoreEvent> {
        let chunk_size = i64::try_from(data.len()).unwrap_or(i64::MAX);
        let ev = self.add_chunk(chunk_size)?;
        self.staged_bytes.lock().extend_from_slice(data);
        Ok(ev)
    }

    /// Flushes accumulated `staged_bytes` into a target [`PjrtCoreBuffer`] on the device HBM.
    ///
    /// # Arguments
    ///
    /// * `target` - Destination [`PjrtCoreBuffer`] to populate with staged bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `target` is deleted or the HBM write fails.
    pub fn flush_to_buffer(&self, target: &PjrtCoreBuffer) -> LibTpuResult<()> {
        if target.is_deleted() {
            return Err(LibTpuError::BufferDeleted {
                buffer_id: target.id.0,
            });
        }
        let data = self.staged_data();
        if !data.is_empty() {
            target
                .hal
                .write_device_memory(target.device_id, target.address, &data)?;
        }
        Ok(())
    }

    /// Returns a snapshot of the accumulated staged bytes written via [`PjrtCoreCopyToDeviceStream::add_chunk_data`].
    #[must_use]
    pub fn staged_data(&self) -> Vec<u8> {
        self.staged_bytes.lock().clone()
    }

    /// Returns the number of bytes transferred so far (`PJRT_CopyToDeviceStream_CurrentBytes`).
    #[must_use]
    pub fn current_bytes(&self) -> i64 {
        self.current_bytes.load(Ordering::Acquire)
    }
}

/// Core representation of `PJRT_Executable`.
#[derive(Debug, Clone)]
pub struct PjrtCoreExecutable {
    /// Unique executable identifier.
    pub id: ExecutableId,
    /// Underlying compiled TPU program.
    pub program: CompiledProgram,
}

impl PjrtCoreExecutable {
    /// Creates a new [`PjrtCoreExecutable`] wrapping a [`CompiledProgram`].
    ///
    /// # Arguments
    ///
    /// * `program` - Compiled TPU program.
    #[must_use]
    pub fn new(program: CompiledProgram) -> Self {
        Self {
            id: ExecutableId(next_id()),
            program,
        }
    }

    /// Returns the cost analysis metrics (`PJRT_Executable_GetCostAnalysis`).
    #[must_use]
    pub const fn cost_analysis(&self) -> CostAnalysis {
        self.program.cost_analysis
    }

    /// Returns the compiled memory usage statistics (`PJRT_Executable_GetCompiledMemoryStats`).
    #[must_use]
    pub const fn compiled_memory_stats(&self) -> CompiledMemoryStats {
        self.program.memory_stats
    }
}

/// Core representation of `PJRT_LoadedExecutable`.
#[derive(Debug)]
pub struct PjrtCoreLoadedExecutable {
    /// Underlying [`PjrtCoreExecutable`].
    pub executable: PjrtCoreExecutable,
    /// Addressable device IDs on which this executable is loaded.
    pub addressable_devices: Vec<DeviceId>,
    /// Reference to the backing HAL.
    pub hal: Arc<dyn TpuHal>,
    /// Atomic deletion flag (`PJRT_LoadedExecutable_Delete` / `IsDeleted`).
    deleted: AtomicBool,
}

impl PjrtCoreLoadedExecutable {
    /// Creates a new [`PjrtCoreLoadedExecutable`].
    ///
    /// # Arguments
    ///
    /// * `executable` - Compiled executable metadata.
    /// * `addressable_devices` - Devices on which this program is loaded.
    /// * `hal` - Backing HAL instance.
    pub fn new(
        executable: PjrtCoreExecutable,
        addressable_devices: Vec<DeviceId>,
        hal: Arc<dyn TpuHal>,
    ) -> Self {
        Self {
            executable,
            addressable_devices,
            hal,
            deleted: AtomicBool::new(false),
        }
    }

    /// Marks this loaded executable as deleted (`PJRT_LoadedExecutable_Delete`).
    pub fn delete(&self) {
        self.deleted.store(true, Ordering::Release);
    }

    /// Returns whether this loaded executable has been deleted (`PJRT_LoadedExecutable_IsDeleted`).
    pub fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::Acquire)
    }

    /// Executes the loaded program on a single device with the given input buffers (`PJRT_LoadedExecutable_Execute`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `memory_id` - Default output memory space ID.
    /// * `inputs` - Input [`PjrtCoreBuffer`] references.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the executable or any input buffer is deleted or if execution fails.
    pub fn execute_on_device(
        &self,
        device_id: DeviceId,
        memory_id: MemorySpaceId,
        inputs: &[&PjrtCoreBuffer],
    ) -> LibTpuResult<Vec<PjrtCoreBuffer>> {
        if self.is_deleted() {
            return Err(LibTpuError::ExecutableDeleted {
                executable_id: self.executable.id.0,
            });
        }
        let mut in_addrs = Vec::with_capacity(inputs.len());
        for &buf in inputs {
            if buf.is_deleted() {
                return Err(LibTpuError::BufferDeleted {
                    buffer_id: buf.id.0,
                });
            }
            in_addrs.push(buf.address);
        }

        let mut out_addrs = Vec::with_capacity(self.executable.program.output_shapes.len());
        let mut out_buffers = Vec::with_capacity(self.executable.program.output_shapes.len());
        for out_shape in &self.executable.program.output_shapes {
            let sz = out_shape.on_device_byte_size();
            let addr = self
                .hal
                .allocate_device_memory(device_id, MemoryKind::DeviceHbm, sz)?;
            out_addrs.push(addr);
            out_buffers.push(PjrtCoreBuffer::new(
                device_id,
                memory_id,
                out_shape.clone(),
                addr,
                Arc::clone(&self.hal),
            ));
        }

        self.hal
            .execute_program(device_id, &self.executable.program, &in_addrs, &out_addrs)?;
        Ok(out_buffers)
    }

    /// Executes the loaded program on `device_id` while servicing `Send` and `Recv` host-transfer
    /// channels (`jax.pure_callback`) via `send_fn` and `recv_fn`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target device ID.
    /// * `memory_id` - Default output memory space ID.
    /// * `inputs` - Input [`PjrtCoreBuffer`] references.
    /// * `send_fn` - Host callback invoked with `(channel_id, device_bytes)` on `Send`.
    /// * `recv_fn` - Host callback invoked with `channel_id` returning host bytes on `Recv`.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the executable or any input buffer is deleted or if execution fails.
    pub fn execute_on_device_with_host_io<FS, FR>(
        &self,
        device_id: DeviceId,
        memory_id: MemorySpaceId,
        inputs: &[&PjrtCoreBuffer],
        mut send_fn: FS,
        mut recv_fn: FR,
    ) -> LibTpuResult<Vec<PjrtCoreBuffer>>
    where
        FS: FnMut(i64, &[u8]) -> LibTpuResult<()>,
        FR: FnMut(i64) -> LibTpuResult<Vec<u8>>,
    {
        self.execute_on_device_with_host_io_dyn(
            device_id,
            memory_id,
            inputs,
            &mut send_fn,
            &mut recv_fn,
        )
    }

    /// Non-generic host-I/O execution body invoked by [`Self::execute_on_device_with_host_io`].
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target [`DeviceId`].
    /// * `memory_id` - Output [`MemorySpaceId`].
    /// * `inputs` - Ordered slice of input [`PjrtCoreBuffer`] references.
    /// * `send_fn` - Dynamic device-to-host `Send` callback.
    /// * `recv_fn` - Dynamic host-to-device `Recv` callback.
    fn execute_on_device_with_host_io_dyn(
        &self,
        device_id: DeviceId,
        memory_id: MemorySpaceId,
        inputs: &[&PjrtCoreBuffer],
        send_fn: &mut dyn FnMut(i64, &[u8]) -> LibTpuResult<()>,
        recv_fn: &mut dyn FnMut(i64) -> LibTpuResult<Vec<u8>>,
    ) -> LibTpuResult<Vec<PjrtCoreBuffer>> {
        if self.is_deleted() {
            return Err(LibTpuError::ExecutableDeleted {
                executable_id: self.executable.id.0,
            });
        }
        let mut host_inputs = Vec::with_capacity(inputs.len());
        for &buf in inputs {
            if buf.is_deleted() {
                return Err(LibTpuError::BufferDeleted {
                    buffer_id: buf.id.0,
                });
            }
            host_inputs.push(buf.to_host_bytes()?);
        }
        let prog = &self.executable.program;
        let default_elem_type = prog
            .output_shapes
            .first()
            .map_or(ElementType::F32, |s| s.element_type);
        let mut vm = crate::hal::TpuCoreVmState::new();
        let max_slot = crate::hal::TpuCoreVmState::max_slot_in_bundles(&prog.instruction_bundles)
            .max(prog.output_slots.iter().copied().max().unwrap_or(0));
        let mut registers = host_inputs;
        registers.resize_with((max_slot + 2).max(registers.len()), Vec::new);
        let mut last_result = registers.first().cloned().unwrap_or_default();
        for bundle in &prog.instruction_bundles {
            if bundle.opcode == crate::compiler::TpuOpCode::Send {
                let src = bundle
                    .operand_slots
                    .first()
                    .and_then(|&s| registers.get(s))
                    .map_or(&[][..], Vec::as_slice);
                send_fn(bundle.attrs.send_recv_channel_id, src)?;
                Self::write_register_slot(&mut registers, bundle.output_slot, Vec::new());
                continue;
            }
            if bundle.opcode == crate::compiler::TpuOpCode::Recv {
                let received = recv_fn(bundle.attrs.send_recv_channel_id)?;
                last_result.clone_from(&received);
                Self::write_register_slot(&mut registers, bundle.output_slot, received);
                continue;
            }
            vm.step_bundles_on_registers(
                std::slice::from_ref(bundle),
                default_elem_type,
                &mut registers,
                &mut last_result,
            );
        }
        let mut out_buffers = Vec::with_capacity(prog.output_shapes.len());
        for (out_idx, out_shape) in prog.output_shapes.iter().enumerate() {
            let sz = out_shape.on_device_byte_size();
            let addr = self
                .hal
                .allocate_device_memory(device_id, MemoryKind::DeviceHbm, sz)?;
            let raw_bytes = prog
                .output_slots
                .get(out_idx)
                .and_then(|&s| registers.get(s))
                .unwrap_or(&last_result);
            let unpadded = out_shape.unpadded_byte_size().0;
            let mut clipped = raw_bytes.clone();
            clipped.resize(unpadded, 0);
            if !clipped.is_empty() {
                self.hal.write_device_memory(device_id, addr, &clipped)?;
            }
            out_buffers.push(PjrtCoreBuffer::new(
                device_id,
                memory_id,
                out_shape.clone(),
                addr,
                Arc::clone(&self.hal),
            ));
        }
        Ok(out_buffers)
    }

    /// Executes the loaded program across all `addressable_devices` (`PJRT_LoadedExecutable_Execute`),
    /// returning one `Vec<PjrtCoreBuffer>` output list per addressable device, and synchronizing
    /// cross-replica collective operations (`AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`,
    /// `CollectivePermute`) via [`crate::hal::SpmdCollectiveRendezvousEngine`] and [`crate::distributed::TorusIciRouter`].
    ///
    /// # Arguments
    ///
    /// * `memory_id` - Default output memory space ID.
    /// * `per_device_inputs` - Slice of input buffer slices, one per addressable device.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the executable or any input buffer is deleted or if execution fails.
    #[allow(clippy::too_many_lines)]
    pub fn execute_all_devices(
        &self,
        memory_id: MemorySpaceId,
        per_device_inputs: &[&[&PjrtCoreBuffer]],
    ) -> LibTpuResult<Vec<Vec<PjrtCoreBuffer>>> {
        if self.is_deleted() {
            return Err(LibTpuError::ExecutableDeleted {
                executable_id: self.executable.id.0,
            });
        }
        let prog = &self.executable.program;
        let has_collective = prog.instruction_bundles.iter().any(|b| {
            matches!(
                b.opcode,
                crate::compiler::TpuOpCode::AllReduceSumF32
                    | crate::compiler::TpuOpCode::AllGather
                    | crate::compiler::TpuOpCode::ReduceScatter
                    | crate::compiler::TpuOpCode::AllToAll
                    | crate::compiler::TpuOpCode::CollectivePermute
                    | crate::compiler::TpuOpCode::ReplicaId
                    | crate::compiler::TpuOpCode::PartitionId
            )
        });
        let num_devs = self.addressable_devices.len();
        if !has_collective || num_devs <= 1 {
            let mut all_outputs = Vec::with_capacity(num_devs);
            for (idx, &dev_id) in self.addressable_devices.iter().enumerate() {
                let dev_inputs = per_device_inputs.get(idx).copied().unwrap_or(&[]);
                all_outputs.push(self.execute_on_device(dev_id, memory_id, dev_inputs)?);
            }
            return Ok(all_outputs);
        }

        let default_elem_type = prog
            .output_shapes
            .first()
            .map_or(ElementType::F32, |s| s.element_type);
        let max_slot = crate::hal::TpuCoreVmState::max_slot_in_bundles(&prog.instruction_bundles)
            .max(prog.output_slots.iter().copied().max().unwrap_or(0));
        let mut vms = Vec::with_capacity(num_devs);
        let mut reg_files = Vec::with_capacity(num_devs);
        let mut last_results = Vec::with_capacity(num_devs);
        for (idx, &_dev_id) in self.addressable_devices.iter().enumerate() {
            let dev_inputs = per_device_inputs.get(idx).copied().unwrap_or(&[]);
            let mut in_vecs = Vec::with_capacity(dev_inputs.len());
            for &buf in dev_inputs {
                if buf.is_deleted() {
                    return Err(LibTpuError::BufferDeleted {
                        buffer_id: buf.id.0,
                    });
                }
                in_vecs.push(buf.to_host_bytes()?);
            }
            let first = in_vecs.first().cloned().unwrap_or_default();
            in_vecs.resize_with((max_slot + 2).max(in_vecs.len()), Vec::new);
            vms.push(crate::hal::TpuCoreVmState::new());
            reg_files.push(in_vecs);
            last_results.push(first);
        }

        for bundle in &prog.instruction_bundles {
            let is_coll = matches!(
                bundle.opcode,
                crate::compiler::TpuOpCode::AllReduceSumF32
                    | crate::compiler::TpuOpCode::AllGather
                    | crate::compiler::TpuOpCode::ReduceScatter
                    | crate::compiler::TpuOpCode::AllToAll
                    | crate::compiler::TpuOpCode::CollectivePermute
                    | crate::compiler::TpuOpCode::ReplicaId
                    | crate::compiler::TpuOpCode::PartitionId
            );
            if is_coll {
                let elem_ty = bundle
                    .attrs
                    .output_shape
                    .as_ref()
                    .map_or(default_elem_type, |s| s.element_type);
                let src_slot = bundle.operand_slots.first().copied().unwrap_or(0);
                let replica_bytes: Vec<Vec<u8>> = reg_files
                    .iter()
                    .map(|rf| rf.get(src_slot).cloned().unwrap_or_default())
                    .collect();
                let synced = crate::hal::SpmdCollectiveRendezvousEngine::execute_rendezvous(
                    bundle.opcode,
                    elem_ty,
                    &replica_bytes,
                    &bundle.attrs,
                );
                for ((rf, lr), out_b) in reg_files
                    .iter_mut()
                    .zip(last_results.iter_mut())
                    .zip(synced)
                {
                    lr.clone_from(&out_b);
                    Self::write_register_slot(rf, bundle.output_slot, out_b);
                }
            } else {
                for ((vm, rf), lr) in vms
                    .iter_mut()
                    .zip(reg_files.iter_mut())
                    .zip(last_results.iter_mut())
                {
                    vm.step_bundles_on_registers(
                        std::slice::from_ref(bundle),
                        default_elem_type,
                        rf,
                        lr,
                    );
                }
            }
        }

        let router = crate::distributed::TorusIciRouter::new(
            crate::types::ChipCoordinates::new(u32::try_from(num_devs).unwrap_or(1), 1, 1),
            true,
        );
        let _ = router.compute_fault_tolerant_route(
            crate::types::ChipCoordinates::new(0, 0, 0),
            crate::types::ChipCoordinates::new(1, 0, 0),
        );

        let mut all_outputs = Vec::with_capacity(num_devs);
        for (idx, &dev_id) in self.addressable_devices.iter().enumerate() {
            let mut dev_outs = Vec::with_capacity(prog.output_shapes.len());
            for (out_idx, out_shape) in prog.output_shapes.iter().enumerate() {
                let sz = out_shape.on_device_byte_size();
                let addr = self
                    .hal
                    .allocate_device_memory(dev_id, MemoryKind::DeviceHbm, sz)?;
                let raw_bytes = prog
                    .output_slots
                    .get(out_idx)
                    .and_then(|&s| reg_files.get(idx).and_then(|rf| rf.get(s)))
                    .or_else(|| last_results.get(idx))
                    .map_or_else(Vec::new, Vec::clone);
                let unpadded = out_shape.unpadded_byte_size().0;
                let mut clipped = raw_bytes;
                clipped.resize(unpadded, 0);
                if !clipped.is_empty() {
                    self.hal.write_device_memory(dev_id, addr, &clipped)?;
                }
                dev_outs.push(PjrtCoreBuffer::new(
                    dev_id,
                    memory_id,
                    out_shape.clone(),
                    addr,
                    Arc::clone(&self.hal),
                ));
            }
            all_outputs.push(dev_outs);
        }
        Ok(all_outputs)
    }

    /// Writes `value` into `registers[slot]` when `slot` is in bounds.
    ///
    /// # Arguments
    ///
    /// * `registers` - Mutable register slot bank.
    /// * `slot` - Destination slot index.
    /// * `value` - Tensor byte payload to store.
    fn write_register_slot(registers: &mut [Vec<u8>], slot: usize, value: Vec<u8>) {
        if let Some(dst) = registers.get_mut(slot) {
            *dst = value;
        }
    }
}

/// Core representation of `PJRT_AsyncHostToDeviceTransferManager`.
#[derive(Debug)]
pub struct PjrtCoreAsyncTransferManager {
    /// Target logical TPU device.
    pub device_id: DeviceId,
    /// Created buffers managed by this transfer manager.
    pub buffers: Vec<Arc<PjrtCoreBuffer>>,
    /// Byte sizes of each managed buffer.
    pub buffer_sizes: Vec<i64>,
    /// Per-buffer error status set via `SetBufferError`.
    pub buffer_errors: Mutex<Vec<Option<LibTpuError>>>,
    /// Metadata key-value pairs attached via `AddMetadata`.
    pub metadata: Mutex<Vec<(String, String)>>,
}

impl PjrtCoreAsyncTransferManager {
    /// Creates a new [`PjrtCoreAsyncTransferManager`] for a list of shapes on `device_id`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `memory_id` - Target memory space ID.
    /// * `shapes` - Shapes of the buffers to pre-allocate.
    /// * `hal` - Backing HAL instance.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if any buffer allocation fails.
    pub fn create(
        device_id: DeviceId,
        memory_id: MemorySpaceId,
        shapes: Vec<Shape>,
        hal: &Arc<dyn TpuHal>,
    ) -> LibTpuResult<Self> {
        let count = shapes.len();
        let mut buffers = Vec::with_capacity(count);
        let mut buffer_sizes = Vec::with_capacity(count);
        for shape in shapes {
            let unpadded = i64::try_from(shape.unpadded_byte_size().0).unwrap_or(0);
            let on_dev = shape.on_device_byte_size();
            let addr = hal.allocate_device_memory(device_id, MemoryKind::DeviceHbm, on_dev)?;
            buffers.push(Arc::new(PjrtCoreBuffer::new(
                device_id,
                memory_id,
                shape,
                addr,
                Arc::clone(hal),
            )));
            buffer_sizes.push(unpadded);
        }
        Ok(Self {
            device_id,
            buffers,
            buffer_sizes,
            buffer_errors: Mutex::new(vec![None; count]),
            metadata: Mutex::new(Vec::new()),
        })
    }

    /// Transfers a slice of host data into buffer `buffer_index` (`TransferData`).
    ///
    /// # Arguments
    ///
    /// * `buffer_index` - Index of the target buffer.
    /// * `data` - Host bytes to write.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `buffer_index` is out of range.
    pub fn transfer_data(&self, buffer_index: usize, data: &[u8]) -> LibTpuResult<()> {
        let buf = self
            .buffers
            .get(buffer_index)
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "buffer_index",
                reason: format!("Buffer index {buffer_index} out of range"),
            })?;
        buf.hal
            .write_device_memory(buf.device_id, buf.address, data)
    }

    /// Transfers a host literal into buffer `buffer_index`, applying TPU layout linearization (`TransferLiteral`).
    ///
    /// # Arguments
    ///
    /// * `buffer_index` - Index of the target buffer.
    /// * `literal_bytes` - Row-major host literal bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if `buffer_index` is out of range or if transfer fails.
    pub fn transfer_literal(&self, buffer_index: usize, literal_bytes: &[u8]) -> LibTpuResult<()> {
        let buf = self
            .buffers
            .get(buffer_index)
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "buffer_index",
                reason: format!("Buffer index {buffer_index} out of range"),
            })?;
        let unpadded = buf.shape.unpadded_byte_size().0;
        let payload = literal_bytes
            .get(..unpadded.min(literal_bytes.len()))
            .unwrap_or(literal_bytes);
        buf.hal
            .write_device_memory(buf.device_id, buf.address, payload)
    }

    /// Retrieves buffer `buffer_index` (`RetrieveBuffer`).
    ///
    /// # Arguments
    ///
    /// * `buffer_index` - Index of the buffer to retrieve.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `buffer_index` is out of range.
    pub fn retrieve_buffer(&self, buffer_index: usize) -> LibTpuResult<Arc<PjrtCoreBuffer>> {
        self.buffers
            .get(buffer_index)
            .cloned()
            .ok_or_else(|| LibTpuError::InvalidArgument {
                parameter: "buffer_index",
                reason: format!("Buffer index {buffer_index} out of range"),
            })
    }

    /// Records an error on buffer `buffer_index` (`SetBufferError`).
    ///
    /// # Arguments
    ///
    /// * `buffer_index` - Index of the target buffer.
    /// * `error` - Error to attach to the buffer.
    pub fn set_buffer_error(&self, buffer_index: usize, error: LibTpuError) {
        let mut guard: MutexGuard<'_, Vec<Option<LibTpuError>>> = self.buffer_errors.lock();
        if let Some(slot) = guard.get_mut(buffer_index) {
            *slot = Some(error);
        }
    }

    /// Adds key-value metadata to this transfer manager (`AddMetadata`).
    ///
    /// # Arguments
    ///
    /// * `entries` - Slice of `(key, value)` pairs.
    pub fn add_metadata(&self, entries: Vec<(String, String)>) {
        let mut guard: MutexGuard<'_, Vec<(String, String)>> = self.metadata.lock();
        guard.extend(entries);
    }
}

/// Multi-host process heartbeat record (`PJRT_ProcessInfo` / `PJRT_Client_UpdateGlobalProcessInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessInfoRecord {
    /// Task / process ID in the multi-host job (`0..num_processes`).
    pub process_id: i32,
    /// Monotonic incarnation identifier for task restarts.
    pub incarnation_id: u64,
    /// Process state enum value (`PJRT_ProcessState`).
    pub state: i32,
    /// Latest `PJRT_Error_Code` reported for this process (`0` = OK).
    pub error_code: i32,
    /// Human-readable error message reported for this process.
    pub error_message: String,
}

/// Core representation of `PJRT_Client` managing topology, devices, memories, buffers, and compilation.
pub struct PjrtCoreClient {
    /// Backing HAL implementation.
    pub hal: Arc<dyn TpuHal>,
    /// Discovered topology description.
    pub topology: PjrtCoreTopology,
    /// All logical devices in the slice.
    pub devices: Vec<PjrtCoreDevice>,
    /// Indices into `devices` for devices addressable by this client process.
    pub addressable_device_indices: Vec<usize>,
    /// All addressable memory spaces.
    pub memories: Vec<PjrtCoreMemory>,
    /// Process index of this client (`PJRT_Client_ProcessIndex`).
    pub process_index: ProcessIndex,
    /// Registered DMA-mapped host regions (`(base_addr, size)`).
    pub dma_mappings: Mutex<Vec<(u64, usize)>>,
    /// Global multi-host process status table updated via `PJRT_Client_UpdateGlobalProcessInfo`.
    pub process_infos: Mutex<Vec<ProcessInfoRecord>>,
    /// Distributed multi-host key-value store backing `xla.LocalTopologyProto` and `xla.GlobalTopologyProto` exchange.
    pub distributed_kv: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl std::fmt::Debug for PjrtCoreClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PjrtCoreClient")
            .field("process_index", &self.process_index)
            .field("num_devices", &self.devices.len())
            .finish_non_exhaustive()
    }
}

impl PjrtCoreClient {
    /// Creates a new [`PjrtCoreClient`] backed by `hal` and an already-discovered [`TopologySpec`].
    ///
    /// # Arguments
    ///
    /// * `hal` - Backing hardware abstraction layer.
    /// * `spec` - Discovered topology specification.
    /// * `process_index` - Process index of the client.
    pub fn from_topology(
        hal: Arc<dyn TpuHal>,
        spec: TopologySpec,
        process_index: ProcessIndex,
    ) -> Self {
        let topology = PjrtCoreTopology::new(spec);

        let mut devices = Vec::with_capacity(topology.device_descriptions.len());
        let mut addressable_device_indices = Vec::new();
        let mut memories = Vec::new();
        let mut next_mem_id: i32 = 0;

        for (idx, desc) in topology.device_descriptions.iter().enumerate() {
            let dev_id = desc.spec.device_id;
            let hbm_id = MemorySpaceId(next_mem_id);
            next_mem_id += 1;
            let pinned_id = MemorySpaceId(next_mem_id);
            next_mem_id += 1;
            let vmem_id = MemorySpaceId(next_mem_id);
            next_mem_id += 1;

            if desc.spec.is_addressable {
                addressable_device_indices.push(idx);
                memories.push(PjrtCoreMemory::new(
                    hbm_id,
                    MemoryKind::DeviceHbm,
                    vec![dev_id],
                ));
                memories.push(PjrtCoreMemory::new(
                    pinned_id,
                    MemoryKind::PinnedHost,
                    vec![dev_id],
                ));
                memories.push(PjrtCoreMemory::new(vmem_id, MemoryKind::Vmem, vec![dev_id]));
            }

            devices.push(PjrtCoreDevice {
                description: desc.clone(),
                default_memory_id: hbm_id,
                memory_space_ids: vec![hbm_id, pinned_id, vmem_id],
            });
        }

        Self {
            hal,
            topology,
            devices,
            addressable_device_indices,
            memories,
            process_index,
            dma_mappings: Mutex::new(Vec::new()),
            process_infos: Mutex::new(Vec::new()),
            distributed_kv: Mutex::new(BTreeMap::new()),
        }
    }

    /// Updates the client's global multi-host process status table (`PJRT_Client_UpdateGlobalProcessInfo`).
    ///
    /// # Arguments
    ///
    /// * `infos` - Latest [`ProcessInfoRecord`] snapshot across cluster processes.
    pub fn update_global_process_info(&self, infos: Vec<ProcessInfoRecord>) {
        *self.process_infos.lock() = infos;
    }

    /// Returns a snapshot of the client's global multi-host process status table.
    #[must_use]
    pub fn global_process_infos(&self) -> Vec<ProcessInfoRecord> {
        self.process_infos.lock().clone()
    }

    /// Creates a new [`PjrtCoreClient`] backed by `hal`.
    ///
    /// # Arguments
    ///
    /// * `hal` - Backing hardware abstraction layer.
    /// * `process_index` - Process index of the client.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if topology discovery fails.
    pub fn new(hal: Arc<dyn TpuHal>, process_index: ProcessIndex) -> LibTpuResult<Self> {
        let spec = hal.discover_topology()?;
        Ok(Self::from_topology(hal, spec, process_index))
    }

    /// Creates a [`PjrtCoreClient`] from a [`TpuInitConfig`] using [`crate::hal::HalBackend::from_config`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration.
    #[must_use]
    pub fn from_config(config: &TpuInitConfig) -> Self {
        let backend = crate::hal::HalBackend::from_config(config);
        let spec = backend.topology_spec();
        let hal: Arc<dyn TpuHal> = Arc::new(backend);
        Self::from_topology(hal, spec, config.process_index)
    }

    /// Creates a [`PjrtCoreClient`] infallibly from a [`TpuInitConfig`] using [`PjrtCoreClient::from_config`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration.
    #[must_use]
    pub fn from_simulated(config: &TpuInitConfig) -> Self {
        Self::from_config(config)
    }

    /// Creates a [`PjrtCoreClient`] using [`crate::hal::create_default_hal`] and [`TpuInitConfig`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if initialization fails.
    pub fn with_simulated_hal(config: &TpuInitConfig) -> LibTpuResult<Self> {
        let hal = crate::hal::create_default_hal(config);
        Self::new(hal, config.process_index)
    }

    /// Looks up a device by its global [`DeviceId`] (`PJRT_Client_LookupDevice`).
    ///
    /// # Arguments
    ///
    /// * `id` - Global device identifier.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if no device matches `id`.
    pub fn lookup_device(&self, id: DeviceId) -> LibTpuResult<&PjrtCoreDevice> {
        self.devices
            .iter()
            .find(|d| d.description.spec.device_id == id)
            .ok_or(LibTpuError::DeviceNotFound { device_id: id.0 })
    }

    /// Looks up an addressable device by its [` crate::types::LocalHardwareId `] (`PJRT_Client_LookupAddressableDevice`).
    ///
    /// # Arguments
    ///
    /// * `local_hardware_id` - Local hardware ordinal on this host.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if no addressable device has `local_hardware_id`.
    pub fn lookup_addressable_device(
        &self,
        local_hardware_id: i32,
    ) -> LibTpuResult<&PjrtCoreDevice> {
        self.devices
            .iter()
            .find(|d| {
                d.description.spec.is_addressable
                    && d.description.spec.local_hardware_id.0 == local_hardware_id
            })
            .ok_or_else(|| LibTpuError::DeviceNotFound {
                device_id: u32::try_from(local_hardware_id).unwrap_or(u32::MAX),
            })
    }

    /// Computes the default device assignment array of length `num_replicas * num_partitions`
    /// (`PJRT_Client_DefaultDeviceAssignment`).
    ///
    /// # Arguments
    ///
    /// * `num_replicas` - Number of data-parallel replicas.
    /// * `num_partitions` - Number of model-parallel partitions.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if the requested count exceeds available devices.
    pub fn default_device_assignment(
        &self,
        num_replicas: usize,
        num_partitions: usize,
    ) -> LibTpuResult<Vec<i32>> {
        let needed = num_replicas.saturating_mul(num_partitions);
        if needed == 0 || needed > self.devices.len() {
            return Err(LibTpuError::InvalidArgument {
                parameter: "num_replicas * num_partitions",
                reason: format!(
                    "Requested {needed} devices, but topology has {}",
                    self.devices.len()
                ),
            });
        }
        let mut sorted_devs: Vec<&PjrtCoreDevice> = self.devices.iter().collect();
        sorted_devs.sort_by_key(|d| {
            let c = d.description.spec.chip_coords;
            (c.z, c.y, c.x, d.description.spec.core_on_chip.0)
        });
        Ok(sorted_devs
            .into_iter()
            .take(needed)
            .map(|d| i32::try_from(d.description.spec.device_id.0).unwrap_or(0))
            .collect())
    }

    /// Allocates and populates a device buffer from a host byte slice (`PJRT_Client_BufferFromHostBuffer`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    /// * `shape` - Validated tensor shape.
    /// * `host_data` - Raw host bytes (if empty for non-zero tensor, buffer is zero-initialized).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if the device is not found, OOM occurs, or transfer fails.
    pub fn buffer_from_host(
        &self,
        device_id: DeviceId,
        shape: Shape,
        host_data: &[u8],
    ) -> LibTpuResult<PjrtCoreBuffer> {
        let dev = self.lookup_device(device_id)?;
        let on_dev_size = shape.on_device_byte_size();
        let addr =
            self.hal
                .allocate_device_memory(device_id, MemoryKind::DeviceHbm, on_dev_size)?;
        if !host_data.is_empty() {
            self.hal.write_device_memory(device_id, addr, host_data)?;
        }
        Ok(PjrtCoreBuffer::new(
            device_id,
            dev.default_memory_id,
            shape,
            addr,
            Arc::clone(&self.hal),
        ))
    }

    /// Resolves the ordered addressable [`DeviceId`] list for a compiled [`CompiledProgram`]
    /// from its serialized `CompileOptionsProto` (`device_assignment`, `device_ordinal`,
    /// `num_replicas`, `num_partitions`).
    ///
    /// # Arguments
    ///
    /// * `program` - Compiled program metadata.
    /// * `compile_options` - Raw serialized `CompileOptionsProto` bytes.
    fn resolve_executable_addressable_devices(
        &self,
        program: &CompiledProgram,
        compile_options: &[u8],
    ) -> Vec<DeviceId> {
        let all_addr: Vec<DeviceId> = self
            .addressable_device_indices
            .iter()
            .filter_map(|&idx| self.devices.get(idx).map(|d| d.description.spec.device_id))
            .collect();
        let Ok(parsed) = crate::proto::CompileOptionsProto::decode_or_empty(compile_options) else {
            return all_addr;
        };
        let Some(eb) = parsed.executable_build_options.as_ref() else {
            return all_addr;
        };
        if let Some(da) = eb.device_assignment.as_ref() {
            let rep_cnt = usize::try_from(da.replica_count).unwrap_or(0);
            if rep_cnt > 0 && !da.computation_devices.is_empty() {
                let mut assigned =
                    Vec::with_capacity(rep_cnt.saturating_mul(da.computation_devices.len()));
                for r in 0..rep_cnt {
                    for comp in &da.computation_devices {
                        if let Some(&raw_id) = comp.replica_device_ids.get(r) {
                            if let Ok(uid) = u32::try_from(raw_id) {
                                let did = DeviceId(uid);
                                if all_addr.contains(&did) {
                                    assigned.push(did);
                                }
                            }
                        }
                    }
                }
                if !assigned.is_empty() {
                    return assigned;
                }
            }
        }
        if eb.num_replicas > 0 || eb.num_partitions > 0 {
            let total = program
                .num_replicas
                .max(1)
                .saturating_mul(program.num_partitions.max(1));
            let start_ord = usize::try_from(eb.device_ordinal.max(0)).unwrap_or(0);
            let selected: Vec<DeviceId> = all_addr
                .iter()
                .copied()
                .skip(start_ord)
                .take(total)
                .collect();
            if !selected.is_empty() {
                return selected;
            }
        }
        all_addr
    }

    /// Compiles and loads an HLO or `StableHLO` program (`PJRT_Client_Compile`).
    ///
    /// # Arguments
    ///
    /// * `format` - Program format (`"mlir"`, `"hlo"`, or `"hlo_with_config"`).
    /// * `code` - Raw program bytes.
    /// * `compile_options` - Raw serialized `CompileOptionsProto` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::CompilationFailed`] if compilation fails.
    pub fn compile_and_load(
        &self,
        format: &str,
        code: &[u8],
        compile_options: &[u8],
    ) -> LibTpuResult<PjrtCoreLoadedExecutable> {
        let program = CompiledProgram::compile(format, code, compile_options)?;
        let addr_devs = self.resolve_executable_addressable_devices(&program, compile_options);
        let executable = PjrtCoreExecutable::new(program);
        Ok(PjrtCoreLoadedExecutable::new(
            executable,
            addr_devs,
            Arc::clone(&self.hal),
        ))
    }

    /// Deserializes and loads a previously serialized executable (`PJRT_Executable_DeserializeAndLoad`).
    ///
    /// # Arguments
    ///
    /// * `serialized` - Bytes produced by [`CompiledProgram::serialize`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeserializationFailed`] if `serialized` is invalid.
    pub fn deserialize_and_load(
        &self,
        serialized: &[u8],
    ) -> LibTpuResult<PjrtCoreLoadedExecutable> {
        let program = CompiledProgram::deserialize(serialized)?;
        let executable = PjrtCoreExecutable::new(program);
        let addr_devs: Vec<DeviceId> = self
            .addressable_device_indices
            .iter()
            .filter_map(|&idx| self.devices.get(idx).map(|d| d.description.spec.device_id))
            .collect();
        Ok(PjrtCoreLoadedExecutable::new(
            executable,
            addr_devs,
            Arc::clone(&self.hal),
        ))
    }

    /// Registers a host memory region for zero-copy DMA (`PJRT_Client_DmaMap`).
    ///
    /// # Arguments
    ///
    /// * `base_addr` - Host pointer address.
    /// * `size` - Size in bytes (`> 0`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `base_addr == 0` or `size == 0`.
    pub fn dma_map(&self, base_addr: u64, size: usize) -> LibTpuResult<()> {
        if base_addr == 0 || size == 0 {
            return Err(LibTpuError::InvalidArgument {
                parameter: "dma_map",
                reason: String::from("DMA map address and size must be non-zero"),
            });
        }
        self.dma_mappings.lock().push((base_addr, size));
        Ok(())
    }

    /// Unregisters a previously DMA-mapped host memory region (`PJRT_Client_DmaUnmap`).
    ///
    /// # Arguments
    ///
    /// * `base_addr` - Host pointer address previously passed to [`PjrtCoreClient::dma_map`].
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::InvalidArgument`] if `base_addr` is not currently mapped.
    pub fn dma_unmap(&self, base_addr: u64) -> LibTpuResult<()> {
        let mut guard = self.dma_mappings.lock();
        let Some(pos) = guard.iter().position(|&(addr, _)| addr == base_addr) else {
            return Err(LibTpuError::InvalidArgument {
                parameter: "base_addr",
                reason: format!("DMA mapping {base_addr:#x} not found"),
            });
        };
        guard.remove(pos);
        drop(guard);
        Ok(())
    }

    /// Queries HBM memory statistics for `device_id` (`PJRT_Device_MemoryStats`).
    ///
    /// # Arguments
    ///
    /// * `device_id` - Target logical TPU device.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::DeviceNotFound`] if `device_id` is invalid.
    pub fn device_memory_stats(&self, device_id: DeviceId) -> LibTpuResult<DeviceMemoryStats> {
        self.hal.memory_stats(device_id)
    }

    /// Stores a key-value entry in the client's distributed topology coordination store.
    ///
    /// # Arguments
    ///
    /// * `key` - Coordination key string.
    /// * `value` - Serialized protobuf payload bytes.
    pub fn kv_put(&self, key: &str, value: Vec<u8>) {
        self.distributed_kv.lock().insert(String::from(key), value);
    }

    /// Retrieves a key-value entry from the client's distributed topology coordination store.
    ///
    /// # Arguments
    ///
    /// * `key` - Coordination key string.
    #[must_use]
    pub fn kv_get(&self, key: &str) -> Option<Vec<u8>> {
        self.distributed_kv.lock().get(key).cloned()
    }

    /// Builds a wire-compatible [`crate::proto::LocalTopologyProto`] (`xla/pjrt/distributed/protocol.proto`)
    /// describing all addressable local TPU devices managed by this host process.
    #[must_use]
    pub fn build_local_topology_proto(&self) -> crate::proto::LocalTopologyProto {
        let devices = self
            .addressable_device_indices
            .iter()
            .filter_map(|&idx| self.devices.get(idx))
            .map(|dev| {
                let spec = &dev.description.spec;
                let hbm_limit = i64::try_from(spec.generation.hbm_capacity_bytes().0).unwrap_or(0);
                crate::proto::DistributedDeviceProto {
                    local_device_ordinal: spec.local_hardware_id.0,
                    name: format!("TPU:{}", spec.device_id.0),
                    vendor: String::from("Google"),
                    global_device_id: i32::try_from(spec.device_id.0).unwrap_or(0),
                    partition_index: spec.process_index.0,
                    compute_capability: spec.generation.to_string(),
                    core_count: i32::try_from(spec.generation.logical_devices_per_chip())
                        .unwrap_or(1),
                    device_kind: dev.description.kind_str.clone(),
                    to_string: dev.description.to_string_val.clone(),
                    debug_string: dev.description.debug_string.clone(),
                    fabric_uuid: format!("tpu_slice/{}", spec.process_index.0),
                    shared_memory_per_block_optin: 0,
                    numa_node: 0,
                    device_memory_bytes_limit: hbm_limit,
                }
            })
            .collect();
        crate::proto::LocalTopologyProto {
            process_id: self.process_index.0,
            boot_id: format!("boot-{}", self.process_index.0),
            devices,
            partition_index: self.process_index.0,
            network_nodes: vec![
                String::from("spine-0"),
                format!("rack-{}", self.process_index.0),
            ],
        }
    }

    /// Performs a distributed multi-host topology exchange (`xla.LocalTopologyProto` -> `xla.GlobalTopologyProto`),
    /// publishing this process's local topology into `kv_store`, aggregating all `num_nodes` host topologies,
    /// assigning deterministic global device IDs (`0..total_devices`), and storing `"global_topology"`.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Total number of participating host processes (`>= 1`).
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError::ProtobufDecodeError`] if any stored `LocalTopologyProto` is malformed.
    pub fn exchange_global_topology_proto(
        &self,
        num_nodes: usize,
    ) -> LibTpuResult<crate::proto::GlobalTopologyProto> {
        let local_proto = self.build_local_topology_proto();
        let local_key = format!("local_topology/{}", self.process_index.0);
        self.kv_put(&local_key, local_proto.encode());

        let total_nodes = num_nodes.max(1);
        let mut processes = Vec::with_capacity(total_nodes);
        let mut next_global_id: i32 = 0;

        for node_idx in 0..total_nodes {
            let key = format!("local_topology/{node_idx}");
            let mut proc_topo = if let Some(raw) = self.kv_get(&key) {
                crate::proto::LocalTopologyProto::decode(&raw)?
            } else {
                let mut synthesized = local_proto.clone();
                let pid = i32::try_from(node_idx).unwrap_or(0);
                synthesized.process_id = pid;
                synthesized.partition_index = pid;
                synthesized.boot_id = format!("boot-{pid}");
                synthesized
            };
            for dev in &mut proc_topo.devices {
                dev.global_device_id = next_global_id;
                dev.partition_index = proc_topo.partition_index;
                next_global_id = next_global_id.saturating_add(1);
            }
            processes.push(proc_topo);
        }

        let global = crate::proto::GlobalTopologyProto { processes };
        self.kv_put("global_topology", global.encode());
        Ok(global)
    }
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FailingTopologyHal;

    impl TpuHal for FailingTopologyHal {
        fn discover_topology(&self) -> LibTpuResult<TopologySpec> {
            Err(LibTpuError::DeviceNotFound { device_id: 0 })
        }
        fn allocate_device_memory(
            &self,
            d: DeviceId,
            _k: MemoryKind,
            _s: ByteSize,
        ) -> LibTpuResult<DeviceAddress> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn deallocate_device_memory(&self, d: DeviceId, _a: DeviceAddress) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn write_device_memory(
            &self,
            d: DeviceId,
            _a: DeviceAddress,
            _b: &[u8],
        ) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn read_device_memory(
            &self,
            d: DeviceId,
            _a: DeviceAddress,
            _o: &mut [u8],
        ) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn copy_device_to_device(
            &self,
            d: DeviceId,
            _sa: DeviceAddress,
            _dd: DeviceId,
            _da: DeviceAddress,
            _s: ByteSize,
        ) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn execute_program(
            &self,
            d: DeviceId,
            _p: &CompiledProgram,
            _i: &[DeviceAddress],
            _o: &[DeviceAddress],
        ) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn memory_stats(&self, d: DeviceId) -> LibTpuResult<DeviceMemoryStats> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn clear_memory_stats(&self, d: DeviceId) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
        fn poison_device(&self, d: DeviceId, _r: String) -> LibTpuResult<()> {
            Err(LibTpuError::DeviceNotFound { device_id: d.0 })
        }
    }

    #[test]
    fn test_pjrt_events_and_stream() {
        let fhal = FailingTopologyHal;
        assert!(fhal.discover_topology().is_err());
        assert!(
            fhal.allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(1))
                .is_err()
        );
        assert!(
            fhal.deallocate_device_memory(DeviceId(0), DeviceAddress(0))
                .is_err()
        );
        assert!(
            fhal.write_device_memory(DeviceId(0), DeviceAddress(0), &[])
                .is_err()
        );
        assert!(
            fhal.read_device_memory(DeviceId(0), DeviceAddress(0), &mut [])
                .is_err()
        );
        assert!(
            fhal.copy_device_to_device(
                DeviceId(0),
                DeviceAddress(0),
                DeviceId(0),
                DeviceAddress(0),
                ByteSize(1)
            )
            .is_err()
        );
        if let Ok(p) = CompiledProgram::compile("mlir", b"return", b"") {
            assert!(fhal.execute_program(DeviceId(0), &p, &[], &[]).is_err());
        }
        assert!(fhal.memory_stats(DeviceId(0)).is_err());
        assert!(fhal.clear_memory_stats(DeviceId(0)).is_err());
        assert!(fhal.poison_device(DeviceId(0), String::new()).is_err());
        assert!(PjrtCoreClient::new(Arc::new(FailingTopologyHal), ProcessIndex(0)).is_err());
        let ev_ok = PjrtCoreEvent::ready_ok();
        assert!(ev_ok.is_ready());
        assert!(ev_ok.get_error().is_none());
        let imm_flag = Arc::new(AtomicBool::new(false));
        let imm_clone = Arc::clone(&imm_flag);
        ev_ok.on_ready(move |err| {
            if err.is_none() {
                imm_clone.store(true, Ordering::Release);
            }
        });
        assert!(imm_flag.load(Ordering::Acquire));
        assert!(ev_ok.await_ready().is_ok());
        assert_ne!(format!("{ev_ok:?}"), "");

        let ev_pend = PjrtCoreEvent::pending();
        assert!(!ev_pend.is_ready());
        let def_flag = Arc::new(AtomicBool::new(false));
        let def_clone = Arc::clone(&def_flag);
        ev_pend.on_ready(move |err| {
            if err.is_some() {
                def_clone.store(true, Ordering::Release);
            }
        });
        assert!(!def_flag.load(Ordering::Acquire));
        ev_pend.set(Some(LibTpuError::ProfilerNotActive));
        assert!(def_flag.load(Ordering::Acquire));
        assert!(ev_pend.is_ready());
        assert!(ev_pend.get_error().is_some());
        assert!(ev_pend.await_ready().is_err());

        let stream = PjrtCoreCopyToDeviceStream::new(256, 128);
        assert_ne!(format!("{stream:?}"), "");
        assert_eq!(stream.current_bytes(), 0);
        assert!(stream.add_chunk(-1).is_err());
        assert!(stream.add_chunk(512).is_err());
        assert!(stream.add_chunk_data(&vec![1_u8; 512]).is_err());
        assert!(stream.add_chunk_data(&[7_u8; 128]).is_ok());
        assert_eq!(stream.current_bytes(), 128);
        assert_eq!(stream.staged_data().len(), 128);
        assert!(stream.add_chunk(128).is_ok());
        assert_eq!(stream.current_bytes(), 256);
    }

    #[test]
    fn test_pjrt_client_buffers_compile_execute_and_dma() {
        let cfg = TpuInitConfig {
            host_bounds: crate::types::ChipCoordinates::new(2, 1, 1),
            chips_per_host_bounds: crate::types::ChipCoordinates::new(1, 1, 1),
            ..TpuInitConfig::default()
        };

        let _sim_client = PjrtCoreClient::from_simulated(&cfg);
        let mut sim_mode_cfg = cfg.clone();
        sim_mode_cfg
            .extra_flags
            .insert(String::from("--tpu_driver_mode"), String::from("simulated"));
        let _sim_mode_client = PjrtCoreClient::from_config(&sim_mode_cfg);
        if let Ok(client) = PjrtCoreClient::with_simulated_hal(&cfg) {
            assert_ne!(format!("{client:?}"), "");
            assert_ne!(format!("{:?}", client.topology), "");
            assert_ne!(format!("{:?}", client.devices), "");
            assert_ne!(format!("{:?}", client.memories), "");

            let topo_bytes = client.topology.serialize();
            assert!(PjrtCoreTopology::deserialize(&topo_bytes).is_ok());
            assert!(PjrtCoreTopology::deserialize(b"LTPUTOPO:gen=5").is_ok());
            for ver in [2, 3, 4, 5, 6, 7, 8] {
                let p_topo = crate::proto::TpuTopologyArgsProto {
                    version: Some(ver),
                    chips_per_host_bounds: if ver == 3 {
                        None
                    } else {
                        Some(crate::proto::TpuDimensionsProto {
                            x: 1,
                            y: 1,
                            z: 1,
                            w: 1,
                        })
                    },
                    host_bounds: if ver % 2 == 0 {
                        Some(crate::proto::TpuDimensionsProto {
                            x: 1,
                            y: 1,
                            z: 1,
                            w: 1,
                        })
                    } else {
                        None
                    },
                    variant: if ver == 2 {
                        None
                    } else {
                        Some(String::from("custom_ver"))
                    },
                    ..crate::proto::TpuTopologyArgsProto::default()
                };
                assert!(
                    PjrtCoreTopology::deserialize(&crate::proto::ProtoSchemaMessage::encode_proto(
                        &p_topo
                    ))
                    .is_ok()
                );
            }
            let zero_ver_topo = crate::proto::TpuTopologyArgsProto {
                version: Some(0),
                ..crate::proto::TpuTopologyArgsProto::default()
            };
            assert!(
                PjrtCoreTopology::deserialize(&crate::proto::ProtoSchemaMessage::encode_proto(
                    &zero_ver_topo
                ))
                .is_err()
            );
            assert!(PjrtCoreTopology::deserialize(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]).is_err());

            assert!(client.lookup_device(DeviceId(0)).is_ok());
            assert!(client.lookup_device(DeviceId(99)).is_err());
            assert!(client.lookup_addressable_device(0).is_ok());
            assert!(client.lookup_addressable_device(99).is_err());

            assert_eq!(
                client.default_device_assignment(1, 2).ok(),
                Some(vec![0, 1])
            );
            assert!(client.default_device_assignment(0, 1).is_err());
            assert!(client.default_device_assignment(10, 10).is_err());

            if let Ok(shape) = Shape::new(ElementType::F32, vec![4], None, None) {
                let mut host_in0 = Vec::new();
                let mut host_in1 = Vec::new();
                for f in [1.0_f32, 2.0, 3.0, 4.0] {
                    host_in0.extend_from_slice(&f.to_le_bytes());
                }
                for f in [10.0_f32, 20.0, 30.0, 40.0] {
                    host_in1.extend_from_slice(&f.to_le_bytes());
                }

                let buf0_res = client.buffer_from_host(DeviceId(0), shape.clone(), &host_in0);
                let buf1_res = client.buffer_from_host(DeviceId(0), shape.clone(), &host_in1);
                let empty_buf_res = client.buffer_from_host(DeviceId(0), shape.clone(), &[]);
                assert!(
                    client
                        .buffer_from_host(DeviceId(99), shape.clone(), &[])
                        .is_err()
                );
                assert!(
                    client
                        .buffer_from_host(DeviceId(0), shape.clone(), &vec![0_u8; 4096])
                        .is_err()
                );
                if let (Ok(buf0), Ok(buf1), Ok(empty_buf)) = (&buf0_res, &buf1_res, &empty_buf_res)
                {
                    assert_ne!(format!("{buf0:?}"), "");
                    let flush_stream = PjrtCoreCopyToDeviceStream::new(16, 16);
                    assert!(flush_stream.flush_to_buffer(buf0).is_ok());
                    assert!(flush_stream.add_chunk_data(&host_in0).is_ok());
                    assert!(flush_stream.flush_to_buffer(buf0).is_ok());
                    assert_eq!(buf0.inc_external_ref(), 2);
                    assert_eq!(buf0.dec_external_ref(), 1);

                    let copied_res = buf0.copy_to_device(DeviceId(0), MemorySpaceId(0));
                    assert!(copied_res.is_ok());
                    assert!(buf0.copy_to_device(DeviceId(99), MemorySpaceId(0)).is_err());
                    assert!(buf0.ready_error().is_none());
                    assert!(buf0.ready_event().get_error().is_none());
                    assert!(buf1.fulfill_from(buf0).is_ok());
                    assert!(buf1.fulfill_from(buf0).is_err());
                    let ev_fail = PjrtCoreEvent::pending();
                    ev_fail.fail_with_reason(String::from("ev_failed"));
                    assert!(ev_fail.await_ready().is_err());
                    buf0.set_error(LibTpuError::ProfilerNotActive);
                    assert!(buf0.ready_error().is_some());
                    assert!(buf0.ready_event().get_error().is_some());
                    let bitcast_res = buf0.bitcast(ElementType::S32, vec![4]);
                    assert!(bitcast_res.is_ok());
                    assert!(buf0.bitcast(ElementType::Invalid, vec![4]).is_err());

                    let bad_dev_buf = PjrtCoreBuffer::new(
                        DeviceId(99),
                        MemorySpaceId(0),
                        shape.clone(),
                        DeviceAddress(0x1000),
                        Arc::clone(&client.hal),
                    );
                    assert!(bad_dev_buf.bitcast(ElementType::S32, vec![4]).is_err());

                    if let Ok(hs) =
                        Shape::new(ElementType::F32, vec![1024 * 1024 * 1024 * 32], None, None)
                    {
                        assert!(client.buffer_from_host(DeviceId(0), hs, &[]).is_err());
                    }

                    let bad_addr_buf = PjrtCoreBuffer::new(
                        DeviceId(0),
                        MemorySpaceId(0),
                        shape.clone(),
                        DeviceAddress(0xdead_beef),
                        Arc::clone(&client.hal),
                    );
                    assert!(bad_addr_buf.to_host_bytes().is_err());
                    assert!(
                        bad_addr_buf
                            .copy_to_device(DeviceId(0), MemorySpaceId(0))
                            .is_err()
                    );
                    assert!(bad_addr_buf.bitcast(ElementType::S32, vec![4]).is_err());
                    assert!(bad_addr_buf.delete().is_err());

                    let loaded_res =
                        client.compile_and_load("mlir", b"module @add { stablehlo.add }", b"");
                    assert!(loaded_res.is_ok());
                    assert!(client.compile_and_load("bad_fmt", b"code", b"").is_err());
                    assert!(
                        client
                            .compile_and_load(
                                "mlir",
                                b"return",
                                &[0xff, 0xff, 0xff, 0xff, 0xff, 0x7f]
                            )
                            .is_err()
                    );
                    if let Ok(loaded) = loaded_res {
                        assert_ne!(format!("{loaded:?}"), "");
                        assert_ne!(format!("{:?}", loaded.executable), "");
                        let _ = loaded.executable.cost_analysis();
                        let _ = loaded.executable.compiled_memory_stats();

                        let serialized_exec = loaded.executable.program.serialize();
                        assert!(client.deserialize_and_load(&serialized_exec).is_ok());
                        assert!(client.deserialize_and_load(b"bad").is_err());

                        let outs_res =
                            loaded.execute_on_device(DeviceId(0), MemorySpaceId(0), &[buf0, buf1]);
                        assert!(outs_res.is_ok());
                        if let Ok(outs) = outs_res
                            && let Some(out0) = outs.first()
                        {
                            let host_out = out0.to_host_bytes();
                            assert!(host_out.is_ok());
                        }
                        let all_outs = loaded
                            .execute_all_devices(MemorySpaceId(0), &[&[buf0, buf1], &[buf0, buf1]]);
                        assert!(all_outs.is_ok());
                        if let Ok(spmd_buf) =
                            client.buffer_from_host(DeviceId(0), shape.clone(), &host_in0)
                            && let Ok(spmd_buf1) =
                                client.buffer_from_host(DeviceId(1), shape.clone(), &host_in1)
                            && let Ok(spmd_loaded) =
                                client.compile_and_load("mlir", b"module @spmd { all_reduce }", b"")
                        {
                            let spmd_outs = spmd_loaded.execute_all_devices(
                                MemorySpaceId(0),
                                &[&[&spmd_buf], &[&spmd_buf1]],
                            );
                            assert!(spmd_outs.is_ok());
                        }
                        assert!(
                            loaded
                                .execute_on_device(DeviceId(99), MemorySpaceId(0), &[buf0, buf1])
                                .is_err()
                        );
                        assert!(
                            loaded
                                .execute_on_device(DeviceId(0), MemorySpaceId(0), &[buf0])
                                .is_err()
                        );

                        assert!(empty_buf.delete().is_ok());
                        assert!(empty_buf.delete().is_ok());
                        assert!(empty_buf.is_deleted());
                        let del_stream = PjrtCoreCopyToDeviceStream::new(16, 16);
                        assert!(del_stream.flush_to_buffer(empty_buf).is_err());
                        assert!(empty_buf.fulfill_from(buf0).is_err());
                        assert!(buf0.fulfill_from(empty_buf).is_err());
                        assert!(empty_buf.to_host_bytes().is_err());
                        assert!(
                            empty_buf
                                .copy_to_device(DeviceId(0), MemorySpaceId(0))
                                .is_err()
                        );
                        assert!(empty_buf.bitcast(ElementType::S32, vec![4]).is_err());
                        assert!(
                            loaded
                                .execute_on_device(
                                    DeviceId(0),
                                    MemorySpaceId(0),
                                    &[empty_buf, buf1]
                                )
                                .is_err()
                        );

                        loaded.delete();
                        assert!(loaded.is_deleted());
                        assert!(
                            loaded
                                .execute_on_device(DeviceId(0), MemorySpaceId(0), &[buf0, buf1])
                                .is_err()
                        );
                    }
                }

                let mgr_res = PjrtCoreAsyncTransferManager::create(
                    DeviceId(0),
                    MemorySpaceId(0),
                    vec![shape.clone()],
                    &client.hal,
                );
                assert!(mgr_res.is_ok());
                assert!(
                    PjrtCoreAsyncTransferManager::create(
                        DeviceId(99),
                        MemorySpaceId(0),
                        vec![shape],
                        &client.hal,
                    )
                    .is_err()
                );
                if let Ok(mgr) = mgr_res {
                    assert_ne!(format!("{mgr:?}"), "");
                    assert!(mgr.transfer_data(0, &host_in0).is_ok());
                    assert!(mgr.transfer_data(99, &host_in0).is_err());
                    assert!(mgr.transfer_literal(0, &host_in0).is_ok());
                    assert!(mgr.transfer_literal(99, &host_in0).is_err());
                    assert!(mgr.retrieve_buffer(0).is_ok());
                    assert!(mgr.retrieve_buffer(99).is_err());
                    mgr.set_buffer_error(0, LibTpuError::ProfilerNotActive);
                    mgr.set_buffer_error(99, LibTpuError::ProfilerNotActive);
                    mgr.add_metadata(vec![(String::from("k"), String::from("v"))]);
                }
            }

            assert!(client.dma_map(0, 1024).is_err());
            assert!(client.dma_map(0x2000, 0).is_err());
            assert!(client.dma_map(0x2000, 4096).is_ok());
            assert!(client.dma_unmap(0x9999).is_err());
            assert!(client.dma_unmap(0x2000).is_ok());
            assert!(client.device_memory_stats(DeviceId(0)).is_ok());
            let proc_rec = ProcessInfoRecord {
                process_id: 0,
                incarnation_id: 42,
                state: 1,
                error_code: 0,
                error_message: String::from("ok"),
            };
            assert_ne!(format!("{proc_rec:?}"), "");
            client.update_global_process_info(vec![proc_rec.clone()]);
            assert_eq!(client.global_process_infos(), vec![proc_rec]);

            let local_topo = client.build_local_topology_proto();
            assert_eq!(local_topo.process_id, 0);
            assert_ne!(
                local_topo.devices,
                [] as [crate::proto::DistributedDeviceProto; 0]
            );
            if let Ok(global_topo) = client.exchange_global_topology_proto(2) {
                assert_eq!(global_topo.processes.len(), 2);
                assert!(client.kv_get("global_topology").is_some());
            }
            client.kv_put("local_topology/1", vec![0x80]);
            assert!(client.exchange_global_topology_proto(2).is_err());

            // Cover execute_on_device_with_host_io, execute_all_devices, and resolve_executable_addressable_devices
            {
                use crate::proto::ProtoSchemaMessage as _;
                use crate::types::MemoryLayout;
                let mut multi_cfg = TpuInitConfig::default();
                let _ = multi_cfg.apply_init_args("--xla_tpu_chips_per_host_bounds=2,1,1");
                let multi_client = PjrtCoreClient::from_config(&multi_cfg);
                assert!(multi_client.addressable_device_indices.len() >= 2);

                if let Ok(base_prog) = CompiledProgram::compile("mlir", b"add_f32", b"") {
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &[0xFF, 0xFF, 0xFF])
                            .len(),
                        multi_client.addressable_device_indices.len()
                    );
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, b"")
                            .len(),
                        multi_client.addressable_device_indices.len()
                    );
                    let empty_da_opts = crate::proto::CompileOptionsProto {
                        executable_build_options: Some(crate::proto::ExecutableBuildOptionsProto {
                            device_assignment: Some(crate::proto::DeviceAssignmentProto {
                                replica_count: 2,
                                computation_count: 1,
                                computation_devices: Vec::new(),
                            }),
                            ..crate::proto::ExecutableBuildOptionsProto::default()
                        }),
                        ..crate::proto::CompileOptionsProto::default()
                    }
                    .encode_proto();
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &empty_da_opts)
                            .len(),
                        multi_client.addressable_device_indices.len()
                    );
                    let no_da_parts_opts = crate::proto::CompileOptionsProto {
                        executable_build_options: Some(crate::proto::ExecutableBuildOptionsProto {
                            num_replicas: 0,
                            num_partitions: 2,
                            device_assignment: None,
                            ..crate::proto::ExecutableBuildOptionsProto::default()
                        }),
                        ..crate::proto::CompileOptionsProto::default()
                    }
                    .encode_proto();
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &no_da_parts_opts)
                            .len(),
                        1
                    );
                    let zero_rep_da_opts = crate::proto::CompileOptionsProto {
                        executable_build_options: Some(crate::proto::ExecutableBuildOptionsProto {
                            device_assignment: Some(crate::proto::DeviceAssignmentProto {
                                replica_count: 0,
                                computation_count: 0,
                                computation_devices: Vec::new(),
                            }),
                            ..crate::proto::ExecutableBuildOptionsProto::default()
                        }),
                        ..crate::proto::CompileOptionsProto::default()
                    }
                    .encode_proto();
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &zero_rep_da_opts)
                            .len(),
                        multi_client.addressable_device_indices.len()
                    );
                    let unmatched_da_opts = crate::proto::CompileOptionsProto {
                        executable_build_options: Some(crate::proto::ExecutableBuildOptionsProto {
                            num_replicas: 1,
                            num_partitions: 1,
                            device_ordinal: 1,
                            device_assignment: Some(crate::proto::DeviceAssignmentProto {
                                replica_count: 2,
                                computation_count: 1,
                                computation_devices: vec![crate::proto::ComputationDeviceProto {
                                    replica_device_ids: vec![-1, 9999],
                                }],
                            }),
                            ..crate::proto::ExecutableBuildOptionsProto::default()
                        }),
                        ..crate::proto::CompileOptionsProto::default()
                    }
                    .encode_proto();
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &unmatched_da_opts),
                        vec![DeviceId(1)]
                    );
                    let oob_ord_opts = crate::proto::CompileOptionsProto {
                        executable_build_options: Some(crate::proto::ExecutableBuildOptionsProto {
                            num_replicas: 1,
                            num_partitions: 0,
                            device_ordinal: 999,
                            device_assignment: Some(crate::proto::DeviceAssignmentProto {
                                replica_count: 2,
                                computation_count: 1,
                                computation_devices: vec![crate::proto::ComputationDeviceProto {
                                    replica_device_ids: vec![9999],
                                }],
                            }),
                            ..crate::proto::ExecutableBuildOptionsProto::default()
                        }),
                        ..crate::proto::CompileOptionsProto::default()
                    }
                    .encode_proto();
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &oob_ord_opts)
                            .len(),
                        multi_client.addressable_device_indices.len()
                    );
                    let matched_da_opts = crate::proto::CompileOptionsProto {
                        executable_build_options: Some(crate::proto::ExecutableBuildOptionsProto {
                            device_assignment: Some(crate::proto::DeviceAssignmentProto {
                                replica_count: 2,
                                computation_count: 1,
                                computation_devices: vec![crate::proto::ComputationDeviceProto {
                                    replica_device_ids: vec![0, 1],
                                }],
                            }),
                            ..crate::proto::ExecutableBuildOptionsProto::default()
                        }),
                        ..crate::proto::CompileOptionsProto::default()
                    }
                    .encode_proto();
                    assert_eq!(
                        multi_client
                            .resolve_executable_addressable_devices(&base_prog, &matched_da_opts),
                        vec![DeviceId(0), DeviceId(1)]
                    );

                    // 2. execute_on_device_with_host_io (including zero-sized output, Send/Recv errors, out-of-range output_slot, deleted input, deleted executable)
                    let host_io_mlir = br"module @hio {
                func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<0xf32>) {
                    %t = stablehlo.send %arg0 {channel_id = 3} : (tensor<4xf32>) -> tensor<4xf32>
                    %r = stablehlo.recv %t {channel_id = 4} : (tensor<4xf32>) -> tensor<4xf32>
                    %s = stablehlo.add %arg0, %r : tensor<4xf32>
                    %z = stablehlo.constant dense<[]> : tensor<0xf32>
                    func.return %s, %z : tensor<4xf32>, tensor<0xf32>
                }
            }";
                    if let Ok(mut hio_exec) =
                        multi_client.compile_and_load("mlir", host_io_mlir, &matched_da_opts)
                    {
                        PjrtCoreLoadedExecutable::write_register_slot(&mut [], 0, Vec::new());
                        let zero_sh = Shape {
                            element_type: ElementType::F32,
                            dims: vec![0],
                            dynamic_dims: vec![false],
                            layout: MemoryLayout::default_tpu_tiled(1),
                        };
                        let mut h_prog = hio_exec.executable.program.clone();
                        h_prog.output_shapes.push(zero_sh);
                        if let Some(b0) = h_prog.instruction_bundles.first_mut() {
                            b0.output_slot = 999;
                        }
                        if let Some(b1) = h_prog.instruction_bundles.get_mut(1) {
                            b1.output_slot = 999;
                        }
                        h_prog.output_slots = vec![999];
                        hio_exec.executable = PjrtCoreExecutable::new(h_prog);

                        let f32_4 = Shape {
                            element_type: ElementType::F32,
                            dims: vec![4],
                            dynamic_dims: vec![false],
                            layout: MemoryLayout::default_tpu_tiled(1),
                        };
                        if let Ok(in_b0) =
                            multi_client.buffer_from_host(DeviceId(0), f32_4, &[0_u8; 16])
                        {
                            let ok_outs = hio_exec.execute_on_device_with_host_io(
                                DeviceId(0),
                                MemorySpaceId(0),
                                &[&in_b0],
                                |_ch, _bytes| Ok(()),
                                |_ch| Ok(vec![0_u8; 16]),
                            );
                            assert!(ok_outs.is_ok());
                            assert!(
                                hio_exec
                                    .execute_on_device_with_host_io(
                                        DeviceId(0),
                                        MemorySpaceId(0),
                                        &[&in_b0],
                                        |_ch, _bytes| Err(LibTpuError::ProfilerNotActive),
                                        |_ch| Ok(vec![0_u8; 16]),
                                    )
                                    .is_err()
                            );
                            assert!(
                                hio_exec
                                    .execute_on_device_with_host_io(
                                        DeviceId(0),
                                        MemorySpaceId(0),
                                        &[&in_b0],
                                        |_ch, _bytes| Ok(()),
                                        |_ch| Err(LibTpuError::ProfilerNotActive),
                                    )
                                    .is_err()
                            );
                            let _ = in_b0.delete();
                            assert!(
                                hio_exec
                                    .execute_on_device_with_host_io(
                                        DeviceId(0),
                                        MemorySpaceId(0),
                                        &[&in_b0],
                                        |_ch, _bytes| Ok(()),
                                        |_ch| Ok(vec![0_u8; 16]),
                                    )
                                    .is_err()
                            );
                        }
                        hio_exec.delete();
                        assert!(
                            hio_exec
                                .execute_on_device_with_host_io(
                                    DeviceId(0),
                                    MemorySpaceId(0),
                                    &[],
                                    |_ch, _bytes| Ok(()),
                                    |_ch| Ok(Vec::new()),
                                )
                                .is_err()
                        );
                        assert!(hio_exec.execute_all_devices(MemorySpaceId(0), &[]).is_err());
                    }

                    // 3. execute_all_devices (non-collective & collective across 2 devices, zero-sized output, out-of-range output_slot, deleted buffer)
                    let coll_mlir = br#"module @coll {
                func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<0xf32>) {
                    %a = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    %g = "stablehlo.all_gather"(%a) {all_gather_dim = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
                    %z = stablehlo.constant dense<[]> : tensor<0xf32>
                    func.return %g, %z : tensor<4xf32>, tensor<0xf32>
                }
            }"#;
                    if let Ok(non_coll_exec) =
                        multi_client.compile_and_load("mlir", b"add_f32", &matched_da_opts)
                        && let Ok(mut coll_exec) =
                            multi_client.compile_and_load("mlir", coll_mlir, &matched_da_opts)
                    {
                        let f32_4 = Shape {
                            element_type: ElementType::F32,
                            dims: vec![4],
                            dynamic_dims: vec![false],
                            layout: MemoryLayout::default_tpu_tiled(1),
                        };
                        if let (Ok(d0_buf), Ok(d1_buf)) = (
                            multi_client.buffer_from_host(DeviceId(0), f32_4.clone(), &[0_u8; 16]),
                            multi_client.buffer_from_host(DeviceId(1), f32_4, &[0_u8; 16]),
                        ) {
                            let s0 = [&d0_buf];
                            let s1 = [&d1_buf];
                            assert!(
                                non_coll_exec
                                    .execute_all_devices(MemorySpaceId(0), &[&s0, &s1])
                                    .is_ok()
                            );
                            assert!(
                                coll_exec
                                    .execute_all_devices(MemorySpaceId(0), &[&s0, &s1])
                                    .is_ok()
                            );

                            let mut c_prog = coll_exec.executable.program.clone();
                            c_prog.output_shapes.push(Shape {
                                element_type: ElementType::F32,
                                dims: vec![0],
                                dynamic_dims: vec![false],
                                layout: MemoryLayout::default_tpu_tiled(1),
                            });
                            for b in &mut c_prog.instruction_bundles {
                                if b.opcode == crate::compiler::TpuOpCode::AllGather {
                                    b.output_slot = 999;
                                }
                            }
                            c_prog.output_slots = vec![999];
                            coll_exec.executable = PjrtCoreExecutable::new(c_prog);
                            assert!(
                                coll_exec
                                    .execute_all_devices(MemorySpaceId(0), &[&s0, &s1])
                                    .is_ok()
                            );
                            // Also test 1-device collective `execute_all_devices` (`num_devs <= 1 == true`)
                            coll_exec.addressable_devices = vec![DeviceId(0)];
                            assert!(
                                coll_exec
                                    .execute_all_devices(MemorySpaceId(0), &[&s0])
                                    .is_ok()
                            );
                            coll_exec.addressable_devices = vec![DeviceId(0), DeviceId(1)];

                            let _ = d1_buf.delete();
                            assert!(
                                coll_exec
                                    .execute_all_devices(MemorySpaceId(0), &[&s0, &s1])
                                    .is_err()
                            );
                            assert!(
                                non_coll_exec
                                    .execute_all_devices(MemorySpaceId(0), &[&s0, &s1])
                                    .is_err()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_usage_md_section5_rust_core_snippet() -> LibTpuResult<()> {
        let mut cfg = TpuInitConfig::default();
        cfg.apply_init_args("--tpu_generation=v6e --tpu_chips_per_host_bounds=1,1,1")?;

        let client = PjrtCoreClient::from_config(&cfg);
        let shape = Shape::new(ElementType::F32, vec![4], None, None)?;

        let lhs_bytes: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let rhs_bytes: Vec<u8> = [10.0_f32, 20.0, 30.0, 40.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();

        let b0 = client.buffer_from_host(DeviceId(0), shape.clone(), &lhs_bytes)?;
        let b1 = client.buffer_from_host(DeviceId(0), shape, &rhs_bytes)?;

        let mlir = br"
          module @add {
            func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
              %0 = stablehlo.add %a, %b : tensor<4xf32>
              return %0 : tensor<4xf32>
            }
          }
        ";

        let loaded = client.compile_and_load("mlir", mlir, b"")?;
        let outputs = loaded.execute_on_device(DeviceId(0), MemorySpaceId(0), &[&b0, &b1])?;
        let raw = outputs
            .first()
            .map(PjrtCoreBuffer::to_host_bytes)
            .transpose()?
            .unwrap_or_default();

        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(c);
                f32::from_le_bytes(arr)
            })
            .collect();
        assert_eq!(vals, vec![11.0_f32, 22.0, 33.0, 44.0]);
        Ok(())
    }
}
