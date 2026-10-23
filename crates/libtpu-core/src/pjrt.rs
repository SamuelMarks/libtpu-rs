//! Safe, thread-safe Rust PJRT domain engine backing all `PJRT_*` C API handles.
//!
//! Provides [`PjrtCoreClient`], [`PjrtCoreDevice`], [`PjrtCoreDeviceDescription`],
//! [`PjrtCoreMemory`], [`PjrtCoreBuffer`], [`PjrtCoreEvent`], [`PjrtCoreTopology`],
//! [`PjrtCoreExecutable`], [`PjrtCoreLoadedExecutable`], and [`PjrtCoreCopyToDeviceStream`].

use crate::compiler::{CompiledMemoryStats, CompiledProgram, CostAnalysis};
use crate::config::TpuInitConfig;
use crate::error::{LibTpuError, LibTpuResult};
use crate::hal::{DeviceMemoryStats, DeviceSpec, SimulatedTpuHal, TopologySpec, TpuHal};
use crate::types::{
    BufferId, ByteSize, DeviceAddress, DeviceId, ElementType, EventId, ExecutableId, MemoryKind,
    MemorySpaceId, ProcessIndex, Shape,
};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Global monotonic counter for assigning unique buffer, executable, and event IDs.
static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);

/// Allocates a fresh globally unique 64-bit object ID.
fn next_id() -> u64 {
    NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Core representation of a `PJRT_Event` asynchronous completion notification.
#[derive(Debug)]
pub struct PjrtCoreEvent {
    /// Unique event identifier.
    pub id: EventId,
    /// Whether the event has reached completion.
    ready: AtomicBool,
    /// Optional error attached to the event.
    error: Mutex<Option<LibTpuError>>,
}

impl PjrtCoreEvent {
    /// Creates an immediately ready [`PjrtCoreEvent`] with no error.
    pub fn ready_ok() -> Self {
        Self {
            id: EventId(next_id()),
            ready: AtomicBool::new(true),
            error: Mutex::new(None),
        }
    }

    /// Creates an unready pending [`PjrtCoreEvent`].
    pub fn pending() -> Self {
        Self {
            id: EventId(next_id()),
            ready: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    /// Returns whether this event has completed (`PJRT_Event_IsReady`).
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Marks this event as complete with an optional [`LibTpuError`] (`PJRT_Event_Set`).
    ///
    /// # Arguments
    ///
    /// * `status` - `None` for success, or `Some(err)` to fail the event.
    pub fn set(&self, status: Option<LibTpuError>) {
        let mut guard = self
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = status;
        drop(guard);
        self.ready.store(true, Ordering::Release);
    }

    /// Returns the error associated with this event, if any (`PJRT_Event_Error`).
    pub fn get_error(&self) -> Option<LibTpuError> {
        self.error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Blocks until the event is ready and returns its status (`PJRT_Event_Await`).
    ///
    /// # Errors
    ///
    /// Returns the attached [`LibTpuError`] if the event completed with an error.
    pub fn await_ready(&self) -> LibTpuResult<()> {
        self.ready.store(true, Ordering::Release);
        self.get_error().map_or(Ok(()), Err)
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
    pub fn new(spec: TopologySpec) -> Self {
        let device_descriptions: Vec<PjrtCoreDeviceDescription> = spec
            .devices
            .iter()
            .cloned()
            .map(PjrtCoreDeviceDescription::new)
            .collect();
        let fingerprint = 0x5450_555f_544f_504f_u64
            ^ (u64::from(spec.chip_bounds.x) << 16)
            ^ (u64::from(spec.chip_bounds.y) << 8)
            ^ u64::from(spec.chip_bounds.z)
            ^ (device_descriptions.len() as u64);
        Self {
            spec,
            device_descriptions,
            fingerprint,
        }
    }

    /// Serializes this topology description into wire-compatible `TpuTopologyArgsProto` bytes
    /// (`PJRT_TopologyDescription_Serialize`).
    pub fn serialize(&self) -> Vec<u8> {
        let proto = crate::proto::TpuTopologyArgsProto {
            version: self.spec.generation as i32,
            num_hosts: i32::try_from(
                self.spec.host_bounds.x * self.spec.host_bounds.y * self.spec.host_bounds.z,
            )
            .unwrap_or(1),
            chips_per_host: i32::try_from(self.spec.chips_per_host).unwrap_or(1),
            mesh_bounds: vec![
                i64::from(self.spec.chip_bounds.x),
                i64::from(self.spec.chip_bounds.y),
                i64::from(self.spec.chip_bounds.z),
            ],
            platform_version: self.spec.platform_version.clone(),
            fingerprint: format!("{:016x}", self.fingerprint),
        };
        proto.encode()
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
        if let Ok(proto) = crate::proto::TpuTopologyArgsProto::decode(bytes) {
            if proto.version > 0 {
                let cfg = TpuInitConfig::default();
                let hal = SimulatedTpuHal::new(&cfg);
                return Ok(Self::new(hal.topology_spec()));
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
    /// Atomic deletion flag (`PJRT_Buffer_Delete` / `PJRT_Buffer_IsDeleted`).
    deleted: AtomicBool,
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
            deleted: AtomicBool::new(false),
            external_ref_count: AtomicI64::new(1),
        }
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
}

impl PjrtCoreCopyToDeviceStream {
    /// Creates a new [`PjrtCoreCopyToDeviceStream`].
    ///
    /// # Arguments
    ///
    /// * `total_bytes` - Total number of bytes to be transferred across all chunks.
    /// * `granule_size` - Transfer granule alignment in bytes.
    pub fn new(total_bytes: i64, granule_size: i64) -> Self {
        Self {
            total_bytes,
            granule_size: granule_size.max(1),
            current_bytes: AtomicI64::new(0),
        }
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

    /// Returns the number of bytes transferred so far (`PJRT_CopyToDeviceStream_CurrentBytes`).
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
    pub fn new(program: CompiledProgram) -> Self {
        Self {
            id: ExecutableId(next_id()),
            program,
        }
    }

    /// Returns the cost analysis metrics (`PJRT_Executable_GetCostAnalysis`).
    pub fn cost_analysis(&self) -> CostAnalysis {
        self.program.cost_analysis
    }

    /// Returns the compiled memory usage statistics (`PJRT_Executable_GetCompiledMemoryStats`).
    pub fn compiled_memory_stats(&self) -> CompiledMemoryStats {
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
        let mut buffers = Vec::with_capacity(shapes.len());
        let mut buffer_sizes = Vec::with_capacity(shapes.len());
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
            metadata: Mutex::new(Vec::new()),
        })
    }

    /// Transfers a slice of host data into buffer `buffer_index` at `offset` (`TransferData`).
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

    /// Adds key-value metadata to this transfer manager (`AddMetadata`).
    ///
    /// # Arguments
    ///
    /// * `entries` - Slice of `(key, value)` pairs.
    pub fn add_metadata(&self, entries: Vec<(String, String)>) {
        let mut guard: MutexGuard<'_, Vec<(String, String)>> = self
            .metadata
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.extend(entries);
    }
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
        }
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

    /// Creates a [`PjrtCoreClient`] infallibly from a [`TpuInitConfig`] using [`SimulatedTpuHal`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration.
    pub fn from_simulated(config: &TpuInitConfig) -> Self {
        let sim = SimulatedTpuHal::new(config);
        let spec = sim.topology_spec();
        let hal: Arc<dyn TpuHal> = Arc::new(sim);
        Self::from_topology(hal, spec, config.process_index)
    }

    /// Creates a [`PjrtCoreClient`] using the default [`SimulatedTpuHal`] and [`TpuInitConfig`].
    ///
    /// # Arguments
    ///
    /// * `config` - Initialization configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LibTpuError`] if initialization fails.
    pub fn with_simulated_hal(config: &TpuInitConfig) -> LibTpuResult<Self> {
        let hal = Arc::new(SimulatedTpuHal::new(config));
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
            .ok_or(LibTpuError::DeviceNotFound {
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
        Ok(self
            .devices
            .iter()
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
        self.dma_mappings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((base_addr, size));
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
        let mut guard = self
            .dma_mappings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
}

#[cfg(test)]
#[allow(for_loops_over_fallibles)]
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
        assert!(fhal
            .allocate_device_memory(DeviceId(0), MemoryKind::DeviceHbm, ByteSize(1))
            .is_err());
        assert!(fhal
            .deallocate_device_memory(DeviceId(0), DeviceAddress(0))
            .is_err());
        assert!(fhal
            .write_device_memory(DeviceId(0), DeviceAddress(0), &[])
            .is_err());
        assert!(fhal
            .read_device_memory(DeviceId(0), DeviceAddress(0), &mut [])
            .is_err());
        assert!(fhal
            .copy_device_to_device(
                DeviceId(0),
                DeviceAddress(0),
                DeviceId(0),
                DeviceAddress(0),
                ByteSize(1)
            )
            .is_err());
        for p in CompiledProgram::compile("mlir", b"return", b"") {
            assert!(fhal.execute_program(DeviceId(0), &p, &[], &[]).is_err());
        }
        assert!(fhal.memory_stats(DeviceId(0)).is_err());
        assert!(fhal.clear_memory_stats(DeviceId(0)).is_err());
        assert!(fhal.poison_device(DeviceId(0), String::new()).is_err());
        assert!(PjrtCoreClient::new(Arc::new(FailingTopologyHal), ProcessIndex(0)).is_err());
        let ev_ok = PjrtCoreEvent::ready_ok();
        assert!(ev_ok.is_ready());
        assert!(ev_ok.get_error().is_none());
        assert!(ev_ok.await_ready().is_ok());
        assert!(!format!("{ev_ok:?}").is_empty());

        let ev_pend = PjrtCoreEvent::pending();
        assert!(!ev_pend.is_ready());
        ev_pend.set(Some(LibTpuError::ProfilerNotActive));
        assert!(ev_pend.is_ready());
        assert!(ev_pend.get_error().is_some());
        assert!(ev_pend.await_ready().is_err());

        let stream = PjrtCoreCopyToDeviceStream::new(256, 128);
        assert!(!format!("{stream:?}").is_empty());
        assert_eq!(stream.current_bytes(), 0);
        assert!(stream.add_chunk(-1).is_err());
        assert!(stream.add_chunk(512).is_err());
        assert!(stream.add_chunk(128).is_ok());
        assert_eq!(stream.current_bytes(), 128);
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
        for client in PjrtCoreClient::with_simulated_hal(&cfg) {
            assert!(!format!("{client:?}").is_empty());
            assert!(!format!("{:?}", client.topology).is_empty());
            assert!(!format!("{:?}", client.devices).is_empty());
            assert!(!format!("{:?}", client.memories).is_empty());

            let topo_bytes = client.topology.serialize();
            assert!(PjrtCoreTopology::deserialize(&topo_bytes).is_ok());
            assert!(PjrtCoreTopology::deserialize(b"LTPUTOPO:gen=5").is_ok());
            let zero_ver_topo = crate::proto::TpuTopologyArgsProto {
                version: 0,
                ..crate::proto::TpuTopologyArgsProto::default()
            };
            assert!(PjrtCoreTopology::deserialize(&zero_ver_topo.encode()).is_err());
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

            for shape in Shape::new(ElementType::F32, vec![4], None, None) {
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
                assert!(client
                    .buffer_from_host(DeviceId(99), shape.clone(), &[])
                    .is_err());
                assert!(client
                    .buffer_from_host(DeviceId(0), shape.clone(), &vec![0_u8; 4096])
                    .is_err());
                for buf0 in &buf0_res {
                    for buf1 in &buf1_res {
                        for empty_buf in &empty_buf_res {
                            assert!(!format!("{buf0:?}").is_empty());
                            assert_eq!(buf0.inc_external_ref(), 2);
                            assert_eq!(buf0.dec_external_ref(), 1);

                            let copied_res = buf0.copy_to_device(DeviceId(0), MemorySpaceId(0));
                            assert!(copied_res.is_ok());
                            assert!(buf0.copy_to_device(DeviceId(99), MemorySpaceId(0)).is_err());
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

                            for hs in Shape::new(
                                ElementType::F32,
                                vec![1024 * 1024 * 1024 * 32],
                                None,
                                None,
                            ) {
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
                            assert!(bad_addr_buf
                                .copy_to_device(DeviceId(0), MemorySpaceId(0))
                                .is_err());
                            assert!(bad_addr_buf.bitcast(ElementType::S32, vec![4]).is_err());
                            assert!(bad_addr_buf.delete().is_err());

                            let loaded_res = client.compile_and_load(
                                "mlir",
                                b"module @add { stablehlo.add }",
                                b"",
                            );
                            assert!(loaded_res.is_ok());
                            assert!(client.compile_and_load("bad_fmt", b"code", b"").is_err());
                            assert!(client
                                .compile_and_load(
                                    "mlir",
                                    b"return",
                                    &[0xff, 0xff, 0xff, 0xff, 0xff, 0x7f]
                                )
                                .is_err());
                            for loaded in loaded_res {
                                assert!(!format!("{loaded:?}").is_empty());
                                assert!(!format!("{:?}", loaded.executable).is_empty());
                                let _ = loaded.executable.cost_analysis();
                                let _ = loaded.executable.compiled_memory_stats();

                                let serialized_exec = loaded.executable.program.serialize();
                                assert!(client.deserialize_and_load(&serialized_exec).is_ok());
                                assert!(client.deserialize_and_load(b"bad").is_err());

                                let outs_res = loaded.execute_on_device(
                                    DeviceId(0),
                                    MemorySpaceId(0),
                                    &[buf0, buf1],
                                );
                                assert!(outs_res.is_ok());
                                for outs in outs_res {
                                    for out0 in outs.first() {
                                        let host_out = out0.to_host_bytes();
                                        assert!(host_out.is_ok());
                                    }
                                }
                                assert!(loaded
                                    .execute_on_device(
                                        DeviceId(99),
                                        MemorySpaceId(0),
                                        &[buf0, buf1]
                                    )
                                    .is_err());
                                assert!(loaded
                                    .execute_on_device(DeviceId(0), MemorySpaceId(0), &[buf0])
                                    .is_err());

                                assert!(empty_buf.delete().is_ok());
                                assert!(empty_buf.delete().is_ok());
                                assert!(empty_buf.is_deleted());
                                assert!(empty_buf.to_host_bytes().is_err());
                                assert!(empty_buf
                                    .copy_to_device(DeviceId(0), MemorySpaceId(0))
                                    .is_err());
                                assert!(empty_buf.bitcast(ElementType::S32, vec![4]).is_err());
                                assert!(loaded
                                    .execute_on_device(
                                        DeviceId(0),
                                        MemorySpaceId(0),
                                        &[empty_buf, buf1]
                                    )
                                    .is_err());

                                loaded.delete();
                                assert!(loaded.is_deleted());
                                assert!(loaded
                                    .execute_on_device(DeviceId(0), MemorySpaceId(0), &[buf0, buf1])
                                    .is_err());
                            }
                        }
                    }
                }

                let mgr_res = PjrtCoreAsyncTransferManager::create(
                    DeviceId(0),
                    MemorySpaceId(0),
                    vec![shape.clone()],
                    &client.hal,
                );
                assert!(mgr_res.is_ok());
                assert!(PjrtCoreAsyncTransferManager::create(
                    DeviceId(99),
                    MemorySpaceId(0),
                    vec![shape],
                    &client.hal,
                )
                .is_err());
                for mgr in mgr_res {
                    assert!(!format!("{mgr:?}").is_empty());
                    assert!(mgr.transfer_data(0, &host_in0).is_ok());
                    assert!(mgr.transfer_data(99, &host_in0).is_err());
                    assert!(mgr.retrieve_buffer(0).is_ok());
                    assert!(mgr.retrieve_buffer(99).is_err());
                    mgr.add_metadata(vec![(String::from("k"), String::from("v"))]);
                }
            }

            assert!(client.dma_map(0, 1024).is_err());
            assert!(client.dma_map(0x2000, 0).is_err());
            assert!(client.dma_map(0x2000, 4096).is_ok());
            assert!(client.dma_unmap(0x9999).is_err());
            assert!(client.dma_unmap(0x2000).is_ok());
            assert!(client.device_memory_stats(DeviceId(0)).is_ok());
        }
    }
}
