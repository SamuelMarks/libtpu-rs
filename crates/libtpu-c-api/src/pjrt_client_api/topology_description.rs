//! `PJRT_TopologyDescription_*` C API entrypoints for TPU topology inspection and serialization.

use super::*;

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
unsafe fn handle_topology_create(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Create_Args,
) -> Result<(), LibTpuError> {
    args.topology = box_default_topology_description();
    Ok(())
}

/// Creates a standalone `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Create`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Create_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_Create(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Create_Args>(
            raw_args,
            handle_topology_create,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
unsafe fn handle_topology_destroy(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args,
) -> Result<(), LibTpuError> {
    if !args.topology.is_null() {
        // SAFETY: `args.topology` was allocated by `PJRT_TopologyDescription_Create` or `Deserialize`.
        unsafe {
            let topo = Box::from_raw(args.topology);
            for d_ptr in topo.desc_ptrs {
                drop(Box::from_raw(d_ptr));
            }
        }
    }
    Ok(())
}

/// Destroys a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Destroy`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_Destroy(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Destroy_Args>(
            raw_args,
            handle_topology_destroy,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
unsafe fn handle_topology_platform_name(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((c"tpu".as_ptr(), 3), |t| {
        (
            t.inner.spec.platform_name.as_ptr().cast::<c_char>(),
            t.inner.spec.platform_name.len(),
        )
    });
    args.platform_name = ptr;
    args.platform_name_size = len;
    Ok(())
}

/// Queries the platform name of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_PlatformName`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_PlatformName(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_PlatformName_Args>(
            raw_args,
            handle_topology_platform_name,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
unsafe fn handle_topology_platform_version(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((c"0.0.50".as_ptr(), 6), |t| {
        (
            t.inner.spec.platform_version.as_ptr().cast::<c_char>(),
            t.inner.spec.platform_version.len(),
        )
    });
    args.platform_version = ptr;
    args.platform_version_size = len;
    Ok(())
}

/// Queries the platform version of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_PlatformVersion`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_PlatformVersion(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_PlatformVersion_Args>(
            raw_args,
            handle_topology_platform_version,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
unsafe fn handle_topology_get_device_descriptions(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let (ptr, len) = unsafe { args.topology.as_ref() }.map_or((ptr::null(), 0), |t| {
        (t.desc_ptrs.as_ptr(), t.desc_ptrs.len())
    });
    args.descriptions = ptr;
    args.num_descriptions = len;
    Ok(())
}

/// Queries all device descriptions in a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_GetDeviceDescriptions`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_GetDeviceDescriptions(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_TopologyDescription_GetDeviceDescriptions_Args,
        >(raw_args, handle_topology_get_device_descriptions)
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
unsafe fn handle_topology_serialize(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let topo_bytes = unsafe { args.topology.as_ref() }.map_or_else(
        || {
            libtpu_core::proto::TpuTopologyArgsProto {
                platform_version: String::from("0.0.50"),
                mesh_bounds: vec![2, 2, 1],
                chips_per_host: 4,
                num_hosts: 1,
                version: 5,
                fingerprint: String::from("00001234"),
            }
            .encode()
        },
        |t| t.inner.serialize(),
    );
    let (ptr, len, holder) = crate::box_serialized_artifact(topo_bytes);
    args.serialized_bytes = ptr;
    args.serialized_bytes_size = len;
    args.serialized_topology = holder.cast();
    args.serialized_topology_deleter = Some(crate::pjrt_serialized_artifact_deleter);
    Ok(())
}

/// Serializes a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Serialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_Serialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Serialize_Args>(
            raw_args,
            handle_topology_serialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
unsafe fn handle_topology_attributes(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args,
) -> Result<(), LibTpuError> {
    args.attributes = GLOBAL_PLUGIN_ATTRIBUTES.as_ptr();
    args.num_attributes = GLOBAL_PLUGIN_ATTRIBUTES.len();
    Ok(())
}

/// Queries topology attributes (`PJRT_TopologyDescription_Attributes`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_Attributes(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Attributes_Args>(
            raw_args,
            handle_topology_attributes,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
unsafe fn handle_topology_deserialize(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args,
) -> Result<(), LibTpuError> {
    let bytes = if args.serialized_topology.is_null() {
        &[][..]
    } else {
        // SAFETY: `serialized_topology` points to `serialized_topology_size` bytes.
        unsafe {
            std::slice::from_raw_parts(
                args.serialized_topology.cast::<u8>(),
                args.serialized_topology_size,
            )
        }
    };
    args.topology = libtpu_core::pjrt::PjrtCoreTopology::deserialize(bytes).map_or_else(
        |_| box_default_topology_description(),
        crate::box_topology_description,
    );
    Ok(())
}

/// Deserializes a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Deserialize`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_Deserialize(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Deserialize_Args>(
            raw_args,
            handle_topology_deserialize,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
unsafe fn handle_topology_fingerprint(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let top_opt = unsafe { args.topology.as_ref() };
    args.fingerprint = top_opt.map_or(0x5450_555F_5635_4500, |t| t.inner.fingerprint);
    Ok(())
}

/// Queries the 64-bit fingerprint of a `PJRT_TopologyDescription` (`PJRT_TopologyDescription_Fingerprint`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_Fingerprint(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<crate::pjrt_types::PJRT_TopologyDescription_Fingerprint_Args>(
            raw_args,
            handle_topology_fingerprint,
        )
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
unsafe fn handle_topology_make_canonical_shape(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
) -> Result<(), LibTpuError> {
    // SAFETY: `topology` is null or a valid `PJRT_TopologyDescription*`.
    let _ = unsafe { args.topology.as_ref() }.map(|t| t.inner.fingerprint);
    let elem = ElementType::from_raw(args.element_type).unwrap_or(ElementType::F32);
    let dims = if args.dims.is_null() {
        vec![8, 128]
    } else {
        // SAFETY: `args.dims` points to `args.num_dims` `i64` elements.
        unsafe { std::slice::from_raw_parts(args.dims, args.num_dims).to_vec() }
    };
    let shape = Shape::new(elem, dims, None, None).unwrap_or_else(|_| Shape {
        element_type: ElementType::F32,
        dims: vec![8, 128],
        dynamic_dims: vec![false, false],
        layout: libtpu_core::types::MemoryLayout::default_tpu_tiled(2),
    });
    args.reserved = shape.on_device_byte_size().0;
    if !args.layout.is_null() {
        // SAFETY: `args.layout` is a valid mutable `PJRT_Buffer_MemoryLayout*`.
        unsafe {
            *args.layout = crate::pjrt_types::PJRT_Buffer_MemoryLayout {
                struct_size: size_of::<crate::pjrt_types::PJRT_Buffer_MemoryLayout>(),
                extension_start: ptr::null_mut(),
                minor_to_major: ptr::null(),
                minor_to_major_size: shape.dims.len(),
                tile_dims: ptr::null(),
                tile_dim_sizes: ptr::null(),
            };
        }
    }
    Ok(())
}

/// Computes canonical tiled device shape for a memory space (`PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args,
        >(raw_args, handle_topology_make_canonical_shape)
    }
}

/// Populates [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
///
/// # Arguments
///
/// * `args` - Validated [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`] structure.
///
/// # Safety
///
/// `args` must be a valid [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
unsafe fn handle_topology_get_memory_space_kind_ids(
    args: &mut crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
) -> Result<(), LibTpuError> {
    args.kind_ids = GLOBAL_MEMORY_KIND_IDS.as_ptr();
    args.num_kind_ids = GLOBAL_MEMORY_KIND_IDS.len();
    Ok(())
}

/// Queries supported memory space kind IDs of a topology (`PJRT_TopologyDescription_GetMemorySpaceKindIds`).
///
/// # Arguments
///
/// * `raw_args` - Pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
///
/// # Safety
///
/// `raw_args` must be null or a valid pointer to [`crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PJRT_TopologyDescription_GetMemorySpaceKindIds(
    raw_args: *mut std::ffi::c_void,
) -> *mut PJRT_Error {
    // SAFETY: Validated by `dispatch_typed_pjrt_args`.
    unsafe {
        dispatch_typed_pjrt_args::<
            crate::pjrt_types::PJRT_TopologyDescription_GetMemorySpaceKindIds_Args,
        >(raw_args, handle_topology_get_memory_space_kind_ids)
    }
}
