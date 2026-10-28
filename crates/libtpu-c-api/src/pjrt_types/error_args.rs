//! `#[repr(C)]` argument structures for error, plugin, event, and context entrypoints.
//!
//! Covers `PJRT_Error_*`, `PJRT_Plugin_*`, `PJRT_Event_*`, `PJRT_AsyncTrackingEvent_*`, and `PJRT_ExecuteContext_*`.

use crate::pjrt_types::{
    PJRT_AsyncTrackingEvent, PJRT_Error, PJRT_Event, PJRT_ExecuteContext, PJRT_Extension_Base,
    PJRT_NamedValue, PjrtArgsStruct,
};
use std::os::raw::c_char;

/// Argument structure for `PJRT_Error_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Error_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle to destroy.
    pub error: *mut PJRT_Error,
}

/// Argument structure for `PJRT_Error_Message` (`40` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Error_Message_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle to inspect.
    pub error: *const PJRT_Error,
    /// Output pointer to UTF-8 message bytes.
    pub message: *const c_char,
    /// Output length of `message` in bytes.
    pub message_size: usize,
}

/// Argument structure for `PJRT_Error_GetCode` (`32` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Error_GetCode_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle to inspect.
    pub error: *const PJRT_Error,
    /// Output `PJRT_Error_Code` integer.
    pub code: i32,
}

/// Argument structure for `PJRT_Error_ForEachPayload` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Error_ForEachPayload_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Error handle.
    pub error: *const PJRT_Error,
    /// Payload visitor callback pointer.
    pub visitor: Option<
        unsafe extern "C" fn(*const c_char, usize, *const c_char, usize, *mut std::ffi::c_void),
    >,
    /// User argument pointer.
    pub user_arg: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Plugin_Initialize` (`16` bytes).
#[repr(C)]
#[derive(Debug)]
pub struct PJRT_Plugin_Initialize_Args {
    /// Size of this struct (`16`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
}

/// Argument structure for `PJRT_Plugin_Attributes` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Plugin_Attributes_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output pointer to `PJRT_NamedValue` array.
    pub attributes: *const PJRT_NamedValue,
    /// Output number of attributes.
    pub num_attributes: usize,
}

/// Argument structure for `PJRT_Event_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle to destroy.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_IsReady` (`32` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_IsReady_Args {
    /// Size of this struct (`32`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
    /// Output readiness flag (`bool`).
    pub is_ready: bool,
    /// Explicit padding.
    pub pad: [u8; 7],
}

/// Argument structure for `PJRT_Event_Error` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Error_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_Await` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Await_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_OnReady` (`40` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_OnReady_Args {
    /// Size of this struct (`40`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
    /// Callback function pointer.
    pub callback: Option<unsafe extern "C" fn(*mut PJRT_Error, *mut std::ffi::c_void)>,
    /// User argument pointer.
    pub user_arg: *mut std::ffi::c_void,
}

/// Argument structure for `PJRT_Event_Create` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Create_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output created `PJRT_Event*`.
    pub event: *mut PJRT_Event,
}

/// Argument structure for `PJRT_Event_Set` (`48` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Event_Set_Args {
    /// Size of this struct (`48`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Event handle.
    pub event: *mut PJRT_Event,
    /// Status code (`i32`).
    pub error_code: i32,
    /// Explicit padding.
    pub pad: i32,
    /// Error message pointer.
    pub error_message: *const c_char,
    /// Error message byte length.
    pub error_message_size: usize,
}

/// Argument structure for `PJRT_AsyncTrackingEvent_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_AsyncTrackingEvent_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Async tracking event handle to destroy.
    pub event: *mut PJRT_AsyncTrackingEvent,
}

/// Argument structure for `PJRT_ExecuteContext_Create` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecuteContext_Create_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Output `PJRT_ExecuteContext*`.
    pub context: *mut PJRT_ExecuteContext,
}

/// Argument structure for `PJRT_ExecuteContext_Destroy` (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_ExecuteContext_Destroy_Args {
    /// Size of this struct (`24`).
    pub struct_size: usize,
    /// Extension pointer (`NULL`).
    pub extension_start: *mut PJRT_Extension_Base,
    /// Context handle to destroy.
    pub context: *mut PJRT_ExecuteContext,
}

impl PjrtArgsStruct for PJRT_Error_Destroy_Args {}
impl PjrtArgsStruct for PJRT_Error_Message_Args {}
impl PjrtArgsStruct for PJRT_Error_GetCode_Args {
    const MIN_STRUCT_SIZE: usize = 28;
}
impl PjrtArgsStruct for PJRT_Error_ForEachPayload_Args {}
impl PjrtArgsStruct for PJRT_Plugin_Initialize_Args {}
impl PjrtArgsStruct for PJRT_Plugin_Attributes_Args {}
impl PjrtArgsStruct for PJRT_Event_Destroy_Args {}
impl PjrtArgsStruct for PJRT_Event_IsReady_Args {
    const MIN_STRUCT_SIZE: usize = 25;
}
impl PjrtArgsStruct for PJRT_Event_Error_Args {}
impl PjrtArgsStruct for PJRT_Event_Await_Args {}
impl PjrtArgsStruct for PJRT_Event_OnReady_Args {}
impl PjrtArgsStruct for PJRT_Event_Create_Args {}
impl PjrtArgsStruct for PJRT_Event_Set_Args {}
impl PjrtArgsStruct for PJRT_AsyncTrackingEvent_Destroy_Args {}
impl PjrtArgsStruct for PJRT_ExecuteContext_Create_Args {}
impl PjrtArgsStruct for PJRT_ExecuteContext_Destroy_Args {}

const _: () = assert!(size_of::<PJRT_Error_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Error_Message_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Error_GetCode_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Error_ForEachPayload_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Plugin_Initialize_Args>() == 16);
const _: () = assert!(size_of::<PJRT_Plugin_Attributes_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Event_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Event_IsReady_Args>() == 32);
const _: () = assert!(size_of::<PJRT_Event_Error_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Event_Await_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Event_OnReady_Args>() == 40);
const _: () = assert!(size_of::<PJRT_Event_Create_Args>() == 24);
const _: () = assert!(size_of::<PJRT_Event_Set_Args>() == 48);
const _: () = assert!(size_of::<PJRT_AsyncTrackingEvent_Destroy_Args>() == 24);
const _: () = assert!(size_of::<PJRT_ExecuteContext_Create_Args>() == 24);
const _: () = assert!(size_of::<PJRT_ExecuteContext_Destroy_Args>() == 24);
