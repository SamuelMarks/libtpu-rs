"""Drop-in `libtpu.sdk.debug` module binding `LibtpuSdk_RegisterDebugCallback`."""

from __future__ import annotations

import ctypes
from collections.abc import Callable

from libtpu.sdk.tpumonitoring import SdkClient, _check_sdk_error


class _RegisterDebugCallbackArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_RegisterDebugCallback_Args`."""

    _fields_ = [
        ("client", ctypes.c_void_p),
        ("callback", ctypes.c_void_p),
        ("user_data", ctypes.c_void_p),
        ("callback_id", ctypes.c_int64),
    ]


class _UnregisterDebugCallbackArgs(ctypes.Structure):
    """C ABI structure matching `LibtpuSdk_UnregisterDebugCallback_Args`."""

    _fields_ = [
        ("client", ctypes.c_void_p),
        ("callback_id", ctypes.c_int64),
    ]


_CDebugCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_ACTIVE_CALLBACKS: dict[int, object] = {}
_SHARED_CLIENT: SdkClient | None = None


def _get_debug_client() -> SdkClient:
    """Returns or initializes the shared `SdkClient` for debug callback registration.

    Returns:
        Singleton `SdkClient` instance.
    """
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = SdkClient()
        _SHARED_CLIENT._cdll.LibtpuSdk_RegisterDebugCallback.argtypes = [
            ctypes.POINTER(_RegisterDebugCallbackArgs)
        ]
        _SHARED_CLIENT._cdll.LibtpuSdk_RegisterDebugCallback.restype = ctypes.c_void_p
        _SHARED_CLIENT._cdll.LibtpuSdk_UnregisterDebugCallback.argtypes = [
            ctypes.POINTER(_UnregisterDebugCallbackArgs)
        ]
        _SHARED_CLIENT._cdll.LibtpuSdk_UnregisterDebugCallback.restype = ctypes.c_void_p
        _SHARED_CLIENT._cdll.LibtpuSdk_TriggerDebugCallbacks.argtypes = [ctypes.c_void_p]
        _SHARED_CLIENT._cdll.LibtpuSdk_TriggerDebugCallbacks.restype = ctypes.c_size_t
    return _SHARED_CLIENT


def register_debug_callback(cb: Callable[[], None]) -> int:
    """Registers a Python debug callback with the TPU runtime via `LibtpuSdk_RegisterDebugCallback`.

    Args:
        cb: Zero-argument Python callable invoked when a debug event triggers.

    Returns:
        Assigned 64-bit integer callback identifier (`callback_id`).
    """
    client = _get_debug_client()

    @_CDebugCallback
    def _trampoline(_user_data: int | None) -> None:
        """C FFI callback trampoline invoking the registered Python callable.

        Args:
            _user_data: Opaque user data pointer passed by the C runtime.
        """
        cb()

    fn_addr = ctypes.cast(_trampoline, ctypes.c_void_p).value or 0
    args = _RegisterDebugCallbackArgs(client._handle, fn_addr, 0, 0)
    err = client._cdll.LibtpuSdk_RegisterDebugCallback(ctypes.byref(args))
    _check_sdk_error(client._cdll, err)
    cb_id = int(args.callback_id)
    _ACTIVE_CALLBACKS[cb_id] = _trampoline
    return cb_id


def unregister_debug_callback(callback_id: int) -> None:
    """Unregisters a previously registered debug callback via `LibtpuSdk_UnregisterDebugCallback`.

    Args:
        callback_id: Numeric callback identifier returned by `register_debug_callback()`.
    """
    client = _get_debug_client()
    args = _UnregisterDebugCallbackArgs(client._handle, callback_id)
    err = client._cdll.LibtpuSdk_UnregisterDebugCallback(ctypes.byref(args))
    _check_sdk_error(client._cdll, err)
    _ACTIVE_CALLBACKS.pop(callback_id, None)


def _trigger_debug_callback_for_test() -> None:
    """Invokes all currently registered debug callbacks via `LibtpuSdk_TriggerDebugCallbacks`."""
    client = _get_debug_client()
    client._cdll.LibtpuSdk_TriggerDebugCallbacks(client._handle)


register_callback = register_debug_callback
unregister_callback = unregister_debug_callback


def trigger_fault_injection() -> None:
    """Triggers registered diagnostic debug callbacks (`_trigger_debug_callback_for_test`)."""
    _trigger_debug_callback_for_test()
