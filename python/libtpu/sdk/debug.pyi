"""Type stubs for `libtpu.sdk.debug`."""

from collections.abc import Callable

def register_debug_callback(cb: Callable[[], None]) -> int:
    """Registers a Python debug callback with `LibtpuSdk_RegisterDebugCallback`.

    Args:
        cb: Zero-argument callable invoked when a debug event triggers.
    """
    ...

def unregister_debug_callback(callback_id: int) -> None:
    """Unregisters a previously registered debug callback via `LibtpuSdk_UnregisterDebugCallback`.

    Args:
        callback_id: Unique callback identifier returned by `register_debug_callback`.
    """
    ...

def _trigger_debug_callback_for_test() -> None:
    """Invokes all currently registered debug callbacks for diagnostic testing."""
    ...
