import ctypes
import sys
from contextlib import contextmanager
from ctypes import wintypes


class _ReasonDetail(ctypes.Structure):
    _fields_ = [
        ("LocalizedReasonModule", wintypes.HMODULE),
        ("LocalizedReasonId", wintypes.ULONG),
        ("ReasonStringCount", wintypes.ULONG),
        ("ReasonStrings", ctypes.POINTER(wintypes.LPWSTR)),
    ]


class _Reason(ctypes.Union):
    _fields_ = [
        ("Detailed", _ReasonDetail),
        ("SimpleReasonString", wintypes.LPWSTR),
    ]


class _ReasonContext(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("Flags", wintypes.DWORD),
        ("Reason", _Reason),
    ]


_SYSTEM_REQUIRED = 1
_EXECUTION_REQUIRED = 3


@contextmanager
def stay_awake():
    """
    Keep this process running through Modern Standby until the run ends.
    The screen can still turn off.
    """

    if sys.platform != "win32":
        yield
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(_ReasonContext)]
    kernel32.PowerCreateRequest.restype = wintypes.HANDLE
    kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerSetRequest.restype = wintypes.BOOL
    kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerClearRequest.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    context = _ReasonContext()
    context.Version = 0
    context.Flags = 0x1
    context.Reason.SimpleReasonString = "DS9000 training"
    handle = kernel32.PowerCreateRequest(ctypes.byref(context))
    if not handle:
        print(f">> stay awake failed: {ctypes.WinError(ctypes.get_last_error())}")
        yield
        return

    held = []
    try:
        for kind in (_SYSTEM_REQUIRED, _EXECUTION_REQUIRED):
            if kernel32.PowerSetRequest(handle, kind):
                held.append(kind)
            else:
                print(
                    f">> stay awake failed: {ctypes.WinError(ctypes.get_last_error())}"
                )
        yield
    finally:
        for kind in held:
            kernel32.PowerClearRequest(handle, kind)
        kernel32.CloseHandle(handle)
