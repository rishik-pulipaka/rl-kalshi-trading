"""Process-level health: memory and uptime.

Small on purpose. The system is meant to run unattended for weeks on the user's
own machine (PRD 15), so "how much RAM is this thing using right now" is a
number both the dashboard and the diagnostic tools need, and neither should
depend on a third-party package to get it.

Windows-only via `K32GetProcessMemoryInfo`, with a graceful fallback everywhere
else. Note the explicit `argtypes`: without them ctypes guesses the pointer
width and the call silently returns a zeroed struct rather than failing, which
reads as "0 MB used" and looks like a working measurement.
"""

import os
import time

_STARTED_AT = time.time()

_pmc_reader = None


def _make_windows_reader():
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    fn = kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE,
                   ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                   wintypes.DWORD]
    fn.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()

    def read():
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not fn(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize, counters.PeakWorkingSetSize

    return read


def _reader():
    global _pmc_reader
    if _pmc_reader is None:
        try:
            _pmc_reader = _make_windows_reader()
        except Exception:
            _pmc_reader = False
    return _pmc_reader or None


def rss_mb():
    """Resident set size in MB, or None if it can't be read."""
    read = _reader()
    if read is None:
        return None
    result = read()
    return result[0] / (1024 * 1024) if result else None


def peak_rss_mb():
    read = _reader()
    if read is None:
        return None
    result = read()
    return result[1] / (1024 * 1024) if result else None


def uptime_s():
    return time.time() - _STARTED_AT


def disk_free_gb(path="."):
    """Free space on the volume holding `path`. PRD 15: stay mindful of disk."""
    try:
        usage = os.statvfs(path)          # POSIX
        return usage.f_bavail * usage.f_frsize / 1e9
    except AttributeError:
        import shutil
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return None


def snapshot():
    """Everything the dashboard's health panel needs, in one call."""
    return {
        "rss_mb": rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "uptime_s": uptime_s(),
        "disk_free_gb": disk_free_gb(),
        "pid": os.getpid(),
    }
