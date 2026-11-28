"""Hardware detection with only the Python standard library.

Works on Linux, macOS (including old Intel Macs) and Windows. Every probe is
best-effort: if something cannot be detected we fall back to safe values
instead of crashing.
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import List, Optional

GB = 1024 ** 3


@dataclass
class Hardware:
    os_name: str
    os_version: str
    arch: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    cpu_flags: List[str] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpu: str = "none"            # "cuda", "metal", or "none" (usable acceleration only)
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0

    @property
    def simd(self) -> str:
        """Best SIMD level the CPU offers; drives llama.cpp build flags."""
        for f in ("avx512f", "avx2", "avx", "neon"):
            if f in self.cpu_flags:
                return f
        return "baseline"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["simd"] = self.simd
        return d

    def summary(self) -> str:
        accel = {"cuda": "NVIDIA CUDA", "metal": "Apple Metal", "none": "CPU only"}[self.gpu]
        lines = [
            f"OS:        {self.os_name} {self.os_version} ({self.arch})",
            f"CPU:       {self.cpu_model}",
            f"Cores:     {self.physical_cores} physical / {self.logical_cores} logical",
            f"SIMD:      {self.simd.upper()}",
            f"RAM:       {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB available",
            f"Disk free: {self.disk_free_gb:.1f} GB",
            f"GPU:       {self.gpu_name or 'n/a'} -> {accel}",
        ]
        if self.gpu_vram_gb:
            lines.append(f"VRAM:      {self.gpu_vram_gb:.1f} GB")
        return "\n".join(lines)


def _run(cmd: List[str], timeout: int = 10) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------- parsers
# Kept as pure functions so they can be unit-tested with captured output.

def parse_meminfo(text: str) -> tuple:
    vals = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) * 1024
    total = vals.get("MemTotal", 0)
    avail = vals.get("MemAvailable", vals.get("MemFree", 0) + vals.get("Cached", 0))
    return total / GB, avail / GB


def parse_cpuinfo(text: str) -> tuple:
    model, flags, cores = "", set(), set()
    phys_id = "0"
    for line in text.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key in ("model name", "Model", "Hardware") and not model:
            model = val
        elif key in ("flags", "Features"):
            flags.update(val.lower().split())
        elif key == "physical id":
            phys_id = val
        elif key == "core id":
            cores.add((phys_id, val))
    if "asimd" in flags:
        flags.add("neon")
    return model, sorted(flags), len(cores)


def parse_mac_features(text: str) -> List[str]:
    """`sysctl machdep.cpu.features machdep.cpu.leaf7_features` -> flag list."""
    flags = {f.lower().replace(".", "_") for f in text.split()}
    # sysctl names AVX as AVX1.0
    if "avx1_0" in flags:
        flags.add("avx")
    return sorted(flags)


# ---------------------------------------------------------------- probes

def _ram() -> tuple:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                return parse_meminfo(f.read())
        except OSError:
            pass
    elif system == "Darwin":
        total = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
        vm = _run(["vm_stat"])
        page = 4096
        m = re.search(r"page size of (\d+)", vm)
        if m:
            page = int(m.group(1))
        free = 0
        for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            m = re.search(k + r":\s+(\d+)", vm)
            if m:
                free += int(m.group(1)) * page
        return total / GB, (free or total * 0.5) / GB
    elif system == "Windows":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return st.ullTotalPhys / GB, st.ullAvailPhys / GB
    return 4.0, 2.0  # conservative guess


def _cpu() -> tuple:
    system = platform.system()
    logical = os.cpu_count() or 1
    model, flags, physical = platform.processor() or platform.machine(), [], 0
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                m, flags, physical = parse_cpuinfo(f.read())
                model = m or model
        except OSError:
            pass
    elif system == "Darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or model
        physical = int(_run(["sysctl", "-n", "hw.physicalcpu"]) or 0)
        flags = parse_mac_features(_run(["sysctl", "-n", "machdep.cpu.features"]) + " "
                                   + _run(["sysctl", "-n", "machdep.cpu.leaf7_features"]))
        if platform.machine() == "arm64":
            flags = sorted(set(flags) | {"neon"})
    elif system == "Windows":
        model = _run(["powershell", "-NoProfile", "-Command",
                      "(Get-CimInstance Win32_Processor).Name"]) or model
        cores = _run(["powershell", "-NoProfile", "-Command",
                      "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"])
        physical = int(cores) if cores.isdigit() else 0
        try:
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40, PF_AVX_INSTRUCTIONS_AVAILABLE = 39 (Win10+)
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if k32.IsProcessorFeaturePresent(39):
                flags.append("avx")
            if k32.IsProcessorFeaturePresent(40):
                flags.append("avx2")
            if k32.IsProcessorFeaturePresent(41):
                flags.append("avx512f")
        except Exception:
            pass
    if not physical:
        physical = max(1, logical // 2) if logical > 2 else logical
    return model.strip(), physical, logical, flags


def _gpu(system: str, arch: str) -> tuple:
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if out:
            name, _, mem = out.splitlines()[0].partition(",")
            try:
                return "cuda", name.strip(), float(mem) / 1024
            except ValueError:
                return "cuda", name.strip(), 0.0
    if system == "Darwin":
        info = _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
        m = re.search(r"Chipset Model:\s*(.+)", info)
        name = m.group(1).strip() if m else ""
        # Metal in llama.cpp is only worthwhile on Apple Silicon. Old Intel Mac
        # GPUs (GeForce 7xxM, Iris, Radeon) are slower than the CPU path or crash.
        if arch == "arm64":
            return "metal", name or "Apple GPU", 0.0
        return "none", name, 0.0
    return "none", "", 0.0


def detect() -> Hardware:
    system = platform.system()
    arch = platform.machine().lower()
    if arch in ("amd64", "x86_64"):
        arch = "x86_64"
    elif arch in ("aarch64", "arm64"):
        arch = "arm64"
    if system == "Darwin":
        version = platform.mac_ver()[0]
    elif system == "Windows":
        version = platform.version()
    else:
        version = platform.release()
    total, avail = _ram()
    model, physical, logical, flags = _cpu()
    gpu, gpu_name, vram = _gpu(system, arch)
    try:
        from . import config
        disk = shutil.disk_usage(str(config.home())).free / GB
    except OSError:
        disk = 0.0
    return Hardware(
        os_name={"Darwin": "macOS"}.get(system, system), os_version=version, arch=arch,
        cpu_model=model, physical_cores=physical, logical_cores=logical, cpu_flags=flags,
        ram_total_gb=round(total, 2), ram_available_gb=round(avail, 2), disk_free_gb=round(disk, 1),
        gpu=gpu, gpu_name=gpu_name, gpu_vram_gb=round(vram, 1),
    )


def simulated(ram_gb: float, cores: int = 4, simd: str = "avx2", gpu: str = "none",
              os_name: str = "macOS", vram_gb: float = 0.0, disk_gb: float = 200.0) -> Hardware:
    """A fake machine profile, used by `tlav recommend --ram ...` and tests."""
    flags = {"avx512f": ["avx", "avx2", "avx512f"], "avx2": ["avx", "avx2"], "avx": ["avx"],
             "neon": ["neon"]}.get(simd, [])
    return Hardware(os_name=os_name, os_version="simulated", arch="arm64" if simd == "neon" else "x86_64",
                    cpu_model=f"Simulated {cores}-core CPU", physical_cores=cores, logical_cores=cores * 2,
                    cpu_flags=flags, ram_total_gb=ram_gb, ram_available_gb=ram_gb * 0.7,
                    disk_free_gb=disk_gb, gpu=gpu, gpu_vram_gb=vram_gb)


def from_dict(d: Optional[dict]) -> Optional[Hardware]:
    """A Hardware from what a worker sent. Tolerates other Tlav versions: unknown fields are
    ignored and missing ones get safe defaults (a cluster can mix versions)."""
    if not d:
        return None
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(Hardware)}
    defaults = {"os_name": "Linux", "os_version": "", "arch": "x86_64", "cpu_model": "", "physical_cores": 2,
                "logical_cores": 4, "cpu_flags": [], "ram_total_gb": 8.0, "ram_available_gb": 5.0,
                "disk_free_gb": 0.0, "gpu": "none", "gpu_name": "", "gpu_vram_gb": 0.0}
    kw = {k: v for k, v in d.items() if k in fields}
    for name, f in fields.items():
        if name not in kw and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            kw[name] = defaults.get(name, "")
    try:
        return Hardware(**kw)
    except TypeError:
        return None
