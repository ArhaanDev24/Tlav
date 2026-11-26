from tlav import hardware

MEMINFO = """MemTotal:       16384000 kB
MemFree:         1000000 kB
MemAvailable:    8192000 kB
"""

CPUINFO = """processor\t: 0
model name\t: Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse sse2 avx avx2 fma f16c
processor\t: 1
physical id\t: 0
core id\t\t: 1
processor\t: 2
physical id\t: 0
core id\t\t: 0
"""


def test_parse_meminfo():
    total, avail = hardware.parse_meminfo(MEMINFO)
    assert round(total, 1) == 15.6
    assert round(avail, 1) == 7.8


def test_parse_cpuinfo_counts_physical_cores():
    model, flags, cores = hardware.parse_cpuinfo(CPUINFO)
    assert "i5-4570" in model
    assert "avx2" in flags
    assert cores == 2


def test_parse_mac_features_haswell():
    flags = hardware.parse_mac_features("FPU SSE SSE2 AVX1.0 FMA F16C AVX2 BMI1")
    assert "avx" in flags and "avx2" in flags


def test_simd_priority():
    hw = hardware.simulated(16, simd="avx2")
    assert hw.simd == "avx2"
    assert hardware.simulated(16, simd="baseline").simd == "baseline"


def test_detect_runs_here():
    hw = hardware.detect()
    assert hw.ram_total_gb > 0
    assert hw.physical_cores >= 1
    assert hardware.from_dict(hw.to_dict()) == hw
