"""What this machine can actually run.

The number that matters is not the size of the weights but the memory available
for weights *plus* KV cache. Ollama reserves the model's maximum context unless
told otherwise, which is how a 2.5 GB model came to ask for 42 GB.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

GB = 1024**3


@dataclass
class Hardware:
    system: str  # Darwin | Linux | Windows
    apple_silicon: bool
    total_ram_gb: float
    vram_gb: float | None  # discrete GPU memory, None if unknown/absent
    gpu_name: str | None

    @property
    def usable_gb(self) -> float:
        """Memory a model can realistically occupy.

        Apple Silicon shares one pool, so a large fraction of RAM is available to
        the GPU. A discrete card is limited to its own VRAM: spilling into system
        memory is far slower than choosing a smaller model that fits.
        """
        if self.apple_silicon:
            return self.total_ram_gb * 0.75
        if self.vram_gb:
            # A desktop, compositor and other processes hold some VRAM; a model
            # sized to the full figure spills into system memory and crawls.
            return max(0.0, self.vram_gb - 1.0)
        return self.total_ram_gb * 0.5  # CPU inference, leave room for the OS

    def describe(self) -> str:
        if self.apple_silicon:
            return f"Apple Silicon, {self.total_ram_gb:.0f} GB unified memory"
        if self.vram_gb:
            return f"{self.gpu_name or 'GPU'}, {self.vram_gb:.0f} GB VRAM ({self.total_ram_gb:.0f} GB RAM)"
        return f"{self.system}, {self.total_ram_gb:.0f} GB RAM, no GPU detected (CPU inference)"


def _run(command: list[str]) -> str:
    if not shutil.which(command[0]):
        return ""
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform dependent
        return ""


def total_ram_gb() -> float:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"]).strip()
        if out.isdigit():
            return int(out) / GB
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024 / GB
        except OSError:  # pragma: no cover - platform dependent
            pass
    elif system == "Windows":  # pragma: no cover - platform dependent
        out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory"])
        digits = re.findall(r"\d{6,}", out)
        if digits:
            return int(digits[0]) / GB
    return 0.0


def detect_gpu() -> tuple[float | None, str | None]:
    """Discrete GPU memory in GB and its name, when detectable."""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if out.strip():
        first = out.strip().splitlines()[0]
        name, _, memory = first.partition(",")
        found = re.search(r"(\d+)", memory)
        if found:
            return int(found.group(1)) / 1024, name.strip()
    # AMD ROCm
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    found = re.search(r"(\d{6,})", out)
    if found:
        return int(found.group(1)) / GB, "AMD GPU"
    return None, None


def detect() -> Hardware:
    system = platform.system()
    apple_silicon = system == "Darwin" and platform.machine() == "arm64"
    vram, gpu_name = (None, None) if apple_silicon else detect_gpu()
    return Hardware(
        system=system,
        apple_silicon=apple_silicon,
        total_ram_gb=total_ram_gb(),
        vram_gb=vram,
        gpu_name=gpu_name,
    )
