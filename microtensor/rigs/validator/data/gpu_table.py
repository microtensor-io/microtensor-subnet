from __future__ import annotations

from dataclasses import dataclass

from microtensor.rigs.validator.data.tiers import Tier, tier_for_memory_gb

VRAM_FLOOR_RATIO = 0.90
VRAM_CEIL_RATIO = 1.05
MIB = 1024

GPU_VRAM_MB: dict[str, tuple[int, ...]] = {
    "NVIDIA B300 SXM6 AC": (294912,),
    "NVIDIA B200": (196608,),
    "NVIDIA H200": (144384,),
    "NVIDIA H200 NVL": (144384,),
    "NVIDIA H100 80GB HBM3": (81920,),
    "NVIDIA H100 NVL": (96256,),
    "NVIDIA H100 PCIe": (81920,),
    "NVIDIA H800 80GB HBM3": (81920,),
    "NVIDIA H800 NVL": (96256,),
    "NVIDIA H800 PCIe": (81920,),
    "NVIDIA A100 80GB PCIe": (81920,),
    "NVIDIA A100-SXM4-80GB": (81920,),
    "NVIDIA A100-SXM4-40GB": (40960,),
    "NVIDIA A100-PCIE-40GB": (40960,),
    "NVIDIA A800 80GB PCIe": (81920,),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (98304,),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": (98304,),
    "NVIDIA RTX PRO 6000D Blackwell Workstation Edition": (86016,),
    "NVIDIA RTX 6000D": (86016,),
    "NVIDIA RTX PRO 5000 Blackwell": (49152, 73728),
    "NVIDIA RTX PRO 4500 Blackwell": (32768,),
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": (32768,),
    "NVIDIA RTX PRO 4000 Blackwell": (24576,),
    "NVIDIA RTX PRO 2000 Blackwell": (16384,),
    "NVIDIA GeForce RTX 5090": (32768,),
    "NVIDIA GeForce RTX 5080": (16384,),
    "NVIDIA GeForce RTX 5070 Ti": (16384,),
    "NVIDIA GeForce RTX 5070": (12288,),
    "NVIDIA GeForce RTX 5060 Ti": (8192, 16384),
    "NVIDIA GeForce RTX 5060": (8192,),
    "NVIDIA GeForce RTX 4090": (24576,),
    "NVIDIA GeForce RTX 4090 D": (24576,),
    "NVIDIA GeForce RTX 4080 SUPER": (16384,),
    "NVIDIA GeForce RTX 4080": (16384,),
    "NVIDIA GeForce RTX 4070 Ti SUPER": (16384,),
    "NVIDIA GeForce RTX 4070 Ti": (12288,),
    "NVIDIA GeForce RTX 4070 SUPER": (12288,),
    "NVIDIA GeForce RTX 4070": (12288,),
    "NVIDIA GeForce RTX 4060 Ti": (8192, 16384),
    "NVIDIA GeForce RTX 4060": (8192,),
    "NVIDIA RTX 6000 Ada Generation": (49152,),
    "NVIDIA RTX 5880 Ada Generation": (49152,),
    "NVIDIA RTX 5000 Ada Generation": (32768,),
    "NVIDIA RTX 4500 Ada Generation": (24576,),
    "NVIDIA RTX 4000 Ada Generation": (20480,),
    "NVIDIA L40S": (49152,),
    "NVIDIA L40": (49152,),
    "NVIDIA L4": (24576,),
    "NVIDIA A10 Tensor Core GPU": (24576,),
    "NVIDIA A40": (49152,),
    "NVIDIA RTX A6000": (49152,),
    "NVIDIA RTX A5000": (24576,),
    "NVIDIA RTX A4500": (20480,),
    "NVIDIA RTX A4000": (16384,),
    "NVIDIA RTX A2000": (6144, 12288),
    "NVIDIA T4 Tensor Core GPU": (16384,),
    "NVIDIA Tesla V100 Tensor Core GPU": (16384, 32768),
    "NVIDIA Quadro RTX 8000": (49152,),
    "NVIDIA Quadro RTX 6000": (24576,),
    "NVIDIA Quadro RTX 5000": (16384,),
    "NVIDIA TITAN V": (12288,),
    "NVIDIA TITAN RTX": (24576,),
    "NVIDIA GeForce RTX 3090 Ti": (24576,),
    "NVIDIA GeForce RTX 3090": (24576,),
    "NVIDIA GeForce RTX 3080 Ti": (12288,),
    "NVIDIA GeForce RTX 3080": (10240, 12288),
    "NVIDIA GeForce RTX 3070 Ti": (8192,),
    "NVIDIA GeForce RTX 3070": (8192,),
    "NVIDIA GeForce RTX 3060 Ti": (8192,),
    "NVIDIA GeForce RTX 3060": (8192, 12288),
    "NVIDIA GeForce RTX 3050": (6144, 8192),
    "NVIDIA GeForce RTX 2080 Ti": (11264,),
    "NVIDIA GeForce RTX 2080 SUPER": (8192,),
    "NVIDIA GeForce RTX 2070 SUPER": (8192,),
    "NVIDIA GeForce RTX 2060 SUPER": (8192,),
    "NVIDIA GeForce RTX 2060": (6144, 12288),
    "NVIDIA GeForce GTX 1660 Ti": (6144,),
    "NVIDIA GeForce GTX 1660 SUPER": (6144,),
    "NVIDIA GeForce GTX 1660": (6144,),
    "NVIDIA GeForce GTX 1080 Ti": (11264,),
    "NVIDIA GeForce GTX 1080": (8192,),
    "NVIDIA GeForce GTX 1070 Ti": (8192,),
    "NVIDIA GeForce GTX 1070": (8192,),
    "NVIDIA GeForce GTX 1060": (3072, 6144),
    "NVIDIA Tesla P100": (12288, 16384),
    "NVIDIA Tesla P40": (24576,),
    "NVIDIA Tesla M40": (12288, 24576),
    "NVIDIA Quadro P4000": (8192,),
    "NVIDIA TITAN Xp": (12288,),
}

NORMALIZATION: dict[str, str] = {
    "Tesla V100-SXM2-16GB": "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla V100-SXM2-32GB": "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla V100-PCIE-16GB": "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla V100-PCIE-32GB": "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla H100 80GB HBM3": "NVIDIA H100 80GB HBM3",
    "Tesla T4": "NVIDIA T4 Tensor Core GPU",
    "NVIDIA T4": "NVIDIA T4 Tensor Core GPU",
    "NVIDIA A10": "NVIDIA A10 Tensor Core GPU",
    "NVIDIA A10G": "NVIDIA A10 Tensor Core GPU",
    "Tesla P100-PCIE-16GB": "NVIDIA Tesla P100",
    "Tesla P100-PCIE-12GB": "NVIDIA Tesla P100",
    "Tesla P40": "NVIDIA Tesla P40",
    "Tesla M40": "NVIDIA Tesla M40",
    "Tesla M40 24GB": "NVIDIA Tesla M40",
    "NVIDIA RTX PRO 6000 Blackwell": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
}


@dataclass(frozen=True)
class Window:
    nominal_mb: int
    floor_mb: int
    ceil_mb: int

    def contains(self, total_mb: float) -> bool:
        return self.floor_mb <= total_mb <= self.ceil_mb


@dataclass(frozen=True)
class Match:
    canonical: str
    window: Window
    tier: Tier | None

    @property
    def nominal_gb(self) -> float:
        return self.window.nominal_mb / MIB


def normalize(name: str) -> str:
    clean = (name or "").strip()
    return NORMALIZATION.get(clean, clean)


def window_for(nominal_mb: int) -> Window:
    return Window(
        nominal_mb, round(nominal_mb * VRAM_FLOOR_RATIO), round(nominal_mb * VRAM_CEIL_RATIO)
    )


def windows_for(name: str) -> tuple[Window, ...] | None:
    sizes = GPU_VRAM_MB.get(normalize(name))
    if sizes is None:
        return None
    return tuple(window_for(size) for size in sizes)


def is_known(name: str) -> bool:
    return normalize(name) in GPU_VRAM_MB


def match(name: str, total_mb: float) -> Match | None:
    canonical = normalize(name)
    windows = windows_for(canonical)
    if windows is None:
        return None
    for window in windows:
        if window.contains(total_mb):
            return Match(canonical, window, tier_for_memory_gb(window.nominal_mb / MIB))
    return None


def mismatch_reason(name: str, total_mb: float) -> str:
    canonical = normalize(name)
    windows = windows_for(canonical)
    if windows is None:
        return f"{canonical or 'unnamed GPU'} is not on the accepted list"
    spans = ", ".join(f"{w.floor_mb}-{w.ceil_mb} MB" for w in windows)
    return f"{canonical} reports {total_mb:.0f} MB, outside every accepted window ({spans})"


def probe_memory_mb(name: str) -> int | None:
    sizes = GPU_VRAM_MB.get(normalize(name))
    if not sizes:
        return None
    return round(min(sizes) * VRAM_FLOOR_RATIO)
