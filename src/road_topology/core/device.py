"""Centralized device management utilities for CUDA/MPS/CPU.

This module provides unified device detection, selection, and management
across different hardware backends (CUDA, Apple MPS, CPU).

Key features:
- Automatic device detection with priority: CUDA > MPS > CPU
- Device-specific autocast context management (MPS uses float32)
- GradScaler support detection (CUDA only)
- Unified memory cache clearing
- Type-safe tensor/module device transfer
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "DeviceInfo",
    "DeviceNotAvailableError",
    "get_device",
    "get_device_info",
    "get_autocast_context",
    "get_grad_scaler",
    "empty_cache",
    "to_device",
]


T = TypeVar("T", torch.Tensor, torch.nn.Module)


class DeviceNotAvailableError(Exception):
    """Raised when the requested device is not available.

    Args:
        device: The device that was requested but unavailable.
        message: Optional custom error message.
    """

    def __init__(self, device: str, message: str | None = None) -> None:
        self.device = device
        if message is None:
            message = f"Requested device '{device}' is not available on this system"
        super().__init__(message)


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable device information and capabilities.

    Attributes:
        device: The torch.device instance.
        name: Human-readable device name (e.g., "NVIDIA RTX 3090", "Apple M2 Pro").
        type: Device type literal ("cuda", "mps", or "cpu").
        supports_amp: Whether automatic mixed precision is supported.
        supports_scaler: Whether GradScaler is supported (CUDA only).
        supports_float16: Whether float16 operations are stable.
        memory_gb: GPU memory in gigabytes, None for CPU.
    """

    device: torch.device
    name: str
    type: Literal["cuda", "mps", "cpu"]
    supports_amp: bool
    supports_scaler: bool
    supports_float16: bool
    memory_gb: float | None


def _is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available.

    MPS requires both the backend to be built and available on the system.

    Returns:
        True if MPS is available, False otherwise.
    """
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _is_cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available compute device.

    Device priority for "auto": CUDA > MPS > CPU.

    Args:
        preference: Device preference. Options:
            - "auto": Automatically select best available device
            - "cuda": Request CUDA device (raises if unavailable)
            - "cuda:N": Request specific CUDA device N
            - "mps": Request MPS device (raises if unavailable)
            - "cpu": Request CPU device

    Returns:
        torch.device instance for the selected device.

    Raises:
        DeviceNotAvailableError: If the requested device is not available.

    Examples:
        >>> device = get_device()  # Auto-select best
        >>> device = get_device("cuda")  # Require CUDA
        >>> device = get_device("cuda:1")  # Specific GPU
        >>> device = get_device("cpu")  # Force CPU
    """
    preference = preference.lower().strip()

    if preference == "auto":
        if _is_cuda_available():
            return torch.device("cuda")
        if _is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" or preference.startswith("cuda:"):
        if not _is_cuda_available():
            raise DeviceNotAvailableError(
                preference,
                f"CUDA is not available. torch.cuda.is_available() = False",
            )
        return torch.device(preference)

    if preference == "mps":
        if not _is_mps_available():
            raise DeviceNotAvailableError(
                preference,
                "MPS is not available. Ensure you're on Apple Silicon with PyTorch MPS support.",
            )
        return torch.device("mps")

    if preference == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device preference: '{preference}'. "
        f"Valid options: 'auto', 'cuda', 'cuda:N', 'mps', 'cpu'"
    )


def get_device_info(device: torch.device | str | None = None) -> DeviceInfo:
    """Get detailed information about a device's capabilities.

    Args:
        device: Device to query. If None, queries the auto-selected device.
            Can be a torch.device, string like "cuda", or None.

    Returns:
        DeviceInfo dataclass with device capabilities.

    Examples:
        >>> info = get_device_info()  # Info for auto-selected device
        >>> info = get_device_info("cuda:0")
        >>> print(f"Device: {info.name}, Memory: {info.memory_gb}GB")
    """
    if device is None:
        device = get_device("auto")
    elif isinstance(device, str):
        device = torch.device(device)

    device_type = device.type

    if device_type == "cuda":
        device_index = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        memory_gb = props.total_memory / (1024**3)

        return DeviceInfo(
            device=device,
            name=name,
            type="cuda",
            supports_amp=True,
            supports_scaler=True,
            supports_float16=True,
            memory_gb=round(memory_gb, 2),
        )

    if device_type == "mps":
        # MPS device name detection
        # Apple Silicon naming is not directly available via PyTorch
        import platform

        processor = platform.processor()
        if "arm" in processor.lower():
            name = f"Apple Silicon ({platform.machine()})"
        else:
            name = "Apple MPS Device"

        # MPS memory is shared with system RAM, not directly queryable
        return DeviceInfo(
            device=device,
            name=name,
            type="mps",
            supports_amp=True,
            supports_scaler=False,  # GradScaler not supported on MPS
            supports_float16=False,  # float16 is unstable on MPS
            memory_gb=None,  # Shared memory, not GPU-specific
        )

    # CPU fallback
    import platform

    cpu_name = platform.processor() or "CPU"
    return DeviceInfo(
        device=device,
        name=cpu_name,
        type="cpu",
        supports_amp=False,
        supports_scaler=False,
        supports_float16=False,
        memory_gb=None,
    )


@contextlib.contextmanager
def get_autocast_context(
    device: torch.device | str,
    enabled: bool = True,
    dtype: torch.dtype | None = None,
) -> Generator[None, None, None]:
    """Get the appropriate autocast context manager for a device.

    Handles device-specific autocast requirements:
    - CUDA: Uses torch.amp.autocast with float16 by default
    - MPS: Uses torch.amp.autocast with float32 (float16 is unstable)
    - CPU: Returns nullcontext (no-op)

    Args:
        device: Target device for autocast.
        enabled: Whether to enable autocast. If False, returns nullcontext.
        dtype: Override dtype for autocast. If None, uses device-appropriate default.

    Yields:
        Context manager for the autocast scope.

    Examples:
        >>> device = get_device()
        >>> with get_autocast_context(device):
        ...     output = model(input)  # Runs with mixed precision if supported

        >>> # Disable autocast
        >>> with get_autocast_context(device, enabled=False):
        ...     output = model(input)  # Runs in full precision
    """
    if isinstance(device, str):
        device = torch.device(device)

    device_type = device.type

    if not enabled:
        yield
        return

    if device_type == "cuda":
        # CUDA supports float16 autocast
        autocast_dtype = dtype if dtype is not None else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            yield

    elif device_type == "mps":
        # MPS: float16 is unstable, use float32 for stability
        autocast_dtype = dtype if dtype is not None else torch.float32
        with torch.amp.autocast(device_type="mps", dtype=autocast_dtype):
            yield

    else:
        # CPU: No autocast support, just pass through
        yield


def get_grad_scaler(device: torch.device | str) -> torch.amp.GradScaler | None:
    """Get a GradScaler if supported by the device.

    GradScaler is used for mixed precision training to prevent gradient underflow.
    Only CUDA devices support GradScaler.

    Args:
        device: Target device for the scaler.

    Returns:
        GradScaler instance for CUDA devices, None for MPS/CPU.

    Examples:
        >>> device = get_device()
        >>> scaler = get_grad_scaler(device)
        >>> if scaler:
        ...     # Use scaler for CUDA mixed precision training
        ...     scaler.scale(loss).backward()
        ...     scaler.step(optimizer)
        ...     scaler.update()
        ... else:
        ...     # Standard backprop for MPS/CPU
        ...     loss.backward()
        ...     optimizer.step()
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        return torch.amp.GradScaler(device="cuda")

    # MPS and CPU do not support GradScaler
    return None


def empty_cache(device: torch.device | str | None = None) -> None:
    """Clear the device memory cache.

    Handles device-specific cache clearing:
    - CUDA: torch.cuda.empty_cache()
    - MPS: torch.mps.empty_cache()
    - CPU: No-op (no cache to clear)

    Args:
        device: Device whose cache to clear. If None, clears all available caches.

    Examples:
        >>> empty_cache()  # Clear all device caches
        >>> empty_cache("cuda")  # Clear only CUDA cache
        >>> empty_cache(torch.device("mps"))  # Clear only MPS cache
    """
    if device is None:
        # Clear all available caches
        if _is_cuda_available():
            torch.cuda.empty_cache()
        if _is_mps_available():
            torch.mps.empty_cache()
        return

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    # CPU has no cache to clear


def to_device(
    obj: T,
    device: torch.device | str,
    non_blocking: bool = True,
) -> T:
    """Move a tensor or module to the specified device.

    Provides unified device transfer with proper handling for different backends.

    Args:
        obj: Tensor or nn.Module to move.
        device: Target device.
        non_blocking: If True, perform asynchronous transfer when possible.
            Only effective for CUDA with pinned memory.

    Returns:
        The tensor or module on the target device.

    Examples:
        >>> tensor = torch.randn(100, 100)
        >>> tensor = to_device(tensor, "cuda")

        >>> model = MyModel()
        >>> model = to_device(model, get_device())
    """
    if isinstance(device, str):
        device = torch.device(device)

    return obj.to(device=device, non_blocking=non_blocking)
