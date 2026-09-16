"""
environment.py  -  record library versions and hardware (report section "experimental setup").

    python -m src.environment --threads 1
"""
import argparse
import json
import os
import platform

import psutil
import torch

from src import config
from src.quantization import select_quantized_engine

# CPU instruction-set extensions that matter for INT8 / FP16 speed
_INTERESTING_FLAGS = ["avx2", "fma", "f16c", "avx512f", "avx512bw",
                      "avx512_vnni", "avx_vnni", "amx_int8"]


def _cpu_name_windows():
    """CPU model name on Windows (there is no /proc/cpuinfo there)."""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu_info():
    """CPU model name and instruction-set flags."""
    name, flags = None, None
    try:  # Linux
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if name is None and line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                elif flags is None and line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass

    if name is None and platform.system() == "Windows":
        name = _cpu_name_windows()
    if name is None and platform.system() == "Darwin":  # macOS
        try:
            import subprocess
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:
            pass
    name = name or platform.processor() or platform.machine() or "unknown"

    if flags is not None:
        flag_info = {f: (f in flags) for f in _INTERESTING_FLAGS}
    else:
        # Windows/macOS have no flag list. Ask PyTorch itself which vector
        # instruction set it compiled kernels for ("AVX512", "AVX2", ...).
        # NOTE: never infer the CPU's capabilities from whether FBGEMM is
        # present - FBGEMM can be missing from a build on a perfectly capable CPU.
        flag_info = {"torch_cpu_capability": _torch_cpu_capability()}
    return name, flag_info


def _torch_cpu_capability():
    """The widest instruction set PyTorch will actually use ("AVX512", "AVX2", ...)."""
    try:
        return torch.backends.cpu.get_cpu_capability()
    except Exception:
        return "unknown"


def collect_environment():
    import datasets
    import numpy
    import transformers

    cpu_name, cpu_flags = _cpu_info()
    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": numpy.__version__,
        "psutil": psutil.__version__,
        "cpu_model": cpu_name,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "torch_num_threads": torch.get_num_threads(),
        "quantized_engine": torch.backends.quantized.engine,
        "supported_quantized_engines": list(torch.backends.quantized.supported_engines),
        # FBGEMM = PyTorch's fastest INT8 backend; it only registers if the CPU has AVX2
        "fbgemm_available": bool({"fbgemm", "x86"} & set(torch.backends.quantized.supported_engines)),
        "cpu_flags": cpu_flags,
        "cuda_available": torch.cuda.is_available(),
    }
    info["torch_cpu_capability"] = _torch_cpu_capability()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save environment info")
    parser.add_argument("--threads", type=int, default=config.NUM_THREADS)
    parser.add_argument("--results_dir", default=config.RESULTS_DIR)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    select_quantized_engine()
    env = collect_environment()

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, "environment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    for key, value in env.items():
        print(f"  {key:30s}: {value}")
    print(f"[env] saved to {path}")
