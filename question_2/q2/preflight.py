"""Saved environment and GPU-operation checks before an expensive phase."""
import shutil
import subprocess
import sys
from pathlib import Path
import torch
from .common import capture_dependencies, environment, write_json


def check_environment(output, require_gpu, allow_cpu=False, min_free_gib=8, cpu_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >=3.10 is required")
    report = environment()
    report["free_disk_gib"] = shutil.disk_usage(output).free / 2**30
    smi = shutil.which("nvidia-smi")
    result = subprocess.run([smi, "--query-gpu=index,name,memory.total", "--format=csv"],
                            capture_output=True, text=True, timeout=30) if smi else None
    report["nvidia_smi"] = {"available": bool(smi), "returncode": result.returncode if result else None,
                            "output": result.stdout if result else "not found"}
    report["framework_cuda_available"] = torch.cuda.is_available()
    report["require_gpu"] = require_gpu
    if cpu_only and torch.cuda.is_available():
        raise RuntimeError("This is a CPU-only phase. Set Kaggle Accelerator=None to preserve GPU quota.")
    if require_gpu and not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("GPU phase requested but torch.cuda.is_available() is False; enable Kaggle GPU")
    if require_gpu and torch.cuda.is_available():
        if result is None or result.returncode:
            raise RuntimeError("CUDA is available but nvidia-smi did not confirm the allocation")
        tensor = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
        answer = float((tensor @ tensor.T).sum().item())
        if not torch.isfinite(torch.tensor(answer)):
            raise RuntimeError("GPU matrix-operation check produced nonfinite output")
        torch.cuda.synchronize()
        report["gpu_operation"] = {"status": "PASS", "device": str(tensor.device), "checksum": answer}
        del tensor
        torch.cuda.empty_cache()
    else:
        report["gpu_operation"] = {"status": "NOT_RUN", "reason": "CPU phase or explicit developer CPU override"}
    if report["free_disk_gib"] < min_free_gib:
        raise RuntimeError(f"Only {report['free_disk_gib']:.2f} GiB free; this phase requires {min_free_gib} GiB")
    write_json(output / "environment.json", report)
    capture_dependencies(output)
    print("ENVIRONMENT_CHECK_PASSED", report, flush=True)
    if torch.cuda.device_count() > 1:
        print("Multiple GPUs allocated; this probe checked cuda:0. The phase launcher verifies the training topology separately.", flush=True)
    return report
