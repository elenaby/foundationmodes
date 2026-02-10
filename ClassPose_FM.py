from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import re

# =============================================================================
# USER SETTINGS
# =============================================================================

CLASSPOSE_SRC = r""
PREDICT_WSI_PATH = r""

SLIDE_PATH = r""

# IMPORTANT: write locally first
LOCAL_OUTPUT_DIR = r""
FINAL_OUTPUT_DIR = r""

MODEL_CONFIG = "conic"

REQUESTED_DEVICE = "cuda"
REQUESTED_BATCH_SIZE = 8
REQUESTED_TILE_SIZE = 1024
REQUESTED_OVERLAP = 64

CPU_BATCH_SIZE = 1
CPU_TILE_SIZE = 512
CPU_OVERLAP = 64

USE_TTA = False
FILTER_ARTEFACTS = False

# Optional output types (GeoJSON is default even without these)
OUTPUT_TYPE: Optional[str] = None  # e.g. "csv" or "spatialdata"
TISSUE_DETECTION_MODEL_PATH: Optional[str] = None  # required if OUTPUT_TYPE set per README

# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(txt: str) -> None:
    print("\n" + "=" * 80)
    print(txt)
    print("=" * 80)

def ok(msg: str) -> None:
    print(f"✅ {msg}")

def warn(msg: str) -> None:
    print(f"⚠ {msg}")

def err(msg: str) -> None:
    print(f"❌ {msg}")

# =============================================================================
# UTIL
# =============================================================================

def compile_py(path: str) -> Tuple[bool, str]:
    try:
        src = Path(path).read_text(encoding="utf-8", errors="replace")
        compile(src, path, "exec")
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def ensure_dir(path: str) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        err(f"Cannot create directory: {path}\n{e}")
        return False

def detect_device_and_params() -> Tuple[str, int, int, int]:
    try:
        import torch  # type: ignore
        if REQUESTED_DEVICE.lower() == "cuda" and torch.cuda.is_available():
            ok("CUDA available. Using cuda settings.")
            return "cuda", REQUESTED_BATCH_SIZE, REQUESTED_TILE_SIZE, REQUESTED_OVERLAP
        warn("CUDA not available in this PyTorch build. Using CPU settings.")
        return "cpu", CPU_BATCH_SIZE, CPU_TILE_SIZE, CPU_OVERLAP
    except Exception as e:
        warn(f"Could not import torch ({e}). Using CPU settings.")
        return "cpu", CPU_BATCH_SIZE, CPU_TILE_SIZE, CPU_OVERLAP

def patch_worker_kwarg_if_needed() -> bool:
    """
    Patch only worker(...) call sites if they pass nclasses= to worker(),
    changing to n_classes=.
    """
    header("CHECKING worker() KWARG COMPATIBILITY")

    text = Path(PREDICT_WSI_PATH).read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"(\bworker\s*\([^)]*?)(\bnclasses\s*=)", re.DOTALL)

    if not pattern.search(text):
        ok("No worker(... nclasses=...) found. Patch not needed.")
        return True

    backup = PREDICT_WSI_PATH + ".backup_before_worker_kwarg_patch"
    Path(backup).write_text(text, encoding="utf-8")
    ok(f"Backup written: {backup}")

    patched = pattern.sub(r"\1n_classes=", text)

    # conservative line pass while in worker call block
    patched_lines = []
    in_worker_call = False
    for line in patched.splitlines(True):
        if "worker(" in line:
            in_worker_call = True
        if in_worker_call:
            line = line.replace("nclasses=", "n_classes=")
            if ")" in line:
                in_worker_call = False
        patched_lines.append(line)
    patched = "".join(patched_lines)

    Path(PREDICT_WSI_PATH).write_text(patched, encoding="utf-8")

    ok_compile, msg = compile_py(PREDICT_WSI_PATH)
    if not ok_compile:
        err(f"Patch broke predict_wsi.py: {msg}. Restoring backup.")
        Path(PREDICT_WSI_PATH).write_text(text, encoding="utf-8")
        return False

    ok("worker() kwarg patched safely.")
    return True

def build_cmd(device: str, batch_size: int, tile_size: int, overlap: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "classpose.entrypoints.predict_wsi",
        "--model_config",
        MODEL_CONFIG,
        "--slide_path",
        SLIDE_PATH,
        "--output_folder",
        LOCAL_OUTPUT_DIR,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--tile_size",
        str(tile_size),
        "--overlap",
        str(overlap),
    ]

    cmd += ["--filter_artefacts"] if FILTER_ARTEFACTS else ["--no-filter_artefacts"]
    cmd += ["--tta"] if USE_TTA else ["--no-tta"]

    if OUTPUT_TYPE:
        cmd += ["--output_type", OUTPUT_TYPE]
        if not TISSUE_DETECTION_MODEL_PATH:
            raise RuntimeError(
                "README requires --tissue_detection_model_path when using --output_type csv/spatialdata."
            )
        cmd += ["--tissue_detection_model_path", TISSUE_DETECTION_MODEL_PATH]

    return cmd

def run_streaming(cmd: list[str]) -> int:
    header("RUNNING CLASSPOSE (outputs to LOCAL disk first)")
    print("Command:")
    print(" ".join([f'"{c}"' if " " in c else c for c in cmd]))

    env = os.environ.copy()
    if os.path.exists(CLASSPOSE_SRC):
        env["PYTHONPATH"] = CLASSPOSE_SRC + (os.pathsep + env.get("PYTHONPATH", ""))

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()

    elapsed = time.time() - start
    header("DONE")
    print(f"Return code: {rc}")
    print(f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s")
    return rc

def list_outputs(folder: str) -> None:
    header(f"OUTPUT FILES in {folder}")
    out = Path(folder)
    files = sorted([p for p in out.glob("*") if p.is_file()])
    if not files:
        warn("No output files found.")
        return
    for p in files:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"- {p.name} ({size_mb:.2f} MB)")

def copy_outputs_to_final() -> None:
    """
    Copy local outputs to network drive if available.
    """
    header("COPYING RESULTS TO FINAL OUTPUT (Z: drive)")
    if not os.path.exists(FINAL_OUTPUT_DIR):
        warn(f"Final output folder not reachable right now: {FINAL_OUTPUT_DIR}")
        warn(f"Your results are SAFE locally in: {LOCAL_OUTPUT_DIR}")
        return

    ensure_dir(FINAL_OUTPUT_DIR)

    for p in Path(LOCAL_OUTPUT_DIR).glob("*"):
        if p.is_file():
            dest = Path(FINAL_OUTPUT_DIR) / p.name
            shutil.copy2(p, dest)
            ok(f"Copied: {p.name} -> {dest}")
    ok("Copy complete.")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    header("CLASSPOSE SAFE RUNNER (LOCAL OUTPUT + COPY TO Z:)")

    # basic checks
    if not os.path.exists(SLIDE_PATH):
        err(f"Slide not found: {SLIDE_PATH}")
        return
    ok(f"Slide exists: {SLIDE_PATH}")

    if not os.path.exists(PREDICT_WSI_PATH):
        err(f"predict_wsi.py not found: {PREDICT_WSI_PATH}")
        return
    ok(f"predict_wsi.py found: {PREDICT_WSI_PATH}")

    if not ensure_dir(LOCAL_OUTPUT_DIR):
        return
    ok(f"Local output folder ready: {LOCAL_OUTPUT_DIR}")

    # compile check
    ok_compile, msg = compile_py(PREDICT_WSI_PATH)
    if not ok_compile:
        err(f"predict_wsi.py does not compile: {msg}")
        err("Restore the file from git or a known-good backup first.")
        return
    ok("predict_wsi.py compiles.")

    # patch worker kwarg only if needed
    if not patch_worker_kwarg_if_needed():
        return

    device, batch_size, tile_size, overlap = detect_device_and_params()
    ok(f"Device: {device}")
    ok(f"Batch size: {batch_size}")
    ok(f"Tile size: {tile_size}")
    ok(f"Overlap: {overlap}")

    cmd = build_cmd(device, batch_size, tile_size, overlap)
    rc = run_streaming(cmd)

    # Always list local outputs (even if rc != 0 there might be partials)
    list_outputs(LOCAL_OUTPUT_DIR)

    if rc == 0:
        ok("Inference completed. Attempting copy to Z: ...")
        copy_outputs_to_final()
        list_outputs(FINAL_OUTPUT_DIR)
    else:
        warn("Inference failed. Your LOCAL folder may still contain partial outputs.")
        warn(f"Check: {LOCAL_OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass