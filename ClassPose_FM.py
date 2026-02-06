

"""
CLASSPOSE: One-piece WSI inference runner (Windows) - robust + compatible with your CLI

Your CLI supports:
  --model_config, --slide_path, --output_folder
  --device, --batch_size, --tile_size, --overlap
  --tta/--no-tta, --filter_artefacts/--no-filter_artefacts
  --output_type {csv,spatialdata} ...
  --tissue_detection_model_path, --artefact_detection_model_path

Your CLI DOES NOT support:
  --num_workers   (we do NOT pass it)

This script:
1) Checks paths
2) Ensures predict_wsi.py compiles (restores from backups if needed)
3) Patches worker() kwarg mismatch if found (nclasses -> n_classes in worker call only)
4) Detects CUDA and falls back to CPU if torch is CPU-only
5) Runs inference using GitHub-style module call
6) Lists outputs
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import re

# ==============================================================================
# USER SETTINGS (edit if needed)
# ==============================================================================

CLASSPOSE_SRC = r""
PREDICT_WSI_PATH = r""

SLIDE_PATH = r""
OUTPUT_DIR = r""

# GitHub preset model config
MODEL_CONFIG = "conic"

# Requested settings (script will auto-correct based on torch capabilities)
REQUESTED_DEVICE = "cuda"  # will fall back to cpu if needed
REQUESTED_BATCH_SIZE = 8
REQUESTED_TILE_SIZE = 1024
REQUESTED_OVERLAP = 64

# CPU-safe defaults (used if CUDA not available)
CPU_BATCH_SIZE = 1          # recommended on CPU
CPU_TILE_SIZE = 1024        # if too slow or memory heavy, set 512
CPU_OVERLAP = 64

# Flags
USE_TTA = False
FILTER_ARTEFACTS = False

# Optional outputs
# If you want csv or spatialdata, your CLI supports:
#   --output_type csv
# But note: repo README says for csv/spatialdata you may need tissue detection path.
OUTPUT_TYPE: Optional[str] = None  # "csv" or "spatialdata" or None
TISSUE_DETECTION_MODEL_PATH: Optional[str] = None  # set if OUTPUT_TYPE is csv/spatialdata


# ==============================================================================
# PRINT HELPERS
# ==============================================================================

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


# ==============================================================================
# CHECKS & RESTORE
# ==============================================================================

def ensure_paths() -> bool:
    good = True

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ok(f"Output folder ready: {OUTPUT_DIR}")

    if not os.path.exists(SLIDE_PATH):
        err(f"Slide not found: {SLIDE_PATH}")
        good = False
    else:
        ok(f"Slide exists: {SLIDE_PATH}")

    if not os.path.exists(PREDICT_WSI_PATH):
        err(f"predict_wsi.py not found: {PREDICT_WSI_PATH}")
        good = False
    else:
        ok(f"predict_wsi.py found: {PREDICT_WSI_PATH}")

    if os.path.exists(CLASSPOSE_SRC) and CLASSPOSE_SRC not in sys.path:
        sys.path.insert(0, CLASSPOSE_SRC)
        ok(f"Added src to sys.path: {CLASSPOSE_SRC}")

    return good


def compile_py(path: str) -> Tuple[bool, str]:
    try:
        src = Path(path).read_text(encoding="utf-8", errors="replace")
        compile(src, path, "exec")
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def find_backup_candidates() -> list[Path]:
    p = Path(PREDICT_WSI_PATH)
    parent = p.parent
    backups = sorted(
        parent.glob(p.name + ".backup*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    # also check predict_wsi.py.backup
    plain = parent / (p.name + ".backup")
    if plain.exists() and plain not in backups:
        backups.append(plain)
    return backups


def restore_predict_wsi_if_broken() -> bool:
    header("CHECKING predict_wsi.py HEALTH")
    ok_now, msg = compile_py(PREDICT_WSI_PATH)
    if ok_now:
        ok("predict_wsi.py compiles.")
        return True

    warn(f"predict_wsi.py is broken: {msg}")
    warn("Attempting restore from backups...")

    backups = find_backup_candidates()
    if not backups:
        err("No backup files found next to predict_wsi.py")
        return False

    ok(f"Found {len(backups)} backup(s). Trying newest first...")
    target = Path(PREDICT_WSI_PATH)
    original = target.read_text(encoding="utf-8", errors="replace")

    for b in backups:
        try:
            candidate = b.read_text(encoding="utf-8", errors="replace")
            try:
                compile(candidate, str(b), "exec")
            except Exception:
                warn(f"Backup does not compile, skipping: {b.name}")
                continue

            target.write_text(candidate, encoding="utf-8")
            ok_after, msg_after = compile_py(PREDICT_WSI_PATH)
            if ok_after:
                ok(f"Restored predict_wsi.py from backup: {b.name}")
                return True
            else:
                warn(f"Restored from {b.name} but still broken: {msg_after}")
                target.write_text(original, encoding="utf-8")

        except Exception as e:
            warn(f"Failed to read/restore backup {b.name}: {e}")

    err("No usable backup could restore a compilable predict_wsi.py")
    return False


# ==============================================================================
# PATCH: worker(nclasses=...) -> worker(n_classes=...)
# ==============================================================================

def patch_worker_kwarg_if_needed() -> bool:
    header("CHECKING worker() KWARG COMPATIBILITY")

    text = Path(PREDICT_WSI_PATH).read_text(encoding="utf-8", errors="replace")

    if "worker(" not in text:
        warn("No 'worker(' call found. Skipping.")
        return True

    # Patch only if worker call uses nclasses=
    pattern = re.compile(r"(\bworker\s*\([^)]*?)(\bnclasses\s*=)", re.DOTALL)
    if not pattern.search(text):
        ok("No worker(... nclasses=...) found. Patch not needed.")
        return True

    backup = PREDICT_WSI_PATH + ".backup_before_worker_kwarg_patch"
    Path(backup).write_text(text, encoding="utf-8")
    ok(f"Backup written: {backup}")

    patched = pattern.sub(r"\1n_classes=", text)

    # extra conservative pass in worker call blocks
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
    ok("Patched worker() kwarg: nclasses -> n_classes")

    ok_compile, msg = compile_py(PREDICT_WSI_PATH)
    if not ok_compile:
        err(f"After patch, predict_wsi.py does not compile: {msg}")
        err("Restoring backup...")
        Path(PREDICT_WSI_PATH).write_text(text, encoding="utf-8")
        return False

    ok("predict_wsi.py compiles after patch.")
    return True


# ==============================================================================
# DEVICE SELECTION (NO CUDA TORCH -> CPU)
# ==============================================================================

def detect_device_and_params() -> Tuple[str, int, int, int]:
    """
    Returns: (device, batch_size, tile_size, overlap)
    """
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


# ==============================================================================
# RUN INFERENCE (NO --num_workers)
# ==============================================================================

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
        OUTPUT_DIR,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--tile_size",
        str(tile_size),
        "--overlap",
        str(overlap),
    ]

    # artefact filter toggle
    if FILTER_ARTEFACTS:
        cmd += ["--filter_artefacts"]
    else:
        cmd += ["--no-filter_artefacts"]

    # TTA toggle
    if USE_TTA:
        cmd += ["--tta"]
    else:
        cmd += ["--no-tta"]

    # optional output type
    if OUTPUT_TYPE:
        cmd += ["--output_type", OUTPUT_TYPE]
        # optional tissue model path (helpful/required for csv/spatialdata in some setups)
        if TISSUE_DETECTION_MODEL_PATH:
            cmd += ["--tissue_detection_model_path", TISSUE_DETECTION_MODEL_PATH]

    return cmd


def run_streaming(cmd: list[str]) -> int:
    header("RUNNING CLASSPOSE (GitHub-style)")
    print("Command:")
    print(" ".join([f'"{c}"' if " " in c else c for c in cmd]))

    env = os.environ.copy()
    if os.path.exists(CLASSPOSE_SRC):
        env["PYTHONPATH"] = CLASSPOSE_SRC + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

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


def list_outputs() -> None:
    header("OUTPUT FILES")
    out = Path(OUTPUT_DIR)
    files = sorted([p for p in out.glob("*") if p.is_file()])
    if not files:
        warn("No output files found yet.")
        return
    for p in files:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"- {p.name} ({size_mb:.2f} MB)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    header("CLASSPOSE: ONE-PIECE CHECK + PATCH + RUN (COMPATIBLE WITH YOUR CLI)")

    if not ensure_paths():
        err("Fix missing paths and re-run.")
        return

    if not restore_predict_wsi_if_broken():
        err("Cannot proceed: predict_wsi.py is broken and no backup restore worked.")
        return

    if not patch_worker_kwarg_if_needed():
        err("worker() kwarg patch failed.")
        return

    device, batch_size, tile_size, overlap = detect_device_and_params()

    ok(f"Using model_config: {MODEL_CONFIG}")
    ok(f"Device: {device}")
    ok(f"Batch size: {batch_size}")
    ok(f"Tile size: {tile_size}")
    ok(f"Overlap: {overlap}")

    if device == "cpu":
        warn("Running on CPU (PyTorch is CPU-only). This will be slow for 2145 tiles.")
        warn("If it is too slow, set CPU_TILE_SIZE=512 and keep CPU_BATCH_SIZE=1.")

    cmd = build_cmd(device, batch_size, tile_size, overlap)
    rc = run_streaming(cmd)

    if rc == 0:
        ok("Inference finished successfully.")
        list_outputs()
    else:
        err("Inference failed.")
        print("\nNext steps if CPU is too slow or crashes:")
        print("- Set CPU_TILE_SIZE=512")
        print("- Keep CPU_BATCH_SIZE=1")
        print("- Optionally enable tissue detection to reduce tiles")
        print("  (requires tissue model path if you use csv/spatialdata outputs)")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
