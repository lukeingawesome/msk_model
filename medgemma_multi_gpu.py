"""
Multi‑GPU, multi‑process inference for MedGemma‑4B‑IT
----------------------------------------------------

Key ideas
---------
1. **One process ⇢ one GPU** – prevents model duplication on the same card.
2. **Pipeline loaded once per process** via a Pool *initializer* instead of
   inside the task function.
3. **inference_mode() + no_grad()** – skips autograd bookkeeping.
4. **Global pipe handle** – eliminates re‑loading the model every call.
5. **Chunkless map** – distribute every image path straight to the Pool;
   tqdm wraps the iterator for progress.
6. **CSV checkpointing every N images** – avoids the heavy full‑file rewrite
   per chunk but still provides crash‑recovery.
7. **Graceful CUDA clean‑up** – torch.cuda.empty_cache() only on exit.
"""

from __future__ import annotations
import os, gc, time, multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


# ──────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────
def load_and_preprocess_image(img_path: str) -> Optional[Image.Image]:
    """Load an image, convert to RGB, return None on failure."""
    try:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[WARN] Missing file: {img_path}")
            return None
        im = Image.open(img_path)
        return im.convert("RGB") if im.mode != "RGB" else im
    except Exception as exc:
        print(f"[ERR ] Could not read {img_path}: {exc}")
        return None


def create_messages(image: Image.Image) -> List[Dict[str, Any]]:
    """Prompts expected by MedGemma‑IT."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this X‑ray"},
                {"type": "image", "image": image},
            ],
        },
    ]


# ──────────────────────────────────────────────────────────────
#  Pool‑level (per‑process) globals
# ──────────────────────────────────────────────────────────────
_PIPE = None  # loaded once in `init_worker`


def init_worker(gpu_count: int):
    """
    Executed *once* in every worker process.  
    Chooses a GPU deterministically from the pool index (0‑based).
    """
    global _PIPE

    # Get this process' slot ID within the pool: 0,1,2,…
    worker_rank = mp.current_process()._identity[0] - 1  # `_identity` starts at 1
    gpu_id = worker_rank % gpu_count if gpu_count else -1

    if torch.cuda.is_available() and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)

    _PIPE = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device=f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu",
    )

    # Optional warm‑up (couple of dummy forward passes lowers first‑batch latency)
    dummy = Image.new("RGB", (512, 512), 0)
    with torch.inference_mode():
        _PIPE(text=create_messages(dummy), max_new_tokens=1)
    print(f"[INFO] Worker {worker_rank} ready on device {gpu_id}")


def infer_image(img_path: str) -> str:
    """Run MedGemma on a single image path; rely on global _PIPE."""
    global _PIPE
    try:
        image = load_and_preprocess_image(img_path)
        if image is None:
            return "ERROR: could not load image"

        with torch.inference_mode():
            out = _PIPE(text=create_messages(image), max_new_tokens=500)

        return out[0]["generated_text"][-1]["content"]
    except Exception as exc:
        return f"ERROR processing {img_path}: {exc}"


# ──────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────
def main() -> None:
    # ---------- 1. Load CSV ----------
    df_path = Path("/data3/private/knee/supplementary/all.csv")
    df: pd.DataFrame = pd.read_csv(df_path)

    if "medgemma" not in df.columns:
        df["medgemma"] = pd.NA

    todo_mask = df["medgemma"].isna() | (df["medgemma"] == "")
    if not todo_mask.any():
        print("Nothing left to do – all rows already have medgemma text.")
        return

    img_paths: List[str] = df.loc[todo_mask, "img_path"].tolist()
    indices: List[int] = df.loc[todo_mask].index.tolist()
    assert len(img_paths) == len(indices)

    # ---------- 2. Pool setup ----------
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("[WARN] No CUDA devices visible – running on CPU.")
    num_workers = max(1, gpu_count)

    mp.set_start_method("spawn", force=True)
    print(f"[INFO] Spawning {num_workers} worker(s) across {gpu_count} GPU(s)…")

    # ---------- 3. Parallel inference ----------
    start = time.time()
    batch_size_checkpt = 1000  # save every N images (increased by 4x)

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(gpu_count,),
    ) as pool:
        for k, result in enumerate(
            tqdm(pool.imap(infer_image, img_paths), total=len(img_paths), desc="Infer")
        ):
            df.at[indices[k], "medgemma"] = result

            if (k + 1) % batch_size_checkpt == 0:
                # intermediate checkpoint
                df.to_csv("all_medgemma_multi_gpu.csv", index=False)

    # ---------- 4. Finalise ----------
    df.to_csv("all_medgemma_multi_gpu.csv", index=False)

    elapsed = time.time() - start
    print(f"✓ Completed {len(img_paths)} images in {elapsed/60:.1f} min "
          f"({elapsed/len(img_paths):.2f} s / image)")

    # Clean‑up CUDA memory *after* processes have terminated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
