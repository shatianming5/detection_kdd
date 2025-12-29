#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy a checkpoint to disk and print its sha256 (for reproducibility).")
    ap.add_argument("--src", required=True, help="Source checkpoint path.")
    ap.add_argument("--dst", required=True, help="Destination checkpoint path.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists.")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.is_file():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(dst)

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(dst)

    digest = sha256_file(dst)
    sha_path = dst.with_suffix(dst.suffix + ".sha256")
    sha_path.write_text(digest + "\n", encoding="utf-8")

    print(f"dst={dst}")
    print(f"sha256={digest}")
    print(f"sha256_file={sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

