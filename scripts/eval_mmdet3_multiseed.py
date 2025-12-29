#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_metrics_json(work_dir: Path) -> Path:
    json_paths = sorted(work_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_paths:
        raise FileNotFoundError(f"No *.json metrics found under {work_dir}")
    return json_paths[-1]


def _infer_metric_type(metrics: dict) -> str:
    if any(k.startswith("coco/") for k in metrics.keys()):
        return "coco"
    if any(k.startswith("lvis/") for k in metrics.keys()):
        return "lvis"
    if any(k.startswith("crowd_human/") for k in metrics.keys()):
        return "crowdhuman"
    return "unknown"


def _parse_metric(metrics_json: Path, metric_type: str) -> dict[str, float | None]:
    import json

    obj = json.loads(metrics_json.read_text(errors="ignore"))
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected metrics json format: {metrics_json}")

    metric_type = metric_type.strip().lower()
    if metric_type == "auto":
        metric_type = _infer_metric_type(obj)

    if metric_type == "coco":
        prefix = "coco"
        if f"{prefix}/bbox_mAP" not in obj:
            raise ValueError(
                f"Missing expected COCO metric key {prefix!r}/bbox_mAP in {metrics_json}; "
                f"got keys={sorted(obj.keys())}"
            )
        return {
            "AP": float(obj.get(f"{prefix}/bbox_mAP", 0.0)) * 100.0,
            "AP50": float(obj.get(f"{prefix}/bbox_mAP_50", 0.0)) * 100.0,
            "AP75": float(obj.get(f"{prefix}/bbox_mAP_75", 0.0)) * 100.0,
            "APs": float(obj.get(f"{prefix}/bbox_mAP_s", 0.0)) * 100.0,
            "APm": float(obj.get(f"{prefix}/bbox_mAP_m", 0.0)) * 100.0,
            "APl": float(obj.get(f"{prefix}/bbox_mAP_l", 0.0)) * 100.0,
            "MR": None,
            "JI": None,
            "time": float(obj.get("time", 0.0)),
        }

    if metric_type == "lvis":
        prefix = "lvis"
        if f"{prefix}/bbox_AP" not in obj:
            raise ValueError(
                f"Missing expected LVIS metric key {prefix!r}/bbox_AP in {metrics_json}; "
                f"got keys={sorted(obj.keys())}. This often means 'testing results is empty' (e.g. no detections)."
            )
        return {
            "AP": float(obj.get(f"{prefix}/bbox_AP", 0.0)) * 100.0,
            "AP50": float(obj.get(f"{prefix}/bbox_AP50", 0.0)) * 100.0,
            "AP75": float(obj.get(f"{prefix}/bbox_AP75", 0.0)) * 100.0,
            "APs": float(obj.get(f"{prefix}/bbox_APs", 0.0)) * 100.0,
            "APm": float(obj.get(f"{prefix}/bbox_APm", 0.0)) * 100.0,
            "APl": float(obj.get(f"{prefix}/bbox_APl", 0.0)) * 100.0,
            "MR": None,
            "JI": None,
            "time": float(obj.get("time", 0.0)),
        }

    if metric_type == "crowdhuman":
        if "crowd_human/mAP" not in obj:
            raise ValueError(
                f"Missing expected CrowdHuman metric key 'crowd_human/mAP' in {metrics_json}; got keys={sorted(obj.keys())}"
            )
        return {
            "AP": float(obj.get("crowd_human/mAP", 0.0)) * 100.0,
            "AP50": None,
            "AP75": None,
            "APs": None,
            "APm": None,
            "APl": None,
            "MR": float(obj.get("crowd_human/mMR", 0.0)) * 100.0,
            "JI": float(obj.get("crowd_human/JI", 0.0)) * 100.0,
            "time": float(obj.get("time", 0.0)),
        }

    raise ValueError(
        f"Unsupported metric_type={metric_type!r} inferred from {metrics_json}; "
        f"keys={sorted(obj.keys())[:30]}"
    )


def _maybe_infer_fps(s_per_img: float | None) -> float | None:
    if s_per_img is None or s_per_img <= 0:
        return None
    return 1.0 / s_per_img


@dataclass(frozen=True)
class EvalRow:
    exp_name: str
    ckpt_path: str
    ckpt_sha256: str
    config_file: str
    sample_step: int
    eval_seed: int
    ap: float
    ap50: float | None
    ap75: float | None
    aps: float | None
    apm: float | None
    apl: float | None
    mr: float | None
    ji: float | None
    inference_s_per_img: float | None
    inference_fps: float | None
    out_dir: str


def _fmt_float(x: float | None, ndigits: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{ndigits}f}"


def _write_tsv(path: Path, rows: Iterable[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "exp_name",
                "ckpt_path",
                "ckpt_sha256",
                "config_file",
                "sample_step",
                "eval_seed",
                "AP",
                "AP50",
                "AP75",
                "APs",
                "APm",
                "APl",
                "MR",
                "JI",
                "inference_s_per_img",
                "inference_fps",
                "out_dir",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.exp_name,
                    r.ckpt_path,
                    r.ckpt_sha256,
                    r.config_file,
                    str(r.sample_step),
                    str(r.eval_seed),
                    f"{r.ap:.4f}",
                    _fmt_float(r.ap50),
                    _fmt_float(r.ap75),
                    _fmt_float(r.aps),
                    _fmt_float(r.apm),
                    _fmt_float(r.apl),
                    _fmt_float(r.mr),
                    _fmt_float(r.ji),
                    "" if r.inference_s_per_img is None else f"{r.inference_s_per_img:.6f}",
                    "" if r.inference_fps is None else f"{r.inference_fps:.3f}",
                    r.out_dir,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="MMDet3 eval-only multi-seed runner (stochastic sampler).")
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--config-file", required=True, help="MMDet3 config path (workspace-relative or absolute).")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    ap.add_argument("--tsv-out", required=True, help="TSV output path.")
    ap.add_argument("--out-base", default="/data/tiasha_archives_runs/mmdet3_evals")
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing metrics json under out_dir when present (skip rerun).",
    )
    ap.add_argument("--sample-step", type=int, default=1, help="Override model.sampling_timesteps.")
    ap.add_argument("--ddim-eta", type=float, default=0.0)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--max-per-img", type=int, default=300)
    ap.add_argument(
        "--metric-type",
        default="auto",
        choices=["auto", "coco", "lvis", "crowdhuman"],
        help="Which evaluator metric to parse from MMEngine json.",
    )
    ap.add_argument("--conda-env", default="mmdet3", help="Conda env name that has mmdet+mmengine.")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    cfg_path = Path(args.config_file)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    ckpt_sha256 = _sha256_file(ckpt_path)

    exp_name = str(args.exp_name)
    out_base = Path(args.out_base).resolve()
    out_root = out_base / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[EvalRow] = []
    for eval_seed in args.eval_seeds:
        out_dir = out_root / f"evalseed{eval_seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cli_log = out_dir / "cli.log"

        metrics_json: Path | None = None
        if bool(args.resume):
            try:
                metrics_json = _latest_metrics_json(out_dir)
            except FileNotFoundError:
                metrics_json = None

        if metrics_json is None:
            cmd: list[str] = [
                "conda",
                "run",
                "-n",
                str(args.conda_env),
                "mim",
                "test",
                "mmdet",
                str(cfg_path),
                "--checkpoint",
                str(ckpt_path),
                "--launcher",
                "none",
                "--work-dir",
                str(out_dir),
                "-y",
                "--cfg-options",
                f"randomness.seed={int(eval_seed)}",
                "randomness.deterministic=False",
                f"model.sampling_timesteps={int(args.sample_step)}",
                f"model.ddim_sampling_eta={float(args.ddim_eta)}",
                f"model.score_thr={float(args.score_thr)}",
                f"model.max_per_img={int(args.max_per_img)}",
            ]

            with cli_log.open("w") as f:
                subprocess.run(
                    cmd, cwd=str(REPO_ROOT), env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT, check=True
                )

            metrics_json = _latest_metrics_json(out_dir)
        metrics = _parse_metric(metrics_json, str(args.metric_type))
        infer_s = metrics.get("time")
        infer_fps = _maybe_infer_fps(infer_s)

        rows.append(
            EvalRow(
                exp_name=exp_name,
                ckpt_path=str(ckpt_path),
                ckpt_sha256=ckpt_sha256,
                config_file=str(cfg_path),
                sample_step=int(args.sample_step),
                eval_seed=int(eval_seed),
                ap=float(metrics["AP"]),
                ap50=None if metrics.get("AP50") is None else float(metrics["AP50"]),  # type: ignore[arg-type]
                ap75=None if metrics.get("AP75") is None else float(metrics["AP75"]),  # type: ignore[arg-type]
                aps=None if metrics.get("APs") is None else float(metrics["APs"]),  # type: ignore[arg-type]
                apm=None if metrics.get("APm") is None else float(metrics["APm"]),  # type: ignore[arg-type]
                apl=None if metrics.get("APl") is None else float(metrics["APl"]),  # type: ignore[arg-type]
                mr=None if metrics.get("MR") is None else float(metrics["MR"]),  # type: ignore[arg-type]
                ji=None if metrics.get("JI") is None else float(metrics["JI"]),  # type: ignore[arg-type]
                inference_s_per_img=float(infer_s) if infer_s is not None and infer_s > 0 else None,
                inference_fps=infer_fps,
                out_dir=str(out_dir),
            )
        )

        extra = ""
        if metrics.get("MR") is not None:
            extra += f" MR={float(metrics['MR']):.4f}"
        if metrics.get("JI") is not None:
            extra += f" JI={float(metrics['JI']):.4f}"
        print(f"eval_seed={eval_seed} AP={metrics['AP']:.4f}{extra} out_dir={out_dir}")

    out_tsv = Path(args.tsv_out)
    _write_tsv(out_tsv, rows)

    ap_vals = [r.ap for r in rows]
    mean_ap = statistics.mean(ap_vals) if ap_vals else float("nan")
    std_ap = statistics.pstdev(ap_vals) if len(ap_vals) > 1 else 0.0
    print(f"wrote_tsv={out_tsv}")
    print(f"ckpt_sha256={ckpt_sha256}")
    print(f"mean_AP={mean_ap:.5f} std_AP_pop={std_ap:.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
