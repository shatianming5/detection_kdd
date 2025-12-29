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

import yaml

from wandb_utils import init_wandb_run, load_dotenv, log_artifact_files, should_enable_wandb, tsv_to_wandb_table

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFDET_DIR = REPO_ROOT / "baselines" / "DiffusionDet"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_copypaste_metrics(log_path: Path) -> dict[str, float]:
    txt = log_path.read_text(errors="ignore")
    for line in reversed(txt.splitlines()):
        if "copypaste:" not in line:
            continue
        m = re.search(
            r"copypaste:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if not m:
            continue
        ap, ap50, ap75, aps, apm, apl = map(float, m.groups())
        return {"AP": ap, "AP50": ap50, "AP75": ap75, "APs": aps, "APm": apm, "APl": apl}
    raise RuntimeError(f"Could not find numeric 'copypaste:' line in {log_path}")


def _load_ims_per_batch(config_yaml: Path) -> float | None:
    try:
        raw = yaml.safe_load(config_yaml.read_text(errors="ignore"))
    except Exception:
        raw = None
    if not isinstance(raw, dict):
        return None

    # Prefer TEST.IMS_PER_BATCH if present; fall back to SOLVER.IMS_PER_BATCH.
    for section in ("TEST", "SOLVER"):
        v = raw.get(section, {})
        if isinstance(v, dict) and v.get("IMS_PER_BATCH") is not None:
            try:
                return float(v["IMS_PER_BATCH"])
            except Exception:
                return None
    return None


def _parse_inference_s_per_img(log_path: Path, config_yaml: Path | None) -> float | None:
    txt = log_path.read_text(errors="ignore")
    # Detectron2 evaluator prints one of:
    #   Total inference time: ... (X.XXXX s / img per device, on N devices)
    #   Total inference time: ... (X.XXXX s / iter per device, on N devices)
    pat = re.compile(
        r"\(([0-9]+(?:\.[0-9]+)?)\s*s\s*/\s*(img|iter)\s*per\s*device(?:,\s*on\s*(\d+)\s*devices)?\)",
        re.IGNORECASE,
    )

    ims_per_batch = None
    if config_yaml is not None and config_yaml.is_file():
        ims_per_batch = _load_ims_per_batch(config_yaml)

    def _to_s_per_img(s_per_unit: float, unit: str, ndev: int | None) -> float | None:
        if unit.lower() == "img":
            return s_per_unit
        if unit.lower() != "iter":
            return None
        if ims_per_batch is None:
            return None
        ndev_i = int(ndev) if ndev is not None and int(ndev) > 0 else 1
        per_device_batch = ims_per_batch / float(ndev_i)
        if per_device_batch <= 0:
            return None
        return s_per_unit / per_device_batch

    for line in reversed(txt.splitlines()):
        if "Total inference time" not in line:
            continue
        m = pat.search(line)
        if not m:
            continue
        s_per_unit = float(m.group(1))
        unit = m.group(2)
        ndev = m.group(3)
        return _to_s_per_img(s_per_unit, unit, ndev)

    for line in reversed(txt.splitlines()):
        m = pat.search(line)
        if not m:
            continue
        s_per_unit = float(m.group(1))
        unit = m.group(2)
        ndev = m.group(3)
        return _to_s_per_img(s_per_unit, unit, ndev)

    return None


def _ensure_rel_to_diffdet(config_file: Path) -> str:
    if config_file.is_absolute():
        try:
            return str(config_file.relative_to(DIFFDET_DIR))
        except ValueError:
            return str(config_file)
    return str(config_file)


@dataclass(frozen=True)
class EvalRow:
    exp_name: str
    ckpt_path: str
    ckpt_sha256: str
    config_file: str
    sample_step: int
    eval_seed: int
    out_dir: str
    ap: float
    ap50: float
    ap75: float
    aps: float
    apm: float
    apl: float
    inference_s_per_img: float | None
    inference_fps: float | None


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
                    r.sample_step,
                    r.eval_seed,
                    f"{r.ap:.4f}",
                    f"{r.ap50:.4f}",
                    f"{r.ap75:.4f}",
                    f"{r.aps:.4f}",
                    f"{r.apm:.4f}",
                    f"{r.apl:.4f}",
                    "" if r.inference_s_per_img is None else f"{r.inference_s_per_img:.6f}",
                    "" if r.inference_fps is None else f"{r.inference_fps:.3f}",
                    r.out_dir,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval-only multi-seed runner (train_seed × eval_seed separation).")
    ap.add_argument("--exp-name", required=True, help="Experiment name (used as output folder prefix).")
    ap.add_argument("--config-file", required=True, help="Detectron2 config file path.")
    ap.add_argument("--weights", required=True, help="Checkpoint path to evaluate.")
    ap.add_argument("--out-base", default=str(REPO_ROOT / "baselines" / "evals"), help="Persistent output base dir.")
    ap.add_argument("--tsv-out", required=True, help="Root TSV output path (workspace-relative recommended).")
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Eval seeds to run.")
    ap.add_argument("--sample-step", type=int, default=1, help="MODEL.DiffusionDet.SAMPLE_STEP override.")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--keep-inference-json", action="store_true", help="Keep inference/*.json outputs (can be large).")
    ap.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log eval results to Weights & Biases. Default: auto-enable when WANDB_PROJECT/WANDB_API_KEY is set.",
    )
    ap.add_argument("--wandb-project", default="", help="Override WANDB_PROJECT.")
    ap.add_argument("--wandb-entity", default="", help="Override WANDB_ENTITY.")
    ap.add_argument("--wandb-name", default="", help="Override W&B run name.")
    ap.add_argument("--wandb-group", default="", help="Override W&B group (WANDB_RUN_GROUP).")
    ap.add_argument("--wandb-tags", nargs="*", default=[], help="Extra W&B tags.")
    ap.add_argument(
        "--opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra detectron2 opts; repeatable. Example: --opt MODEL.DiffusionDet.GEO_FEAT True",
    )
    args = ap.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / args.config_file).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    exp_name = args.exp_name
    out_base = Path(args.out_base).resolve()
    out_root = out_base / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_sha256 = _sha256_file(weights_path)

    config_for_train_net = _ensure_rel_to_diffdet(config_path)
    rows: list[EvalRow] = []
    for eval_seed in args.eval_seeds:
        out_dir = out_root / f"evalseed{eval_seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cli_log = out_dir / "cli.log"

        cmd: list[str] = [
            sys.executable,
            "train_net.py",
            "--config-file",
            config_for_train_net,
            "--num-gpus",
            str(args.num_gpus),
            "--eval-only",
            "MODEL.WEIGHTS",
            str(weights_path),
            "MODEL.DiffusionDet.SAMPLE_STEP",
            str(args.sample_step),
            "OUTPUT_DIR",
            str(out_dir),
            "SEED",
            str(eval_seed),
        ]
        for k, v in args.opt:
            cmd.extend([k, v])

        with cli_log.open("w") as f:
            subprocess.run(cmd, cwd=str(DIFFDET_DIR), env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT, check=True)

        metrics = _parse_copypaste_metrics(out_dir / "log.txt")
        infer_s = _parse_inference_s_per_img(out_dir / "log.txt", out_dir / "config.yaml")
        infer_fps = (1.0 / infer_s) if infer_s is not None and infer_s > 0 else None
        if not args.keep_inference_json:
            inference_dir = out_dir / "inference"
            if inference_dir.exists():
                for p in inference_dir.glob("*.json"):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass

        rows.append(
            EvalRow(
                exp_name=exp_name,
                ckpt_path=str(weights_path),
                ckpt_sha256=ckpt_sha256,
                config_file=str(config_path),
                sample_step=args.sample_step,
                eval_seed=eval_seed,
                out_dir=str(out_dir),
                ap=metrics["AP"],
                ap50=metrics["AP50"],
                ap75=metrics["AP75"],
                aps=metrics["APs"],
                apm=metrics["APm"],
                apl=metrics["APl"],
                inference_s_per_img=infer_s,
                inference_fps=infer_fps,
            )
        )

        print(f"eval_seed={eval_seed} AP={metrics['AP']:.4f} out_dir={out_dir}")

    tsv_out_path = Path(args.tsv_out)
    _write_tsv(tsv_out_path, rows)

    aps = [r.ap for r in rows]
    mean_ap = statistics.mean(aps)
    std_ap = statistics.pstdev(aps) if len(aps) > 1 else 0.0
    print(f"wrote_tsv={args.tsv_out}")
    print(f"ckpt_sha256={ckpt_sha256}")
    print(f"mean_AP={mean_ap:.5f} std_AP_pop={std_ap:.5f}")

    load_dotenv(REPO_ROOT / ".env")
    if should_enable_wandb(args.wandb):
        run = init_wandb_run(
            name=(args.wandb_name or exp_name),
            project=(args.wandb_project or None),
            entity=(args.wandb_entity or None),
            group=(args.wandb_group or None),
            tags=["detectron2", "eval", f"sample_step={int(args.sample_step)}", *list(args.wandb_tags or [])],
            config={
                "stack": "detectron2",
                "exp_name": exp_name,
                "config_file": str(config_path),
                "weights": str(weights_path),
                "ckpt_sha256": ckpt_sha256,
                "sample_step": int(args.sample_step),
                "eval_seeds": list(map(int, args.eval_seeds)),
                "num_gpus": int(args.num_gpus),
                "tsv_out": str(tsv_out_path),
            },
        )
        try:
            payload: dict[str, float | object] = {
                "mean_AP": float(mean_ap),
                "std_AP_pop": float(std_ap),
                "eval_table": tsv_to_wandb_table(tsv_out_path.resolve()),
            }
            fps_vals = [r.inference_fps for r in rows if r.inference_fps is not None]
            if fps_vals:
                payload["mean_inference_fps"] = float(statistics.mean(fps_vals))
            run.log(payload)
            log_artifact_files(
                run,
                name=f"{exp_name}_tsv",
                files=[tsv_out_path.resolve()],
                artifact_type="results",
            )
        finally:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
