from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return

    for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in ("'", '"')):
            v = v[1:-1]
        os.environ.setdefault(k, v)


def should_enable_wandb(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    return bool(os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_API_KEY"))


def _require_wandb() -> Any:
    try:
        import wandb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("wandb is not installed; install it in your runtime env (e.g. `pip install wandb`).") from e
    return wandb


def init_wandb_run(
    *,
    name: str,
    config: dict[str, Any],
    tags: list[str] | None = None,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
) -> Any:
    wandb = _require_wandb()

    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_DIR", "wandb")

    project = (project or os.environ.get("WANDB_PROJECT") or "").strip()
    if not project:
        raise ValueError("WANDB_PROJECT is required (set it in env or pass --wandb-project).")

    entity = (entity or os.environ.get("WANDB_ENTITY") or "").strip() or None
    group = (group or os.environ.get("WANDB_RUN_GROUP") or "").strip() or None

    api_key = (os.environ.get("WANDB_API_KEY") or "").strip()
    if api_key:
        wandb.login(key=api_key, relogin=True)

    return wandb.init(project=project, entity=entity, name=name, config=config, tags=tags or [], group=group, reinit=True)


def tsv_to_wandb_table(tsv_path: Path, *, max_rows: int = 5000) -> Any:
    wandb = _require_wandb()

    with tsv_path.open(newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        if not header:
            raise ValueError(f"empty tsv: {tsv_path}")
        rows: list[list[str]] = []
        for i, row in enumerate(r):
            if i >= max_rows:
                break
            rows.append([str(x) for x in row])

    return wandb.Table(columns=[str(c) for c in header], data=rows)


def log_artifact_files(run: Any, *, name: str, files: Iterable[Path], artifact_type: str = "results") -> None:
    wandb = _require_wandb()
    art = wandb.Artifact(name=name, type=artifact_type)
    for p in files:
        if p.is_file():
            art.add_file(str(p))
    run.log_artifact(art)

