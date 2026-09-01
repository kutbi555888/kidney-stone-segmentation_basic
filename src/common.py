"""Project bo'ylab ishlatiladigan umumiy yordamchi funksiyalar."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML config faylini o'qiydi."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """src/ ichidan project root papkasini qaytaradi."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | Path) -> Path:
    """Relative pathni project rootga nisbatan absolute pathga aylantiradi."""
    path = Path(path_value)
    return path if path.is_absolute() else project_root() / path


def ensure_parent(path: str | Path) -> Path:
    """Faylning parent papkasini yaratadi."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Natijalar takrorlanuvchan bo'lishi uchun seedlarni o'rnatadi."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: Any, path: str | Path) -> None:
    """Python obyektini JSON sifatida saqlaydi."""
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
