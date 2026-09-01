"""Dataset yuklangandan keyin real papka/fayl strukturasini ko'rsatadi."""
from __future__ import annotations

import argparse
from collections import Counter

from common import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(cfg["data"]["raw_dir"])
    print("Dataset root:", root)
    if not root.exists():
        print("Dataset papkasi hali mavjud emas. Avval download_data.py ni run qiling.")
        return

    files = [p for p in root.rglob("*") if p.is_file()]
    print("Jami fayl:", len(files))
    print("Extensionlar:", Counter(p.suffix.lower() for p in files))
    print("\nBirinchi fayllar:")
    for p in files[: args.limit]:
        print(" -", p.relative_to(root))


if __name__ == "__main__":
    main()
