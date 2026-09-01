"""
KSSD2025 datasetini Kaggle'dan yuklab olish.

Bu bo'limning vazifasi:
1) KaggleHub orqali public datasetni olish;
2) uni project ichidagi data/raw/kssd2025 papkasiga joylashtirish;
3) keyingi prepare_data.py bosqichi uchun tayyorlash.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import load_config, resolve_path

DATASET_HANDLE = "murillobouzon/kssd2025-kidney-stone-segmentation-dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(cfg["data"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset:", DATASET_HANDLE)
    print("Saqlash papkasi:", out_dir)

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub o'rnatilmagan. pip install -r requirements.txt ni run qiling.") from exc

    # Yangi KaggleHub versiyalarida output_dir mavjud.
    # Agar muhitdagi versiya bu parametrni qabul qilmasa, fallback ishlaydi.
    try:
        downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(out_dir)))
    except TypeError:
        downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE))
        print("Kaggle cache path:", downloaded)
        if downloaded.resolve() != out_dir.resolve():
            for item in downloaded.iterdir():
                target = out_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)

    print("\nDataset yuklandi.")
    print("KaggleHub qaytargan path:", downloaded)
    print("Project raw path:", out_dir)

    files = [p for p in out_dir.rglob("*") if p.is_file()]
    print("Topilgan fayllar:", len(files))
    for p in files[:20]:
        print(" -", p.relative_to(out_dir))


if __name__ == "__main__":
    main()
