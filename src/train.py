"""2D U-Net training: KSSD2025 binary kidney-stone segmentation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import load_config, resolve_path, set_seed
from dataset import KidneyStoneDataset
from losses import dice_bce_loss
from model import build_model


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    pred = (torch.sigmoid(logits) >= threshold).float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return float((2 * inter + 1e-6) / (denom + 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    splits = resolve_path(cfg["data"]["splits_dir"])
    train_df = pd.read_csv(splits / "train.csv")
    val_df = pd.read_csv(splits / "val.csv")
    size = tuple(int(x) for x in cfg["data"]["image_size"])

    train_ds = KidneyStoneDataset(train_df, size, augment=True)
    val_ds = KidneyStoneDataset(val_df, size, augment=False)

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    amp_enabled = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_path = resolve_path(cfg["paths"]["best_model"])
    last_path = resolve_path(cfg["paths"]["last_checkpoint"])
    best_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val = -1.0
    stale = 0

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", -1.0))
        stale = int(ckpt.get("stale", 0))
        print(f"Checkpointdan davom: epoch {start_epoch + 1}")

    # ========================================================
    # Training history
    #
    # Resume bo'lsa oldingi history CSVni davom ettiramiz.
    # Fresh training bo'lsa yangi history boshlanadi.
    # ========================================================

    history_path = resolve_path(
        cfg["paths"]["history_csv"]
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []

    if args.resume and history_path.exists():

        try:

            old_history_df = pd.read_csv(
                history_path
            )

            # Checkpoint bilan mos epochlargacha historyni olamiz.
            # start_epoch = keyingi boshlanadigan epoch indexi.
            #
            # Masalan:
            # checkpoint epoch 15 tugagan bo'lsa
            # start_epoch = 15
            # CSVda epoch <= 15 qoldiriladi.
            old_history_df = old_history_df[
                old_history_df["epoch"]
                <= start_epoch
            ].copy()

            history = old_history_df.to_dict(
                orient="records"
            )

            print(
                "Oldingi training history yuklandi:",
                len(history),
                "epoch"
            )

        except Exception as e:

            print(
                "History CSVni o'qishda xato:",
                e
            )

            print(
                "History bo'sh holatdan davom etadi."
            )

            history = []

    epochs = int(tcfg["epochs"])
    threshold = float(tcfg["threshold"])
    patience = int(tcfg["early_stopping_patience"])

    for epoch in range(start_epoch, epochs):
        # 1-BO'LIM: training.
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} train"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = dice_bce_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # 2-BO'LIM: validation.
        model.eval()
        val_loss = 0.0
        dice_num = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} val"):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(x)
                    loss = dice_bce_loss(logits, y)
                val_loss += loss.item() * x.size(0)
                dice_num += dice_from_logits(logits, y, threshold) * x.size(0)
        val_loss /= len(val_ds)
        val_dice = dice_num / len(val_ds)

        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice}
        history.append(row)
        print(row)

        # ====================================================
        # 3-BO'LIM:
        # Eng yaxshi model va early-stopping state'ni
        # AVVAL yangilaymiz.
        # ====================================================

        if val_dice > best_val:

            best_val = val_dice
            stale = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "best_val_dice": best_val,
                },
                best_path
            )

            print(
                f"Yangi best model saqlandi: "
                f"Dice={best_val:.4f}"
            )

        else:

            stale += 1


        # ====================================================
        # 4-BO'LIM:
        # Resume checkpointni best_val va stale
        # yangilangandan KEYIN saqlaymiz.
        #
        # Shu sabab keyingi --resume aynan oxirgi
        # training state'dan davom etadi.
        # ====================================================

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "stale": stale,
                "config": cfg,
            },
            last_path
        )

        # Har epochdan keyin to'liq training history saqlanadi.
        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False
        )

        if stale >= patience:
            print(f"Early stopping: {patience} epoch davomida improvement bo'lmadi.")
            break


if __name__ == "__main__":
    main()
