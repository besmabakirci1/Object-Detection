#!/usr/bin/env python3
"""
Araç Marka/Model Sınıflandırıcı Eğitimi
- ImageFolder yapısıyla (data_dir/brand_model/...) veri yükler
- Torchvision modeli (varsayılan: resnet18) ile transfer öğrenme
- En iyi ağırlıkları kaydeder, TorchScript olarak dışa aktarır (car_model_classifier.pt)
- labels.txt dosyası üretir

Örnek dizin yapısı (ImageFolder):
 dataset_cars_cls/
 ├── bmw_3series/
 │   ├── img001.jpg
 │   └── ...
 ├── audi_a4/
 │   ├── img010.jpg
 │   └── ...
 └── toyota_corolla/
     ├── ...

Kullanım:
 python train_car_classifier.py \
   --data-dir /path/to/dataset_cars_cls \
   --val-split 0.1 \
   --epochs 15 \
   --batch-size 32 \
   --model resnet18 \
   --out-dir models/car_cls_v1

Eğitim sonrası entegrasyon:
 python Object_Detection_V.py \
   --source 0 \
   --enable-model-cls \
   --model-cls-weights models/car_cls_v1/car_model_classifier.pt \
   --model-cls-labels models/car_cls_v1/labels.txt
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("🍎 Using Apple Silicon (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("🚀 Using CUDA GPU")
        return torch.device("cuda")
    print("💻 Using CPU")
    return torch.device("cpu")


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def create_datasets(data_dir: str, val_split: float, img_size: int, seed: int):
    train_tf, val_tf = build_transforms(img_size)
    full_ds = ImageFolder(root=data_dir, transform=train_tf)
    class_to_idx = full_ds.class_to_idx
    classes = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        classes[idx] = name

    if val_split <= 0:
        return full_ds, None, classes

    val_len = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)

    # validation için farklı transform
    val_ds.dataset = ImageFolder(root=data_dir, transform=val_tf)

    return train_ds, val_ds, classes


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def evaluate(model, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train(args):
    device = get_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, val_ds, classes = create_datasets(args.data_dir, args.val_split, args.img_size, args.seed)
    if len(classes) < 2:
        raise RuntimeError("En az 2 sınıf gerekli. ImageFolder yapınızı kontrol edin.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels.txt yaz
    labels_path = out_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in classes:
            f.write(f"{label}\n")
    print(f"📝 labels.txt yazıldı: {labels_path}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = build_model(args.model, num_classes=len(classes)).to(device)

    # Fine-tuning parametreleri
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 1))
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type in ["cuda"]))

    best_acc = 0.0
    best_path = out_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type in ["cuda"])):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_acc = (0.0, 0.0)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # En iyi modeli kaydet
        metric = val_acc if val_loader is not None else -train_loss
        if metric > best_acc:
            best_acc = metric
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "args": vars(args),
            }, best_path)
            print(f"💾 En iyi model güncellendi: {best_path} (metric={best_acc:.4f})")

    # TorchScript export (model mimarisi kod içinde olduğundan trace/script uygundur)
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    try:
        scripted = torch.jit.trace(model, dummy)
    except Exception:
        scripted = torch.jit.script(model)
    ts_path = out_dir / "car_model_classifier.pt"
    scripted.save(str(ts_path))
    print(f"📦 TorchScript olarak dışa aktarıldı: {ts_path}")
    print(f"✅ Eğitim tamamlandı. En iyi pth: {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Araç marka/model sınıflandırıcı eğitimi")
    parser.add_argument("--data-dir", type=str, required=True, help="ImageFolder kök klasör")
    parser.add_argument("--val-split", type=float, default=0.1, help="Doğrulama oranı (0-1)")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v3"], help="Temel model")
    parser.add_argument("--img-size", type=int, default=224, help="Giriş görüntü boyutu")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=3e-4, help="Öğrenme oranı")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Ağırlık çürümesi")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker sayısı")
    parser.add_argument("--out-dir", type=str, default="models/car_cls_v1", help="Çıktı klasörü")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
