#!/usr/bin/env python3
"""
Araç Veri Seti Model Eğitimi Scripti
Bu script YOLOv5 modelini araç veri setinizle eğitir.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def check_requirements():
    """Gerekli paketlerin yüklü olup olmadığını kontrol eder."""
    try:
        import torch
        import torchvision
        print("✅ PyTorch yüklü")
    except ImportError:
        print("❌ PyTorch yüklü değil. Yükleniyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    
    try:
        import ultralytics
        print("✅ Ultralytics yüklü")
    except ImportError:
        print("❌ Ultralytics yüklü değil. Yükleniyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"])

def check_dataset_structure(dataset_path):
    """Veri seti yapısını kontrol eder."""
    required_dirs = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Eksik klasörler:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\n📁 Veri seti yapısı şöyle olmalı:")
        print("dataset/")
        print("├── images/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("└── labels/")
        print("    ├── train/")
        print("    ├── val/")
        print("    └── test/")
        return False
    
    print("✅ Veri seti yapısı doğru")
    return True

def count_images(dataset_path):
    """Veri setindeki görüntü sayısını sayar."""
    counts = {}
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(dataset_path, 'images', split)
        if os.path.exists(image_dir):
            image_count = len([f for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[split] = image_count
            print(f"📊 {split}: {image_count} görüntü")
    
    return counts

def train_model(dataset_yaml, epochs=100, batch_size=16, img_size=640):
    """YOLOv5 modelini eğitir."""
    print(f"🚀 Model eğitimi başlatılıyor...")
    print(f"📋 Parametreler:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Image Size: {img_size}")
    print(f"   - Dataset: {dataset_yaml}")
    
    # Eğitim komutu
    cmd = [
        "yolo", "train",
        "--data", dataset_yaml,
        "--weights", "yolov5s.pt",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--img-size", str(img_size),
        "--project", "car_detection_model",
        "--name", "car_model_v1"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Model eğitimi başarıyla tamamlandı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model eğitimi hatası: {e}")
        print(f"Çıktı: {e.stdout}")
        print(f"Hata: {e.stderr}")
        return False

def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description='Araç Veri Seti Model Eğitimi')
    parser.add_argument('--dataset', type=str, default='../dataset',
                       help='Veri seti klasörü')
    parser.add_argument('--config', type=str, default='car_dataset.yaml',
                       help='Veri seti konfigürasyon dosyası')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Eğitim epoch sayısı')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch boyutu')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Görüntü boyutu')
    parser.add_argument('--check-only', action='store_true',
                       help='Sadece veri setini kontrol et, eğitim yapma')
    
    args = parser.parse_args()
    
    print("🚗 Araç Veri Seti Model Eğitimi")
    print("=" * 50)
    
    # Gereksinimleri kontrol et
    print("\n1️⃣ Gereksinimler kontrol ediliyor...")
    check_requirements()
    
    # Veri seti yapısını kontrol et
    print(f"\n2️⃣ Veri seti kontrol ediliyor: {args.dataset}")
    if not check_dataset_structure(args.dataset):
        print("❌ Veri seti yapısı hatalı. Lütfen düzeltin.")
        return
    
    # Görüntü sayısını say
    print(f"\n3️⃣ Veri seti istatistikleri:")
    counts = count_images(args.dataset)
    
    total_images = sum(counts.values())
    print(f"📊 Toplam görüntü: {total_images}")
    
    if total_images < 100:
        print("⚠️  Uyarı: Çok az görüntü var. Model performansı düşük olabilir.")
    
    # Sadece kontrol modunda ise çık
    if args.check_only:
        print("\n✅ Sadece kontrol tamamlandı.")
        return
    
    # Model eğitimi
    print(f"\n4️⃣ Model eğitimi başlatılıyor...")
    success = train_model(args.config, args.epochs, args.batch_size, args.img_size)
    
    if success:
        print("\n🎉 Model eğitimi başarıyla tamamlandı!")
        print("📁 Eğitilen model: car_detection_model/car_model_v1/weights/best.pt")
        print("\n📝 Sonraki adımlar:")
        print("1. Eğitilen modeli test edin:")
        print("   python car_detection.py --model car_detection_model/car_model_v1/weights/best.pt")
        print("2. Model performansını değerlendirin")
        print("3. Gerekirse daha fazla veri ekleyip yeniden eğitin")
    else:
        print("\n❌ Model eğitimi başarısız oldu.")
        print("Lütfen hata mesajlarını kontrol edin.")

if __name__ == "__main__":
    main()


