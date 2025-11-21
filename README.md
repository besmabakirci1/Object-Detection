# 🚗 Araç Nesne Algılama ve Model Sınıflandırma Projesi

Bu proje, YOLOv5 modelini kullanarak video akışlarından araç nesnelerini algılayan ve araç marka/model bilgilerini gösteren bir Python uygulamasıdır. Hem YouTube videoları hem de yerel video dosyaları işleyebilir.

## ✨ Özellikler

- 🎥 **YouTube Video Desteği**: YouTube URL'lerinden doğrudan video işleme
- 📁 **Yerel Dosya Desteği**: Bilgisayarınızdaki video dosyalarını işleme
- 🚀 **Gerçek Zamanlı İşleme**: Canlı video akışı ile nesne algılama
- 🎯 **Araç Modeli Sınıflandırma**: Algılanan araçların marka/model bilgisini gösterir
- 🎨 **Renkli Görselleştirme**: Her araç türü için farklı renkler
- 💻 **Çoklu Platform**: macOS, Windows ve Linux desteği
- 🔧 **GPU Hızlandırma**: CUDA ve Apple Silicon (MPS) desteği

## 🛠️ Kurulum

### Gereksinimler

- Python 3.8+ (Python 3.10+ önerilir)
- PyTorch
- OpenCV
- yt-dlp

### Adım Adım Kurulum

1. **Projeyi klonlayın:**
```bash
git clone https://github.com/besmabakirci1/Object-Detection.git
cd Object-Detection
```

2. **Sanal ortam oluşturun:**
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# veya
.venv\Scripts\activate  # Windows
```

3. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

**Not:** PyTorch'u sisteminize göre yükleyin:
- **macOS (Apple Silicon):** `pip install torch torchvision`
- **CUDA GPU:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- **CPU:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

4. **YOLOv5 modeli otomatik indirilecek** (ilk çalıştırmada)

5. **Araç Modeli Sınıflandırma (Opsiyonel):**
   - Eğer `models/car_cls_v1/car_model_classifier.pt` dosyası varsa otomatik aktif olur
   - Yoksa sadece araç türü (car, truck, bus vb.) gösterilir

## 🚀 Kullanım

### Temel Kullanım

**Yerel video dosyası ile:**
```bash
python Object_Detection_V.py --source "video.mp4" --output "cikti.mp4"
```

**YouTube videosu ile:**
```bash
python Object_Detection_V.py --source "https://youtube.com/watch?v=VIDEO_ID" --output "cikti.mp4"
```

**Webcam ile:**
```bash
python Object_Detection_V.py --source 0 --output "webcam_cikti.mp4"
```

### Komut Satırı Parametreleri

```bash
# Temel kullanım
python Object_Detection_V.py --source "video.mp4" --output "cikti.mp4"

# Araç modeli sınıflandırmasını aktif et
python Object_Detection_V.py --source "video.mp4" --enable-model-cls \
  --model-cls-weights "models/car_cls_v1/car_model_classifier.pt" \
  --model-cls-labels "models/car_cls_v1/labels.txt"

# Farklı çıktı dosyası ile
python Object_Detection_V.py --source "video.mp4" --output "ozel_cikti.mp4"
```

### Python Kodu ile Kullanım

```python
from Object_Detection_V import ObjectDetection

# Yerel dosya ile
detector = ObjectDetection('video.mp4', out_file="cikti.mp4")
detector()

# YouTube videosu ile
detector = ObjectDetection('https://youtube.com/watch?v=VIDEO_ID', 
                          out_file="youtube_cikti.mp4")
detector()

# Araç modeli sınıflandırması ile
detector = ObjectDetection(
    'video.mp4',
    out_file="cikti.mp4",
    enable_model_cls=True,
    model_cls_weights="models/car_cls_v1/car_model_classifier.pt",
    model_cls_labels=["BMW 3 Series", "Mercedes C-Class", ...]
)
detector()
```

## 🎨 Çıktı Özellikleri

### Renkli Görselleştirme

Her araç türü için farklı renkler kullanılır:

- 🚗 **Car** → Kırmızı
- 🚛 **Truck** → Turuncu
- 🚌 **Bus** → Mor
- 🏍️ **Motorcycle** → Pembe
- 🚲 **Bicycle** → Cyan
- ✈️ **Airplane** → Altın
- 🚂 **Train** → Koyu Yeşil

### Label Formatı

- **Araç modeli sınıflandırması aktifse:** `BMW 3 Series (85%)`
- **Sadece araç türü:** `car (90%)`

## 📋 Dosya Yapısı

```
Object-Detection/
├── Object_Detection_V.py          # Ana uygulama (geliştirilmiş versiyon)
├── MuratHoca_ObjectDetection_V.py # Orijinal versiyon
├── car_model_classifier.py         # Araç modeli sınıflandırıcı
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
├── models/
│   └── car_cls_v1/
│       ├── car_model_classifier.pt # Araç modeli sınıflandırma modeli
│       └── labels.txt              # Araç modeli etiketleri
└── datasets/                       # Veri setleri (opsiyonel)
```

## 🔧 Sorun Giderme

### YouTube Videoları Açılmıyor

- Video DRM korumalı olabilir → Yerel video dosyası kullanın
- yt-dlp'yi güncelleyin: `pip install --upgrade yt-dlp`
- İnternet bağlantınızı kontrol edin

### Model Yüklenmiyor

- İnternet bağlantınızı kontrol edin (YOLOv5 otomatik indirilir)
- PyTorch'un doğru yüklendiğinden emin olun: `python -c "import torch; print(torch.__version__)"`

### GPU Kullanılmıyor

- **Apple Silicon:** MPS otomatik aktif olur
- **NVIDIA GPU:** CUDA yüklü olduğundan emin olun
- **CPU:** Otomatik olarak CPU kullanılır

### Video Penceresi Açılmıyor

- OpenCV'nin doğru yüklendiğinden emin olun: `python -c "import cv2; print(cv2.__version__)"`
- GUI desteği olan OpenCV yükleyin: `pip install opencv-python` (opencv-python-headless değil)

## 📊 Örnek Çıktılar

Video işlendiğinde:
- Gerçek zamanlı video penceresi açılır
- Araçlar renkli kutularla işaretlenir
- Marka/model bilgileri gösterilir
- FPS bilgisi ekranda görünür
- İşlenmiş video kaydedilir

**Kontroller:**
- `q` tuşu → Çıkış
- Video penceresi kapatılırsa işlem durur

## 📂 Dataset Seçenekleri – Araç Model Tespiti

### 1. Stanford Cars Dataset
- **İçerik:** 16,185 görsel, 196 sınıf
- **Detay:** Marka + model + yıl (ör. *2012 Tesla Model S*)
- **Ekstra:** Bounding box + sınıf etiketleri mevcut
- **Amaç:** İnce ayrım (fine-grained classification)

### 2. VeRi-776
- **İçerik:** 49,357 görsel, 776 araç, 20 kamera
- **Ekstra:** Bounding box, marka, tip, renk etiketleri
- **Amaç:** Araç yeniden tanıma (Re-ID), trafik senaryolarında takip

### 3. Vehicle Dataset for YOLO
- **İçerik:** 3,000 görsel, 3,830 nesne
- **Sınıflar:** `car`, `threewheel`, `bus`, `truck`, `motorbike`, `van`
- **Amaç:** YOLO için hızlı başlangıç – genel araç tespiti

### 4. Roboflow Car Model Detection
- **Kaynak:** [Roboflow Universe](https://universe.roboflow.com/mxk/car-model-detection/dataset/1)
- **İndirme Komutu:**
```bash
curl -L "https://universe.roboflow.com/ds/FVQJTmNQ5U?key=LaeWMqO6ju" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip
```

## 📖 Nesne Tespiti (Object Detection)

- **Tanım:** Görüntü/videoda nesneleri **sınıflandırma + lokalizasyon**
- **Çıktı:** Bounding Box + Class
- **Farklar:**
  - **Object Classification:** Tek sınıf → "Bu resimde araba var mı?"
  - **Object Detection:** Nesneleri bulma ve etiketleme
  - **Object Segmentation:** Piksel bazlı ayırma (daha maliyetli, daha detaylı)

## 📊 Performans Ölçütleri

- **IoU (Intersection over Union):** Tahmin kutusu ile gerçek kutu kesişim oranı
- **Precision (Kesinlik):** Doğru pozitif / tüm pozitif tahminler
- **Recall (Duyarlılık):** Doğru pozitif / gerçek pozitifler
- **mAP (mean Average Precision):** Çoklu sınıf ortalaması

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🙏 Teşekkürler

- [Ultralytics](https://github.com/ultralytics/yolov5) - YOLOv5 implementasyonu
- [OpenCV](https://opencv.org/) - Görüntü işleme kütüphanesi
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube video indirme

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
