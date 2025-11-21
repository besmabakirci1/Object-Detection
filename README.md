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
git clone https://github.com/kullaniciadi/objectDetectionPart1.git
cd objectDetectionPart1
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
objectDetectionPart1/
├── Object_Detection_V.py          # Ana uygulama
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
