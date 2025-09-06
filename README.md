# 🚀 Object Detection 101 – To Do & Knowledge Document

## 📌 To Do List

* **n8n self-hosted**

  * Starter Kit kurulumu
  * Ollama – LLM entegrasyonu
* **OpenCV çalışmaları**

  * Object Detection (Nesne Tespiti)
  * Video üzerinde gerçek zamanlı (GPU / CPU karşılaştırması)
* **YOLOv5 (CUDA / OpenCL)**

  * Performans testleri
  * Genişletme: Araç markası & modeli detaylı analiz (fine-grained classification)

---

## 📂 Dataset Seçenekleri – Araç Model Tespiti

### 1. Stanford Cars Dataset

* **İçerik:** 16,185 görsel, 196 sınıf.
* **Detay:** Marka + model + yıl (ör. *2012 Tesla Model S*).
* **Ekstra:** Bounding box + sınıf etiketleri mevcut.
* **Amaç:** İnce ayrım (fine-grained classification).

### 2. VeRi-776

* **İçerik:** 49,357 görsel, 776 araç, 20 kamera.
* **Ekstra:** Bounding box, marka, tip, renk etiketleri.
* **Amaç:** Araç yeniden tanıma (Re-ID), trafik senaryolarında takip.

### 3. Vehicle Dataset for YOLO

* **İçerik:** 3,000 görsel, 3,830 nesne.
* **Sınıflar:** `car`, `threewheel`, `bus`, `truck`, `motorbike`, `van`.
* **Amaç:** YOLO için hızlı başlangıç – genel araç tespiti.

### 4. Vehicle Images Dataset (Mendeley Data)

* **İçerik:** 3,847 görsel.
* **Sınıflar:** 48 araç modeli.
* **Amaç:** Küçük ama iyi etiketlenmiş dataset (marka & model tanıma).

### 5. Roboflow Car Model Detection

* **Kaynak:** [Roboflow Universe](https://universe.roboflow.com/mxk/car-model-detection/dataset/1)
* **İndirme Komutu:**

```bash
curl -L "https://universe.roboflow.com/ds/FVQJTmNQ5U?key=LaeWMqO6ju" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip
```

---

## 📖 Nesne Tespiti (Object Detection)

* **Tanım:** Görüntü/videoda nesneleri **sınıflandırma + lokalizasyon**.
* **Çıktı:** Bounding Box + Class.
* **Farklar:**

  * **Object Classification:** Tek sınıf → “Bu resimde araba var mı?”
  * **Object Detection:** Nesneleri bulma ve etiketleme.
  * **Object Segmentation:** Piksel bazlı ayırma (daha maliyetli, daha detaylı).

---

## 📊 Performans Ölçütleri

* **IoU (Intersection over Union):** Tahmin kutusu ile gerçek kutu kesişim oranı.
* **Precision (Kesinlik):** Doğru pozitif / tüm pozitif tahminler.
* **Recall (Duyarlılık):** Doğru pozitif / gerçek pozitifler.
* **mAP (mean Average Precision):** Çoklu sınıf ortalaması.

---

## 🕰️ Tarihçe

* **1970’ler:** Basit kenar / köşe tabanlı yöntemler.
* **2001 – Viola-Jones:** Yüz tanıma (akıllı telefonlarda yaygın).
* **2005 – HOG (Histogram of Oriented Gradients):** İnsan algılama.
* **2012 – AlexNet:** CNN devrimi (ImageNet).
* **2014 – R-CNN, Fast R-CNN, Faster R-CNN:** Bölge tabanlı, daha hızlı.
* **2015 – YOLO (You Only Look Once):** Tek geçişte gerçek zamanlı nesne tespiti.
* **Günümüz – YOLOv8:** En gelişmiş sürümlerden biri (Ultralytics).

---

## ⚙️ OpenCV & GPU Kullanımı

* **OpenCV-Python ile YOLO Entegrasyonu.**
* **MacOS’ta CUDA/OpenCL sınırlamaları var** (M3/M1 çiplerde NVIDIA GPU yok).
* Alternatif:

  * **CPU inference:** Daha yavaş.
  * **GPU inference (CUDA destekli NVIDIA kartlarda):** Gerçek zamanlı yüksek FPS.

📌 Ultralytics repo: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

---

## 🔬 Gelişmiş Modeller

* **YOLO5 / YOLOv8:** Araçları tespit eder → markaya özel eğitilebilir.
* **SAM2 (Facebook):** Segment Anything Model – piksel bazlı segmentasyon.

  * Repo: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

---

25 Aug 2025: Kodu modifiye edip macte çalışacak hale sokman lazım dedi: Müh 101 Step By Step.
for that why we start from begining .


Last login: Fri Aug 22 16:42:43 on ttys014
/Users/basmabakirci/.zshrc:4: command not found: pyenv
pip install pafy
(base) basmabakirci@Basma-MacBook-Pro ~ % pip install pafy
Collecting pafy
  Using cached pafy-0.5.5-py2.py3-none-any.whl.metadata (10 kB)
Using cached pafy-0.5.5-py2.py3-none-any.whl (35 kB)
Installing collected packages: pafy
Successfully installed pafy-0.5.5
(base) basmabakirci@Basma-MacBook-Pro ~ % pip install pafy

Requirement already satisfied: pafy in ./miniconda3/lib/python3.13/site-packages (0.5.5)
(base) basmabakirci@Basma-MacBook-Pro ~ % pip install yt-dlp

Collecting yt-dlp
  Downloading yt_dlp-2025.8.22-py3-none-any.whl.metadata (175 kB)
Downloading yt_dlp-2025.8.22-py3-none-any.whl (3.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 1.4 MB/s eta 0:00:00
Installing collected packages: yt-dlp
Successfully installed yt-dlp-2025.8.22
(base) basmabakirci@Basma-MacBook-Pro ~ % 

import torch
import numpy as np
import cv2
import pafy
from time import time
import os

# YouTube videosu işleme için pafy ve youtube_dl veya yt-dlp gerekli
# Eğer pafy ile sorun yaşarsanız, YouTube URL'lerini doğrudan işlemek yerine
# önce video dosyasını indirip yerel olarak kullanmak daha stabil bir çözüm olabilir.

class ObjectDetection:
    """
    Yolo5 modelini kullanarak bir video akışından veya dosyasından nesne algılama yapar.
    """

    def __init__(self, source, out_file="Labeled_Video.avi"):
        """
        Sınıfı başlatır.
        :param source: Video kaynağı. Bu bir YouTube URL'si veya yerel dosya yolu olabilir.
        :param out_file: Kaydedilecek çıktı dosyasının adı.
        """
        self._SOURCE = source
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        
        # Cihaz belirleme: macOS'ta M-serisi çipler için 'mps' kullanın,
        # aksi takdirde 'cuda' veya 'cpu'ya geri dönün.
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using Apple Silicon (MPS) for GPU acceleration.")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("Using CUDA for GPU acceleration.")
        else:
            self.device = 'cpu'
            print("Using CPU.")

    def get_video_from_source(self):
        """
        Video kaynağına göre bir OpenCV video yakalama nesnesi oluşturur.
        Yerel dosya veya YouTube URL'si olabilir.
        :return: opencv2 video yakalama nesnesi.
        """
        # Kaynağın bir dosya yolu mu yoksa URL mi olduğunu kontrol et
        if os.path.isfile(self._SOURCE):
            return cv2.VideoCapture(self._SOURCE)
        else:
            try:
                # YouTube URL'si ise pafy ile stream'i al
                play = pafy.new(self._SOURCE).streams[-1]
                return cv2.VideoCapture(play.url)
            except Exception as e:
                print(f"YouTube video stream could not be loaded: {e}")
                return None

    def load_model(self):
        """
        PyTorch Hub'dan YOLOv5 modelini yükler.
        :return: Eğitilmiş PyTorch modeli.
        """
        # YOLOv5 modelini PyTorch Hub'dan yükle
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Tek bir kareyi alır ve YOLOv5 modeli ile nesne algılama yapar.
        :param frame: Giriş karesi.
        :return: Model tarafından algılanan nesnelerin etiketleri ve koordinatları.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        Sayısal etiket değerine karşılık gelen metin etiketini döndürür.
        :param x: Sayısal etiket
        :return: Karşılık gelen metin etiket
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Kare üzerine algılanan nesnelerin kutularını ve etiketlerini çizer.
        :param results: Modelin tahmin sonuçları.
        :param frame: İşlenen kare.
        :return: Çizim yapılmış kare.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # Güven puanı 0.2'den yüksek olan tahminleri al
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0) # Yeşil renk
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def __call__(self):
        """
        Ana döngüyü çalıştırır. Video karesini okur, işler ve çıktı dosyasına yazar.
        """
        player = self.get_video_from_source()
        if player is None or not player.isOpened():
            print("Error: Could not open video source.")
            return

        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = int(player.get(cv2.CAP_PROP_FPS))
        
        # macOS uyumlu bir codec seçin (örn. 'mp4v' veya 'DIVX')
        four_cc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.out_file, four_cc, fps_source, (x_shape, y_shape))
        
        if not out.isOpened():
            print("Error: Could not create video writer. Check file permissions or codec.")
            player.release()
            return
            
        print("Processing video...")
        while True:
            start_time = time()
            ret, frame = player.read()
            if not ret:
                print("End of video stream.")
                break
            
            # Kareyi işle
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            # Kare hızını ekrana ve videoya yazdırabilirsiniz
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print(f"Frames Per Second : {fps}")
            
            # İşlenmiş kareyi çıktı dosyasına yaz
            out.write(frame)

        # İşlem bitince kaynakları serbest bırak
        player.release()
        out.release()
        print(f"Video saved to {self.out_file}")

# Kullanım örneği:
# Eğer yerel dosya kullanacaksanız:
# a = ObjectDetection('/Users/kullaniciadi/Desktop/gece_yagmurlu.mkv')
# Eğer YouTube URL'si kullanacaksanız:
a = ObjectDetection('https://www.youtube.com/watch?v=obw4rbNqWK0', out_file="YouTube_Labeled_Video.mp4")
a()

