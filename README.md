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

