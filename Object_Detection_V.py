import torch
import numpy as np
import cv2
import yt_dlp
from time import time
import os
import argparse
from typing import Optional, List
try:
    from car_model_classifier import CarModelClassifier
    CAR_CLASSIFIER_AVAILABLE = True
except ImportError:
    CAR_CLASSIFIER_AVAILABLE = False
    print("Warning: car_model_classifier not found. Car model classification disabled.")

# YouTube videosu işleme için yt-dlp kullanılıyor
# Modern ve güncel bir alternatif olarak pafy yerine yt-dlp tercih edildi

class ObjectDetection:
    def __init__(self, source, out_file="Labeled_Video.avi", enable_model_cls: bool = False,
                 model_cls_weights: Optional[str] = None, model_cls_labels: Optional[List[str]] = None):
        self._SOURCE = source
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.enable_model_cls = enable_model_cls and CAR_CLASSIFIER_AVAILABLE
        self.model_classifier = None
        self._temp_video_file = None  # İndirilen geçici video dosyası
        
        # Otomatik olarak model varsa aktif et
        if not self.enable_model_cls and CAR_CLASSIFIER_AVAILABLE:
            default_weights = "models/car_cls_v1/car_model_classifier.pt"
            default_labels = "models/car_cls_v1/labels.txt"
            if os.path.exists(default_weights):
                model_cls_weights = model_cls_weights or default_weights
                model_cls_labels = model_cls_labels or default_labels
                self.enable_model_cls = True
                print(f"Auto-enabling car model classification with {default_weights}")
        
        if self.enable_model_cls and CAR_CLASSIFIER_AVAILABLE:
            self.model_classifier = CarModelClassifier(weights_path=model_cls_weights, labels=model_cls_labels)
        
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
        # Kaynağın ne olduğu konusunda esnek davran: yerel dosya, webcam id (örn '0'), URL/YouTube
        src = os.path.expanduser(str(self._SOURCE))

        # Yerel dosya var mı?
        if os.path.isfile(src):
            return cv2.VideoCapture(src)

        # Eğer kaynak bir sayısal string veya integer ise webcam ID olarak kullan
        if isinstance(self._SOURCE, int) or (isinstance(self._SOURCE, str) and self._SOURCE.isdigit()):
            try:
                cam_id = int(self._SOURCE)
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    return cap
            except Exception:
                pass

        # URL veya YouTube ise önce yt-dlp ile videoyu indirip yerel olarak kullan
        lower = src.lower()
        if lower.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            try:
                import tempfile
                import uuid
                
                # Geçici dosya adı oluştur
                temp_dir = tempfile.gettempdir()
                temp_video = os.path.join(temp_dir, f"yt_video_{uuid.uuid4().hex[:8]}.mp4")
                
                print(f"Downloading video to temporary file: {temp_video}")
                ydl_opts = {
                    'format': 'best[height<=720]/best',
                    'outtmpl': temp_video.replace('.mp4', '.%(ext)s'),
                    'quiet': False,
                    'noplaylist': True,  # Sadece tek video indir, playlist değil
                    'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([self._SOURCE])
                
                # İndirilen dosyayı bul (ext değişebilir)
                downloaded_file = temp_video.replace('.mp4', '')
                for ext in ['.mp4', '.webm', '.mkv', '.flv']:
                    if os.path.exists(downloaded_file + ext):
                        downloaded_file = downloaded_file + ext
                        break
                
                if os.path.exists(downloaded_file):
                    print(f"Video downloaded successfully: {downloaded_file}")
                    cap = cv2.VideoCapture(downloaded_file)
                    if cap.isOpened():
                        # Geçici dosyayı işlem bitince silmek için sakla
                        self._temp_video_file = downloaded_file
                        return cap
                    else:
                        # Açılamazsa sil
                        try:
                            os.remove(downloaded_file)
                        except:
                            pass
                        
            except Exception as e:
                print(f"YouTube/video could not be downloaded with yt-dlp: {e}")
                import traceback
                traceback.print_exc()

        # Son çare: OpenCV'ye doğrudan ver; bu yerel bir yol veya doğrudan akış URL'si olabilir
        cap = cv2.VideoCapture(self._SOURCE)
        if cap.isOpened():
            return cap

        print("Could not open video source with yt-dlp or OpenCV. Verify path/URL or try a webcam id like '0'.")
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
        # MPS tensörlerini CPU'ya taşı, sonra numpy'a çevir
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        Sayısal etiket değerine karşılık gelen metin etiketini döndürür.
        :param x: Sayısal etiket
        :return: Karşılık gelen metin etiket
        """
        return self.classes[int(x)]

    def get_vehicle_color(self, label):
        """Araç türüne göre renk döndürür"""
        vehicle_colors = {
            'car': (0, 0, 255),        # 🔴 Kırmızı
            'truck': (255, 165, 0),    # 🟠 Turuncu
            'bus': (128, 0, 128),      # 🟣 Mor
            'motorcycle': (255, 20, 147), # 🌸 Pembe
            'bicycle': (0, 255, 255),  # 🔵 Cyan
            'airplane': (255, 215, 0), # 🟡 Altın
            'train': (0, 128, 0),      # 🟢 Koyu Yeşil
        }
        return vehicle_colors.get(label, (0, 255, 0))  # Varsayılan: Yeşil

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
                base_label = self.class_to_label(labels[i])
                confidence = row[4]
                
                # Araç türüne göre renk seç
                color = self.get_vehicle_color(base_label)
                
                # Daha kalın ve güzel kutular
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                display_label = base_label
                # Eğer ikinci aşama model sınıflandırma aktifse ve nesne 'car' ise kırpıp sınıflandır
                if self.enable_model_cls and self.model_classifier is not None and base_label == "car":
                    # Kırpma sınırlarını güvenli hale getir
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(x_shape, x2), min(y_shape, y2)
                    # Küçük kutuları atla (gürültüyü azaltmak için)
                    if (x2c - x1c) >= 64 and (y2c - y1c) >= 64 and x2c > x1c and y2c > y1c:
                        crop = frame[y1c:y2c, x1c:x2c]
                        try:
                            make_model, conf = self.model_classifier.predict(crop)
                            # Yeterli güven yoksa yalnızca temel etiketi göster
                            if conf >= 0.50:
                                display_label = f"{make_model} ({conf:.1%})"
                            else:
                                display_label = f"{base_label} ({confidence:.1%})"
                        except Exception:
                            display_label = f"{base_label} ({confidence:.1%})"
                else:
                    # Araç türüne göre güven puanı ekle
                    display_label = f"{base_label} ({confidence:.1%})"
                
                # Daha güzel font ve boyut
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                thickness = 2
                
                # Metin boyutunu hesapla
                (text_width, text_height), baseline = cv2.getTextSize(display_label, font, font_scale, thickness)
                
                # Metin arka planı için dikdörtgen
                cv2.rectangle(frame, (x1, y1-text_height-baseline-10), (x1+text_width+10, y1), color, -1)
                
                # Metni beyaz renkte yaz
                cv2.putText(frame, display_label, (x1+5, y1-5), font, font_scale, (255, 255, 255), thickness)
                
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
        fps_source = player.get(cv2.CAP_PROP_FPS)
        # Bazı kaynaklar FPS bilgisini döndüremeyebilir (0 veya NaN). Güvenli bir fallback kullan.
        try:
            fps_source = float(fps_source)
            if fps_source <= 0 or np.isnan(fps_source):
                print("Warning: source FPS could not be determined, defaulting to 25 FPS.")
                fps_source = 25.0
        except Exception:
            print("Warning: source FPS invalid, defaulting to 25 FPS.")
            fps_source = 25.0
        
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
            # FPS bilgisini daha güzel göster
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            print(f"Frames Per Second : {fps}")
            
            # İşlenmiş kareyi ekranda göster
            cv2.imshow('Object Detection', frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # İşlenmiş kareyi çıktı dosyasına yaz
            out.write(frame)

        # İşlem bitince kaynakları serbest bırak
        player.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Geçici video dosyasını sil
        if self._temp_video_file and os.path.exists(self._temp_video_file):
            try:
                os.remove(self._temp_video_file)
                print(f"Temporary video file deleted: {self._temp_video_file}")
            except Exception as e:
                print(f"Could not delete temporary file: {e}")
        
        print(f"Video saved to {self.out_file}")

def load_labels_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Warning: labels file not found: {path}")
        return None
    labels: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels



def main():
    parser = argparse.ArgumentParser(description="YOLOv5 + Opsiyonel Araç Modeli Sınıflandırma")
    parser.add_argument('--source', type=str, default='https://youtube.com/shorts/hPsH7GJEQjg?si=phmcvmIuJfa2C_WM', help='Video kaynağı (dosya yolu veya URL)')
    parser.add_argument('--output', type=str, default='YouTube_Labeled_Video.mp4', help='Çıktı video dosyası')
    parser.add_argument('--enable-model-cls', action='store_true', help='Araç marka/model sınıflandırmasını etkinleştir')
    parser.add_argument('--model-cls-weights', type=str, default=None, help='Araç marka/model sınıflandırma ağırlıkları (TorchScript .pt önerilir)')
    parser.add_argument('--model-cls-labels', type=str, default=None, help='Sınıflandırıcı için etiket dosyası (satır başına bir etiket)')
    args = parser.parse_args()

    labels = load_labels_file(args.model_cls_labels) if args.enable_model_cls else None
    detector = ObjectDetection(
        args.source,
        out_file=args.output,
        enable_model_cls=args.enable_model_cls,
        model_cls_weights=args.model_cls_weights,
        model_cls_labels=labels
    )
    detector()


if __name__ == '__main__':
    main()