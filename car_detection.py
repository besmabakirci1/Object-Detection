import torch
import numpy as np
import cv2
import os
from time import time
import argparse

class CarDetection:
    """
    Araç veri seti için özelleştirilmiş nesne algılama sınıfı.
    Araç türlerini sınıflandırır ve detaylı bilgi verir.
    """

    def __init__(self, source, model_path="yolov5s.pt", out_file="Car_Detection_Output.mp4"):
        """
        Araç algılama sınıfını başlatır.
        :param source: Video kaynağı (dosya yolu veya kamera)
        :param model_path: YOLOv5 model dosyası
        :param out_file: Çıktı video dosyası
        """
        self._SOURCE = source
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.out_file = out_file
        
        # Araç sınıfları ve açıklamaları
        self.car_classes = {
            0: "sedan",      # Binek otomobil
            1: "van",        # Minibüs/Ticari araç
            2: "truck",      # Kamyon
            3: "motorcycle", # Motosiklet
            4: "bus",        # Otobüs
            5: "pickup"      # Kamyonet
        }
        
        # Cihaz belirleme
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("🍎 Apple Silicon (MPS) kullanılıyor.")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("🚀 CUDA GPU kullanılıyor.")
        else:
            self.device = 'cpu'
            print("💻 CPU kullanılıyor.")

    def load_model(self, model_path):
        """
        YOLOv5 modelini yükler.
        :param model_path: Model dosyası yolu
        :return: Yüklenen model
        """
        try:
            # Önce yerel dosyadan yüklemeyi dene
            if os.path.exists(model_path):
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                print(f"✅ Model başarıyla yüklendi: {model_path}")
            else:
                # Varsayılan modeli yükle
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                print("✅ Varsayılan YOLOv5s modeli yüklendi.")
            
            return model
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return None

    def get_car_info(self, class_id, confidence):
        """
        Araç sınıfı hakkında detaylı bilgi verir.
        :param class_id: Sınıf ID'si
        :param confidence: Güven skoru
        :return: Araç bilgisi
        """
        car_info = {
            "sedan": {
                "türkçe": "Binek Otomobil",
                "açıklama": "4-5 kişilik şahsi araç",
                "ortalama_hız": "120 km/s",
                "yakıt_tüketimi": "6-8 L/100km"
            },
            "van": {
                "türkçe": "Minibüs/Ticari Araç",
                "açıklama": "Yolcu ve yük taşıma aracı",
                "ortalama_hız": "100 km/s",
                "yakıt_tüketimi": "8-12 L/100km"
            },
            "truck": {
                "türkçe": "Kamyon",
                "açıklama": "Ağır yük taşıma aracı",
                "ortalama_hız": "80 km/s",
                "yakıt_tüketimi": "25-35 L/100km"
            },
            "motorcycle": {
                "türkçe": "Motosiklet",
                "açıklama": "İki tekerlekli motorlu araç",
                "ortalama_hız": "130 km/s",
                "yakıt_tüketimi": "3-5 L/100km"
            },
            "bus": {
                "türkçe": "Otobüs",
                "açıklama": "Toplu taşıma aracı",
                "ortalama_hız": "70 km/s",
                "yakıt_tüketimi": "20-30 L/100km"
            },
            "pickup": {
                "türkçe": "Kamyonet",
                "açıklama": "Hafif yük taşıma aracı",
                "ortalama_hız": "100 km/s",
                "yakıt_tüketimi": "10-15 L/100km"
            }
        }
        
        class_name = self.car_classes.get(class_id, "bilinmeyen")
        return car_info.get(class_name, {})

    def score_frame(self, frame):
        """
        Tek bir kareyi işler ve araç algılama yapar.
        :param frame: Giriş karesi
        :return: Algılama sonuçları
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        
        # Sonuçları CPU'ya taşı ve numpy'a çevir
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cord = results.xyxyn[0][:, :-1].cpu().numpy()
        
        return labels, cord

    def plot_boxes_with_info(self, results, frame):
        """
        Algılanan araçları çizer ve bilgilerini gösterir.
        :param results: Algılama sonuçları
        :param frame: İşlenecek kare
        :return: Çizilmiş kare
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        # Araç sayacı
        car_count = {}
        
        for i in range(n):
            row = cord[i]
            # Güven puanı 0.3'ten yüksek olan tahminleri al
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                # Araç sınıfını belirle
                class_id = int(labels[i])
                class_name = self.car_classes.get(class_id, "bilinmeyen")
                confidence = row[4]
                
                # Araç sayısını güncelle
                car_count[class_name] = car_count.get(class_name, 0) + 1
                
                # Araç bilgilerini al
                car_info = self.get_car_info(class_id, confidence)
                
                # Renk belirleme (araç türüne göre)
                colors = {
                    "sedan": (0, 255, 0),      # Yeşil
                    "van": (255, 0, 0),        # Mavi
                    "truck": (0, 0, 255),      # Kırmızı
                    "motorcycle": (255, 255, 0), # Cyan
                    "bus": (255, 0, 255),      # Magenta
                    "pickup": (0, 255, 255)    # Sarı
                }
                
                color = colors.get(class_name, (255, 255, 255))
                
                # Kutuyu çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Etiket metni
                label_text = f"{car_info.get('türkçe', class_name)} ({confidence:.2f})"
                
                # Etiket arka planı
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1 + text_width, y1), color, -1)
                
                # Etiket metni
                cv2.putText(frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Araç bilgilerini göster
                info_text = f"Hız: {car_info.get('ortalama_hız', 'N/A')}"
                cv2.putText(frame, info_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Araç sayısı özetini göster
        self.draw_car_summary(frame, car_count)
        
        return frame

    def draw_car_summary(self, frame, car_count):
        """
        Ekranın üst kısmında araç sayısı özetini gösterir.
        :param frame: Kare
        :param car_count: Araç sayıları
        """
        y_offset = 30
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.putText(frame, "🚗 Araç Algılama Özeti", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        for car_type, count in car_count.items():
            car_info = self.get_car_info(list(self.car_classes.keys())[list(self.car_classes.values()).index(car_type)], 0)
            text = f"{car_info.get('türkçe', car_type)}: {count}"
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

    def __call__(self):
        """
        Ana işleme döngüsü.
        """
        # Video yakalayıcı oluştur
        if self._SOURCE.isdigit():
            player = cv2.VideoCapture(int(self._SOURCE))
        else:
            player = cv2.VideoCapture(self._SOURCE)
        
        if not player.isOpened():
            print(f"❌ Video kaynağı açılamadı: {self._SOURCE}")
            return

        # Video özelliklerini al
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = int(player.get(cv2.CAP_PROP_FPS))
        
        # Video yazıcı oluştur
        four_cc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.out_file, four_cc, fps_source, (x_shape, y_shape))
        
        if not out.isOpened():
            print("❌ Video yazıcı oluşturulamadı.")
            player.release()
            return
            
        print("🚗 Araç algılama başlatılıyor...")
        print("📊 Algılanan araç türleri:")
        for class_id, class_name in self.car_classes.items():
            car_info = self.get_car_info(class_id, 0)
            print(f"   {class_id}: {car_info.get('türkçe', class_name)} - {car_info.get('açıklama', '')}")
        
        frame_count = 0
        while True:
            start_time = time()
            ret, frame = player.read()
            if not ret:
                print("✅ Video işleme tamamlandı.")
                break
            
            # Kareyi işle
            results = self.score_frame(frame)
            frame = self.plot_boxes_with_info(results, frame)
            
            # FPS hesapla
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            
            # FPS bilgisini göster
            cv2.putText(frame, f"FPS: {fps:.1f}", (x_shape-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Kare sayısını göster
            frame_count += 1
            cv2.putText(frame, f"Kare: {frame_count}", (x_shape-120, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # İşlenmiş kareyi göster
            cv2.imshow('🚗 Araç Algılama', frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Kullanıcı tarafından durduruldu.")
                break
            
            # İşlenmiş kareyi kaydet
            out.write(frame)

        # Kaynakları serbest bırak
        player.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"💾 Video kaydedildi: {self.out_file}")

def main():
    """
    Ana fonksiyon - komut satırı argümanlarını işler.
    """
    parser = argparse.ArgumentParser(description='Araç Algılama Sistemi')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video kaynağı (dosya yolu, kamera numarası veya URL)')
    parser.add_argument('--model', type=str, default='yolov5s.pt',
                       help='YOLOv5 model dosyası')
    parser.add_argument('--output', type=str, default='Car_Detection_Output.mp4',
                       help='Çıktı video dosyası')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Güven eşiği (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Araç algılama sistemini başlat
    detector = CarDetection(args.source, args.model, args.output)
    detector()

if __name__ == "__main__":
    main()


