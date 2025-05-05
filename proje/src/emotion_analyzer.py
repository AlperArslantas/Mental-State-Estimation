import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from body_language_analyzer import BodyLanguageAnalyzer
from tensorflow.keras.models import load_model
import unicodedata

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_data = []
        self.timestamps = []
        self.body_analyzer = BodyLanguageAnalyzer()
        # Eğitilmiş modeli yükle
        self.model = load_model('models/emotion_cnn_foldered.h5')
        # İngilizce etiketleri Türkçeye çevir
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_labels_tr = {
            'angry': 'Sinirli',
            'disgust': 'Igrenme',
            'fear': 'Korku',
            'happy': 'Mutlu',
            'neutral': 'Notr',
            'sad': 'Uzgun',
            'surprise': 'Saskin'
        }
        # Haar Cascade ile yüz tespiti için model yükle
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def _remove_turkish_chars(self, text):
        # Türkçe karakterleri İngilizce karşılıklarıyla değiştir
        replace_map = str.maketrans('çğıöşüÇĞİÖŞÜ', 'cgiosuCGIOSU')
        return text.translate(replace_map)

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        emotion_result = {label: 0.0 for label in self.emotion_labels}
        # Sadece ilk yüzü analiz et
        for (x, y, w, h) in faces[:1]:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)
            preds = self.model.predict(face_img, verbose=0)[0]
            for i, label in enumerate(self.emotion_labels):
                emotion_result[label] = float(preds[i]) * 100
            # Sonuçları ekrana yaz
            for i, (emotion, value) in enumerate(emotion_result.items()):
                label_tr = self.emotion_labels_tr[emotion]
                label_tr = self._remove_turkish_chars(label_tr)
                cv2.putText(frame, f"{label_tr}: {value:.2f}", (10, 110 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Yüzü kutu ile göster
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            break
        # Vücut dili analizi
        frame = self.body_analyzer.analyze_pose(frame)
        self.emotion_data.append(emotion_result)
        self.timestamps.append(datetime.now())
        return frame

    def visualize_results(self):
        if not self.emotion_data:
            return
        plt.figure(figsize=(12, 6))
        emotions = list(self.emotion_data[0].keys())
        # Her duygunun ortalamasını hesapla
        emotion_means = {emotion: np.mean([data[emotion] for data in self.emotion_data]) for emotion in emotions}
        # En yüksek ortalamaya sahip 4 duyguyu seç
        top_emotions = sorted(emotion_means, key=emotion_means.get, reverse=True)[:4]
        colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0'
        ]
        # Zamanı saniye cinsinden hesapla
        if self.timestamps:
            t0 = self.timestamps[0]
            seconds = [(t - t0).total_seconds() for t in self.timestamps]
        else:
            seconds = list(range(len(self.emotion_data)))
        for i, emotion in enumerate(top_emotions):
            values = [data[emotion] for data in self.emotion_data]
            label_tr = self._remove_turkish_chars(self.emotion_labels_tr[emotion])
            plt.plot(
                seconds, values, label=label_tr,
                color=colors[i % len(colors)], linewidth=2.5
            )
        plt.title('En Baskın 4 Duygunun Zaman İçindeki Değişimi', fontsize=16)
        plt.xlabel('Geçen Süre (saniye)', fontsize=13)
        plt.ylabel('Duygu Skoru (%)', fontsize=13)
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('emotion_analysis_simple.png')
        plt.close()

    def get_statistics(self):
        if not self.emotion_data:
            return "Veri yok"
        emotion_stats = {}
        for emotion in self.emotion_data[0].keys():
            values = [data[emotion] for data in self.emotion_data]
            emotion_stats[emotion] = {
                'ortalama': np.mean(values),
                'maksimum': np.max(values),
                'minimum': np.min(values)
            }
        body_stats = self.body_analyzer.get_statistics()
        return {
            'emotion_stats': emotion_stats,
            'body_stats': body_stats,
            'total_frames': len(self.emotion_data)
        }

def main():
    analyzer = EmotionAnalyzer()
    cap = cv2.VideoCapture(0)
    print("Analiz başlatılıyor... (Çıkmak için 'q' tuşuna basın)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = analyzer.analyze_frame(frame)
        cv2.imshow('Duygu ve Vücut Dili Analizi', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    analyzer.visualize_results()
    stats = analyzer.get_statistics()
    print("\nAnaliz İstatistikleri:")
    print("----------------------")
    print(f"Toplam Kare: {stats['total_frames']}")
    print("\nDuygu İstatistikleri:")
    for emotion, values in stats['emotion_stats'].items():
        print(f"{emotion}:")
        print(f"  Ortalama: {values['ortalama']:.2f}")
        print(f"  Maksimum: {values['maksimum']:.2f}")
        print(f"  Minimum: {values['minimum']:.2f}")
    print("\nVücut Dili İstatistikleri:")
    for key, value in stats['body_stats'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 