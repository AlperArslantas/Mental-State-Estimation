# Zihin Durumu Analizi Projesi

## Proje Hakkında
Bu proje, gerçek zamanlı olarak yüz ifadeleri ve vücut dili üzerinden duygu analizi yapmayı amaçlar. Python, derin öğrenme ve görüntü işleme teknikleri kullanılarak, kişinin sinirli, mutlu, üzgün, şaşkın gibi temel duyguları ve vücut duruşu tespit edilir. Sonuçlar hem grafik hem de kullanıcı arayüzünde sade ve Türkçe olarak sunulur.

## Özellikler
- **Gerçek zamanlı kamera veya video dosyası ile analiz**
- **Kendi eğitilmiş CNN modeli ile duygu tanıma**
- **MediaPipe ile vücut dili (pose) analizi**
- **Tkinter tabanlı kullanıcı dostu arayüz**
- **Türkçe ve sadeleştirilmiş etiketler**
- **Grafik ve istatistiksel sonuçlar**
- **Veri artırma ve class_weight ile güçlendirilmiş model**

## Kurulum
1. **Projeyi klonlayın:**
   ```bash
   git clone https://github.com/kullaniciadi/zihin-durumu-analizi.git
   cd zihin-durumu-analizi
   ```
2. **Sanal ortam oluşturun ve aktif edin:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Gereksinimleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Veri setini ekleyin:**
   - `data/archive/train` ve `data/archive/test` klasörlerinde duyguya göre ayrılmış resimler olmalı (örn. FER veri seti).

## Kullanım
- **Kamera ile analiz başlatmak için:**
  ```bash
  python3 src/gui_app.py
  ```
  - "Analizi Başlat" ile canlı analiz
  - "Video ile Analiz" ile video dosyası seçip analiz
  - "Grafiği Göster" ile analiz sonrası grafik
  - "Durdur" ile işlemi sonlandırabilirsiniz.

- **Modeli yeniden eğitmek için:**
  ```bash
  python3 src/train_emotion_model.py
  ```
  - Eğitim sonunda model ve karışıklık matrisi kaydedilir.

## Gereksinimler
- Python 3.10+
- OpenCV
- TensorFlow / Keras
- MediaPipe
- scikit-learn
- matplotlib
- pillow

Tüm gereksinimler `requirements.txt` dosyasında listelenmiştir.

## Örnek Görseller
- **Arayüz:**
  ![GUI](docs/gui_example.png)
- **Duygu Analizi Grafiği:**
  ![Grafik](emotion_analysis_simple.png)
- **Karışıklık Matrisi:**
  ![Confusion Matrix](confusion_matrix.png)

## Notlar
- Model dosyaları ve büyük veri setleri `.gitignore` ile hariç tutulmuştur.
- Türkçe karakterler arayüzde sadeleştirilmiştir (ör: "Üzgün" → "Uzgun").
- Proje modülerdir, kolayca yeni özellikler eklenebilir.

## Katkı ve İletişim
Her türlü katkı, öneri ve hata bildirimi için pull request gönderebilir veya issue açabilirsiniz.

---
**Hazırlayan:** Alper Kamil Arslantaş 
**Tarih:** 5 Mayıs 2025