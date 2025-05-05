import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Veri seti yolları
TRAIN_DIR = os.path.join('data', 'archive', 'train')
TEST_DIR = os.path.join('data', 'archive', 'test')

# Duygu sınıflarını bul
emotion_labels = sorted(os.listdir(TRAIN_DIR))
print('Duygu sınıfları:', emotion_labels)

# Görüntüleri ve etiketleri oku

def load_images_from_folder(folder, emotion_labels):
    images = []
    labels = []
    for emotion in emotion_labels:
        emotion_folder = os.path.join(folder, emotion)
        for filename in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion)
    return np.array(images), np.array(labels)

print('Eğitim verisi yükleniyor...')
X_train, y_train = load_images_from_folder(TRAIN_DIR, emotion_labels)
print('Test verisi yükleniyor...')
X_test, y_test = load_images_from_folder(TEST_DIR, emotion_labels)

# Normalizasyon ve boyutlandırma
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Etiketleri sayısal değere çevir ve one-hot encode yap
le = LabelEncoder()
y_train_num = le.fit_transform(y_train)
y_test_num = le.transform(y_test)
y_train_cat = to_categorical(y_train_num, num_classes=len(emotion_labels))
y_test_cat = to_categorical(y_test_num, num_classes=len(emotion_labels))

# Modeli oluştur
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Veri artırıcı (augmentation) tanımla
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Sınıf ağırlıklarını otomatik hesapla
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(emotion_labels)),
    y=np.argmax(y_train_cat, axis=1)
)
class_weights = dict(enumerate(class_weights))

# Modeli eğit (augmentation ve class_weight ile)
print('Model eğitiliyor (augmentation ve class_weight ile)...')
history = model.fit(
    train_datagen.flow(X_train, y_train_cat, batch_size=64),
    epochs=30,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weights
)

# Modeli kaydet
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'emotion_cnn_foldered.h5'))
print('Model kaydedildi: models/emotion_cnn_foldered.h5')

# 7. Karışıklık Matrisi (Confusion Matrix) Hesapla ve Görselleştir
print('Test verisi üzerinde tahminler yapılıyor...')
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)
labels = emotion_labels

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Duygu Sınıflandırma Karışıklık Matrisi')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print('Karışıklık matrisi confusion_matrix.png olarak kaydedildi.') 