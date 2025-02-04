import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Görselleri yükleyip işleyen fonksiyon
def load_images_from_directory(directory, label, phase="Eğitim"):
    data = []
    file_list = os.listdir(directory)
    for idx, filename in enumerate(file_list):
        img_path = os.path.join(directory, filename)
        print(f"Şu an {phase} aşamasında {idx + 1}. görsel işleniyor: {filename}")


        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        data.append((img.flatten(), label))
    return data


# Eğitim ve doğrulama veri yolları
train_path = 'dataset/Training'
val_path = 'dataset/Validation'

# Eğitim verilerini yükle
train_data = load_images_from_directory(os.path.join(train_path, 'male'), 0, phase="Eğitim (Erkek)") + \
             load_images_from_directory(os.path.join(train_path, 'female'), 1, phase="Eğitim (Kadın)")

# Doğrulama verilerini yükle
val_data = load_images_from_directory(os.path.join(val_path, 'male'), 0, phase="Doğrulama (Erkek)") + \
           load_images_from_directory(os.path.join(val_path, 'female'), 1, phase="Doğrulama (Kadın)")


X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)


print("\nModel eğitiliyor...")
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
print("Model eğitildi!")

# Doğrulama setinde modeli test et
print("\nDoğrulama verileri üzerinde test ediliyor...")
y_pred = model.predict(X_val)

# Sonuçları yazdır
print("\nDoğruluk Oranı:", accuracy_score(y_val, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_val, y_pred))


model_path = 'svm_gender_classifier.pkl'
joblib.dump(model, model_path)
print(f"\nModel '{model_path}' olarak kaydedildi!")


loaded_model = joblib.load(model_path)
print("\nKaydedilen model başarıyla yüklendi ve kullanılabilir.")