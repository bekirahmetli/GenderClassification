import cv2
import numpy as np
import joblib

# Eğitilmiş modelin yolu
model_path = 'svm_gender_classifier.pkl'

# Modeli yükle
model = joblib.load(model_path)

# Yüz algılama için OpenCV'nin pre-trained Haar Cascade modelini yükleyelim
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    # Gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:

        cv2.putText(frame, "Yuz Algilanmadi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


            face_region = gray[y:y + h, x:x + w]

            # Yüzü 64x64 boyutunda yeniden boyutlandır
            face_resized = cv2.resize(face_region, (64, 64))


            face_flatten = face_resized.flatten().reshape(1, -1)

            # Cinsiyet tahmini
            gender = model.predict(face_flatten)[0]

            # Cinsiyeti ekranda göster
            if gender == 0:
                cv2.putText(frame, "Erkek", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Kadin", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('SvM', frame)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()