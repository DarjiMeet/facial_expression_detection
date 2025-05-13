import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model_training import train_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import dlib

def get_class_indices(train_dir="./facial_expression/train"):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=1,
        class_mode='categorical'
    )
    return generator.class_indices

model = load_model('expression_model.keras')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class_indices = get_class_indices()
emotional_label = [label for label, idx in sorted(class_indices.items(), key=lambda item: item[1])]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            x,y,x1,y1 = face.left(), face.top(), face.right(), face.bottom()

            cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)

            landmarks = predictor(gray,face)
            for n in range(0,68):
                cx = landmarks.part(n).x
                cy = landmarks.part(n).y
                cv2.circle(frame, (cx,cy), 1, (255,0,0), -1)


            roi = gray[y:y1, x:x1]
            roi = cv2.resize(roi, (48,48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = emotional_label[np.argmax(preds)]


            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow('FER Demo', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
cap.release()
cv2.destroyAllWindows()
