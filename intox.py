
import cv2
import numpy as np
from tensorflow import keras

new_model = keras.models.load_model("my_model.h5")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        color = (0, 255, 0)  
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            
            final_image = cv2.resize(eye_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)  
            final_image = final_image / 255.0  
            
            predictions = new_model.predict(final_image)

            status = "Sober" if predictions > -3 else "Intoxicated"
            
            if status == "Intoxicated":
                color = (0, 0, 255)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Face Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

