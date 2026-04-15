import cv2
import numpy as np
from tensorflow import keras


def facebox(faceNet,frame):
    # print(frame)
    frameHeight= frame.shape[0]
    frameWidth= frame.shape[1]
    blob= cv2.dnn.blobFromImage(frame,1.0,(300,300),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection =faceNet.forward()
    # print(detection.shape)
    bboxs =[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
    # return detection
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

new_model = keras.models.load_model("my_model.h5")


faceNet = cv2.dnn.readNet(faceModel,faceProto)

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)

padding = 20
while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            
            final_image = cv2.resize(eye_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)  
            final_image = final_image / 255.0  
            
            predictions = new_model.predict(final_image)
            
            status = "Sober" if predictions > -3 else "Intoxicated"
            
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ret, frame = video.read()
    frame, bboxs = facebox(faceNet, frame)
    for bbox in bboxs:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()

        age_index = agePred[0].argmax()

        age_category = ageList[age_index]

        label = "{},{}".format(gender, age_category)

        if age_category.startswith('(0') or age_category.startswith('(4') or age_category.startswith('(8') or age_category.startswith('(15'):
            box_color = (0, 0, 255)  
        else:
            box_color = (0, 255, 0)  

        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), box_color, -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    # cv2.imshow("Capture", frame)
    cv2.imshow('Face Cam', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break  
video.release()
cv2.destroyAllWindows()
