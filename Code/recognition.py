import numpy as np
import cv2
from os import listdir
from os.path import isfile , join


data_path = 'D:/pythontut/smart_attandance/dataset/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]


Traning_data, Lables = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(data_path, file)
    # image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Traning_data.append(np.asarray(images, dtype=np.uint8))
    Lables.append(i)


Lables = np.asarray(Lables, dtype=int)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Traning_data), np.asarray(Lables))

print("dataset model traning completed")

face_classifier = cv2.CascadeClassifier('C:/Users/84097/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:  # Check if no faces found
        return img, None
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h, x:x+w]

        return img,roi
    
    return img, None
    
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    image, face= face_detector(frame)

    try:
        if face is not None:

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                    
            if confidence > 82:
                cv2.putText(image, "Sumit" + " verified", (250,450), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Unknown",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.imshow('Face Cropper')
                pass

    except:
        cv2.putText(image,"",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()


