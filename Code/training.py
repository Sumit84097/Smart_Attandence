import numpy as np
import cv2

from os import listdir
from os.path import isfile , join

data_path = 'D:/pythontut/smart_attandance/dataset/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]


Traning_data, Lables = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Traning_data.append(np.asarray(images, dtype=np.uint8))
    Lables.append(i)


Lables = np.asarray(Lables, dtype=int)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Traning_data), np.asarray(Lables))

print("dataset model traning completed")

