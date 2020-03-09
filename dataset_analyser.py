import pickle
import cv2
INSPECTION_PATH = 'downloaded_datasets/Dataset0.pkl'
with open(INSPECTION_PATH, 'rb') as f:
    dataset = pickle.load(f)
for item in dataset:
    cv2.imshow('eye_region', item[0])
    print(str(item[1])+", "+str(item[2]))
    cv2.waitKey(0)
