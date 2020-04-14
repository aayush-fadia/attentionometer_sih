import pickle
import cv2
import os
big_dataset = []
DEST_DIR = 'datasets/'
DATASET_LIST = os.listdir(DEST_DIR)
for filename in DATASET_LIST:
    dataset_path = os.path.join(DEST_DIR, filename)
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    big_dataset += dataset
print("{} Records in big dataset".format(len(big_dataset)))
if(input("Do You want to inspect Dataset? Y/N").lower() == "Y".lower()):
    for image, x, y, shape in big_dataset:
        cv2.imshow("IMAGE", image)
        print("x={}, y={}".format(x, y))
        cv2.waitKey(0)