import cv2, os
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    height, width = 456, 456
    root = r'/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification'
    folder_source = os.path.join(root, 'train_images')
    folder_target = folder_source + '_{}_{}'.format(height, width)
    if not os.path.exists(folder_target):
        os.makedirs(folder_target)
    data_csv = pd.read_csv(os.path.join(root, 'train.csv'))
    for index, row in tqdm(data_csv.iterrows(), total=len(data_csv)):
        img_name = row['image_id']
        img_path_source = os.path.join(folder_source, img_name)
        img_path_target = os.path.join(folder_target, img_name)
        if not os.path.exists(img_path_target):
            imgA = cv2.imread(img_path_source)
            imgB = cv2.resize(imgA, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_path_target, imgB)

    print()
