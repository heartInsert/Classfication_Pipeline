from torchvision import transforms as t
from PIL import Image
import matplotlib.pyplot as plt
import datetime
# import albumentations as A
import cv2
import numpy as np
from sklearn.manifold import TSNE

#
data_path = r'C:\Users\Administrator\Desktop\DL_Data\cassava_leaf_disease_classification\train_images\6103.jpg'
# # Declare an augmentation pipeline
# transform = A.Compose([
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])
#
# # Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread(data_path)
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Augment an image
# transformed = transform(image=image)
# transformed_image = transformed["image"]

img = Image.open(data_path)
img_array = np.array(img)
plt.imshow(Image.fromarray(image2))
plt.show()
transform = t.ToTensor()
img = transform(img)
test = t.GaussianBlur(kernel_size=[3, 3])

starttime1 = datetime.datetime.now()
for i in range(500):
    test(img)
endtime1 = datetime.datetime.now()
print((endtime1 - starttime1))

img2 = img.unsqueeze(0).repeat_interleave(500, 0)
starttime2 = datetime.datetime.now()
img3 = test(img2)
endtime2 = datetime.datetime.now()
print((endtime2 - starttime2))
print()

# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
