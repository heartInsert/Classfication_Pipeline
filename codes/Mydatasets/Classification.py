import torch.utils.data as data
import torch
import os
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# train_transform = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=160),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#         A.RandomCrop(height=128, width=128),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )

# img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
# img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
#     # put it from HWC to CHW format
#     img = img.permute((2, 0, 1)).contiguous()
#     if isinstance(img, torch.ByteTensor):
#         return img.float().div(255)
class classification_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['train', 'test']
        self.kwargs = kwargs
        self.data_csv, self.data_transforms = None, None

        if flag == 'train':
            self.flag_train(kwargs)
        else:
            self.flag_test(kwargs)
        self.data_folder_path = kwargs['data_folder_path']

    def flag_train(self, kwargs):
        self.data_csv = kwargs['train_csv']
        # 注意没有ToTensor
        self.data_transforms = kwargs['train_transforms']

    def flag_test(self, kwargs):
        self.data_csv = kwargs['test_csv']
        self.data_transforms = kwargs['test_transforms']

    def __getitem__(self, item):
        row = self.data_csv.iloc[item]
        data_path = os.path.join(self.data_folder_path, row['image_id'])
        label = row['label']
        img = self.data_transforms(data_path)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.data_csv)


class classification_predict_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        self.kwargs = kwargs
        self.data_csv, self.data_transforms = None, None
        self.flag_predict(kwargs)
        self.data_folder_path = kwargs['data_folder_path']

    def flag_predict(self, kwargs):
        self.data_csv = kwargs['predict_csv']
        self.data_transforms = kwargs['predict_transforms']

    def __getitem__(self, item):
        row = self.data_csv.iloc[item]
        data_path = os.path.join(self.data_folder_path, row['image_id'])
        img = self.data_transforms(data_path)
        return img

    def __len__(self):
        return len(self.data_csv)
