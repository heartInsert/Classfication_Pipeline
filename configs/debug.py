import os
from PIL import Image

Train_mode = False
current_dir = r'/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification'
height, width = 300, 400
kfold = 5
# model setting
model_entity = dict(
    model_name='efficientnet',
    model_version='efficientnet-b2',
    pretrained=True,
    pretrained_dir=None,
    num_class=5)

# dataset setting
from torchvision import transforms
import getpass
import cv2


def read_from_cv2(data_path):
    image = cv2.imread(data_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def read_from_cv2_and_resize(data_path):
    image = cv2.imread(data_path)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


user_name = getpass.getuser()  # 获取当前用户名
if user_name == 'xjz':
    from codes.Mytransforms import Rand_Augment

    # image = cv2.imread(data_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    # img = self.data_transforms(image)
    dataset_entity = dict(
        data_csv_path='/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/train.csv',
        data_folder_path='/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/train_images_{}_{}'.format(
            height, width),
        dataset_name='classification_dataset',
        train_transforms=transforms.Compose([
            lambda data_path: read_from_cv2(data_path),
            Rand_Augment(max_Magnitude=20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing()
        ]),
        test_transforms=transforms.Compose([
            lambda data_path: read_from_cv2(data_path),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    )

dataset_entity_predict = dict(
    data_folder_path='/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/test_images',
    dataset_name='classification_predict_dataset',
    predict_transforms=transforms.Compose([
        lambda data_path: read_from_cv2_and_resize(data_path),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    num_TTA=2,
)
dataloader_entity = dict(
    batch_size=45,
    shuffle=True if Train_mode else False,
    num_wokers=6 if Train_mode else 0,
    drop_last=False,
)
# Trainer setting
trainer_entity = dict(
    gpus=1,
    max_epochs=3 if Train_mode else 4,
    check_val_every_n_epoch=1,
    deterministic=True,
    amp_level='O2',
    precision=16,
    # default_root_dir=os.getcwd() + '/model_checkpoint/' + model_entity['model_name']
)
# loss setting
loss_fc_entity = dict(
    loss_name="LabelSmoothCEloss",
    loss_args=dict()
)
# optimizer setting
optimzier_entity = dict(
    optimizer_name='ranger',
    optimizer_args=dict(
        lr=1e-3,
        use_gc=False,
    )
)
lrschdule_entity = dict(
    lrschdule_name='polynomial_decay_schedule_with_warmup',
    lrschdule_args=dict(
        num_warmup_steps=1 if Train_mode else 2,
        num_training_steps=trainer_entity['max_epochs'],
        lr_end=1e-6,
        power=1.2,
        last_epoch=-1
    ),
    SWA=dict(
        SWA_enable=True,
        SWA_start_epoch=1,
    )
)
training_way = dict(
    training_way_name='Normal_training',
    # optional  Fimix or cutmix
    training_way_args=dict()
)
swa = False
# logger_setting
logger_entity = dict(
    weight_savepath=os.path.join(current_dir, 'model_weights'),
    ModelCheckpoint=dict(
        monitor='val_Accuracy_epoch',
        verbose=True,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
        period=1,
        prefix="",
    )

)
