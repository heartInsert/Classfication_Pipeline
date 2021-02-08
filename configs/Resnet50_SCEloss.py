import os
from PIL import Image

Train_mode = True
current_dir = r'/home/xjz/Desktop/Coding/PycharmProjects/competition/kaggle/cassava_leaf_disease_classification'
height, width = 300, 400
kfold = 5
# model setting
model_entity = dict(
    model_name='Resnet50',
    pretrained=True,
    pretrained_dir=None,
    num_class=5)

# dataset setting
from torchvision import transforms
import getpass

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
            lambda data_path: Image.open(data_path),
            Rand_Augment(max_Magnitude=20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        test_transforms=transforms.Compose([
            lambda data_path: Image.open(data_path),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    )

dataset_entity_predict = dict(
    data_folder_path='/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/test_images',
    dataset_name='classification_predict_dataset',
    predict_transforms=transforms.Compose([
        lambda data_path: Image.open(data_path),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
dataloader_entity = dict(
    batch_size=65,
    shuffle=True if Train_mode else False,
    num_wokers=6 if Train_mode else 0,
    drop_last=True,
)
# Trainer setting
trainer_entity = dict(
    gpus=1,
    max_epochs=11 if Train_mode else 3,
    check_val_every_n_epoch=1,
    deterministic=True,
    amp_level='O2',
    precision=16,
    # default_root_dir=os.getcwd() + '/model_checkpoint/' + model_entity['model_name']
)
# loss setting
loss_fc_entity = dict(
    loss_name="SCEloss",
    loss_args=dict(alpha=0.1, beta=1.0, num_classes=5)
)
# loss setting
# loss_fc_entity = dict(
#     loss_name="CrossEntropyLoss",
#     loss_args=dict()
# )
# optimizer setting
optimzier_entity = dict(
    optimizer_name='ranger',
    optimizer_args=dict(
        lr=1e-4,
        use_gc=False,
    )
)
# lr schdule setting
# lrschdule_entity = dict(
#     lrschdule_name='Linear_schedule_with_warmup',
#     lrschdule_args=dict(
#         num_warmup_steps=3 if Train_mode else 1,
#         num_training_steps=trainer_entity['max_epochs'],
#         last_epoch=-1
#     )
# )
# lrschdule_entity = dict(
#     lrschdule_name='cosine_schedule_with_warmup',
#     lrschdule_args=dict(
#         num_warmup_steps=3 if Train_mode else 1,
#         num_training_steps=trainer_entity['max_epochs'],
#         num_cycles=0.5,
#         last_epoch=-1
#     )
# )
lrschdule_entity = dict(
    lrschdule_name='polynomial_decay_schedule_with_warmup',
    lrschdule_args=dict(
        num_warmup_steps=3 if Train_mode else 1,
        num_training_steps=trainer_entity['max_epochs']-1,
        lr_end=1e-6,
        power=1.0,
        last_epoch=-1
    )
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
