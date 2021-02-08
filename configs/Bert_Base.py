import os

Train_mode = True
kfold = 5
# model setting
model_entity = dict(
    model_name='resnet50',
    pretrained=False,
    num_class=10)
# dataset setting
from torchvision import transforms

dataset_entity = dict(
    root='/home/xjz/Desktop/Coding/PycharmProjects/Anything/Pytorch_lighting',
    dataset='MNIST',
    dataset_val='Mnist_predict',
    download=False,
    transform_train=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    transform_test=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
)
dataloader_entity = dict(
    batch_size=300,
    shuffle=True,
    num_wokers=6 if Train_mode else 0,
    drop_last=False,
)
# Trainer setting
trainer_entity = dict(
    gpus=1,
    max_epochs=20 if Train_mode else 3,
    check_val_every_n_epoch=1,
    deterministic=True,
    amp_level='O2',
    precision=16,
    # default_root_dir=os.getcwd() + '/model_checkpoint/' + model_entity['model_name']
)
# loss setting
loss_fc_entity = dict(
    loss_name="CrossEntropyLoss",
    loss_args=dict()
)
# optimizer setting
optimzier_entity = dict(
    optimizer_name='Adam',
    optimizer_args=dict(lr=1e-3)
)
# lr schdule setting
lrschdule_entity = dict(
    lrschdule_name='MultiStepLR',
    # lrschdule_name='My_SWALR',
    lrschdule_args=dict(milestones=[4, 6], gamma=0.1)
)
swa = False
# logger_setting
logger_entity = dict(
    weight_savepath='/home/xjz/Desktop/Coding/PycharmProjects/Anything/Pytorch_lighting/model_weights',
    ModelCheckpoint=dict(
        monitor='valid_acc_epoch',
        verbose=True,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
        period=1,
        prefix="",
    )

)
