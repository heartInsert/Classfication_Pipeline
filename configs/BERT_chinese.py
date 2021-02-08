import os

current_dir = '/home/xjz/Desktop/Coding/PycharmProjects/competition/datafoutain/house_bargin'
Train_mode = True
kfold = 5
# model setting
model_entity = dict(
    model_name='BERT_chinese',
    pretrained=True,
    pretrained_dir=r'/home/xjz/Desktop/Coding/Pretrained_path/bert_chinese/',
    num_class=2)

# dataset setting
from torchvision import transforms

dataset_entity = dict(
    root='/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data',
    dataset='House_Bargin',
    dataset_Predict='House_Bargin_Predict',
)
dataloader_entity = dict(
    batch_size=30,
    shuffle=True if Train_mode else False,
    num_wokers=6 if Train_mode else 0,
    drop_last=False,
)
# Trainer setting
trainer_entity = dict(
    gpus=1,
    max_epochs=9 if Train_mode else 3,
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
    optimizer_name='AdamW',
    optimizer_args=dict(
        lr=1e-5,
        weight_decay=1e-5
    )
)
# lr schdule setting
lrschdule_entity = dict(
    lrschdule_name='Linear_schedule_with_warmup',
    lrschdule_args=dict(
        num_warmup_steps=3 if Train_mode else 1,
        num_training_steps=trainer_entity['max_epochs'],
        last_epoch=-1
    )
)
swa = False
# logger_setting
logger_entity = dict(
    weight_savepath=os.path.join(current_dir, 'model_weights'),
    ModelCheckpoint=dict(
        monitor='val_f1_epoch',
        verbose=True,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
        period=1,
        prefix="",
    )

)
