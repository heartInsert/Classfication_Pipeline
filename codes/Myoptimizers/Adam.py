from torch.optim import Adam
from transformers import AdamW


class MyAdam(Adam):
    def __init__(self, kwargs):
        super(MyAdam, self).__init__(**kwargs)


class MyAdamW(AdamW):
    def __init__(self, kwargs):
        super(MyAdamW, self).__init__(**kwargs)
