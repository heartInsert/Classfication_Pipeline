from torchvision.models import resnet50
import torch

from transformers import BertTokenizer, BertModel
import torch


class Bert_Chinese_Model(torch.nn.Module):
    def __init__(self, kwargs):
        super(Bert_Chinese_Model, self).__init__()
        self.num_class = kwargs['num_class']
        self.layer1 = BertModel.from_pretrained(kwargs['pretrained_dir'], return_dict=True)
        self.layer2 = torch.nn.Linear(self.layer1.config.hidden_size, self.num_class)
        # self.dropout = torch.nn.Dropout(p=0.2)  # drop out

    def forward(self, kwargs):
        out = self.layer1(input_ids=kwargs['input_ids'], token_type_ids=kwargs['token_type_ids'],
                          attention_mask=kwargs['attention_mask'])
        # last_hidden_state = out.last_hidden_state[:, 0]
        # last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = torch.mean(out.last_hidden_state, dim=1)
        output = self.layer2(last_hidden_state)
        return output
