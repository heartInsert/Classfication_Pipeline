import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import json
import torch
import copy
import re
import jionlp as jio


def clean_text_original(text):
    res = jio.clean_text(text)
    res = jio.remove_qq(res)
    res = jio.remove_email(res)
    res = jio.remove_phone_number(res)
    res = jio.remove_url(res)
    res = jio.remove_id_card(res)
    res = jio.remove_exception_char(res)
    res = res.replace(' ', "")
    if len(res) == 0:
        res = '腿汇'
    return res


#
# class HouseBargin_dataset(data.Dataset):
#     def __init__(self, flag: str, kwargs):
#         assert flag in ['train', 'test']
#         self.kwargs = kwargs
#         self.tokenizer = kwargs['tokenizer']
#         if flag == 'train':
#             self.data_csv = kwargs['train_csv']
#         else:
#             self.data_csv = kwargs['test_csv']
#         self.encode_input = []
#         self.encode_cls = []
#         self.encode_csv()
#
#     def encode_csv(self):
#         for index, row in self.data_csv.iterrows():
#             encode_data = self.tokenizer.encode_plus(clean_text(row['query_str']), clean_text(row['reply_str']),
#                                                      add_special_tokens=True, padding='max_length', max_length=260)
#             self.encode_input.append(encode_data.data)
#             self.encode_cls.append(row['reply_cls'])
#
#     def __getitem__(self, item):
#         input = self.encode_input[item]
#         input_ids = input['input_ids']
#         token_type_ids = input['token_type_ids']
#         attention_mask = input['attention_mask']
#         cls = self.encode_cls[item]
#         return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(cls)
#
#     def __len__(self):
#         return len(self.encode_input)

from random import sample
import synonyms
from datetime import datetime

class HouseBargin_dataset(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['train', 'test']
        self.kwargs = kwargs
        self.tokenizer = kwargs['tokenizer']
        if flag == 'train':
            self.data_csv = kwargs['train_csv']
        else:
            self.data_csv = kwargs['test_csv']
        self.encode_input = []
        self.encode_cls = []
        self.replaceable_tag = ['a', 'n', 'v', 't']
        # self.encode_csv()

    def synonym_replace(self, text):

        split_words, tags = synonyms.seg(text)
        index = [i for i in range(len(tags)) if tags[i] in self.replaceable_tag]
        if len(index) > 0:
            choose_word_index = sample(index, 1)[0]
            choose_word = split_words[choose_word_index]
            nearby_words, nearby_score = synonyms.nearby(choose_word, size=4)
            nearby_index = [i for i in range(len(nearby_score)) if nearby_score[i] > 0.7]
            if len(nearby_index) > 0:
                choose_nearby = sample(nearby_index, 1)[0]
                split_words[choose_word_index] = nearby_words[choose_nearby]

        return ''.join(split_words)

    def clean_text(self, text):
        res = jio.clean_text(text)
        res = jio.remove_qq(res)
        res = jio.remove_email(res)
        res = jio.remove_phone_number(res)
        res = jio.remove_url(res)
        res = jio.remove_id_card(res)
        res = jio.remove_exception_char(res)
        res = res.replace(' ', "")
        if len(res) == 0:
            res = '腿汇'
        # 同义词替换
        res = self.synonym_replace(res)

        #
        return res

    def encode_csv(self):
        for index, row in self.data_csv.iterrows():
            encode_data = self.tokenizer.encode_plus(self.clean_text(row['query_str']),
                                                     self.clean_text(row['reply_str']),
                                                     add_special_tokens=True, padding='max_length', max_length=260)
            self.encode_input.append(encode_data.data)
            self.encode_cls.append(row['reply_cls'])

    def __getitem__(self, item):
        row = self.data_csv.iloc[item]
        start = datetime.now()
        encode_input = self.tokenizer.encode_plus(self.clean_text(row['query_str']), self.clean_text(row['reply_str']),
                                                  add_special_tokens=True, padding='max_length', max_length=260)
        end = datetime.now()
        input = encode_input
        input_ids = input['input_ids']
        token_type_ids = input['token_type_ids']
        attention_mask = input['attention_mask']
        cls = row['reply_cls']
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(cls)

    def __len__(self):
        return len(self.data_csv)


# rewrite the datast when  you  doing  prediction
class HouseBargin_dataset_Predict(data.Dataset):
    def __init__(self, flag: str, kwargs):
        assert flag in ['val']
        self.kwargs = kwargs
        self.tokenizer = kwargs['tokenizer']
        self.data_csv = kwargs['val_csv']
        self.encode_input = []
        self.encode_csv()

    def encode_csv(self):
        maxlen = 0
        for index, row in self.data_csv.iterrows():
            encode_data = self.tokenizer.encode_plus(self.clean_text(row['query_str']),
                                                     self.clean_text(row['reply_str']),
                                                     add_special_tokens=True, padding='max_length', max_length=260)
            if len(encode_data['input_ids']) > maxlen:
                maxlen = len(encode_data['input_ids'])
            self.encode_input.append(encode_data.data)
        pass

    def __getitem__(self, item):
        input = self.encode_input[item]
        input_ids = input['input_ids']
        token_type_ids = input['token_type_ids']
        attention_mask = input['attention_mask']
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)

    def __len__(self):
        return len(self.encode_input)

    def clean_text(self, text):
        res = jio.clean_text(text)
        res = jio.remove_qq(res)
        res = jio.remove_email(res)
        res = jio.remove_phone_number(res)
        res = jio.remove_url(res)
        res = jio.remove_id_card(res)
        res = jio.remove_exception_char(res)
        res = res.replace(' ', "")
        if len(res) == 0:
            res = '腿汇'
        return res
