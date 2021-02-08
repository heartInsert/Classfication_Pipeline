from transformers import BertTokenizerFast

token_dict = {
    'BERT_chinese': BertTokenizerFast,
}


def tokenizer_call(kwargs):
    tokenizer_name = kwargs['model_name']
    assert tokenizer_name in token_dict.keys()
    tokenizer = token_dict[tokenizer_name].from_pretrained(pretrained_model_name_or_path=kwargs['pretrained_dir'])
    return tokenizer
