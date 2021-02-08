import pandas as pd
import os
import jionlp as jio


def clean_text(text):
    res = jio.clean_text(text)
    res = jio.remove_qq(res)
    res = jio.remove_email(res)
    res = jio.remove_phone_number(res)
    res = jio.remove_url(res)
    res = jio.remove_id_card(res)
    res = jio.remove_exception_char(res)
    return res


def generate_traindata():
    data_path = '/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data'
    train_query = pd.read_csv(os.path.join(data_path, 'train.query.tsv'), sep='\t', header=None, index_col=None,
                              names=['query_id', 'query_str'])
    train_reply = pd.read_csv(os.path.join(data_path, 'train.reply.tsv'), sep='\t', header=None, index_col=None,
                              names=['query_id', 'reply_id', 'reply_str', 'reply_cls'])
    assert train_query.isna().sum()['query_str'] == 0
    query_str = [train_query[train_query['query_id'] == row['query_id']]['query_str'].iloc[0] for idx, row in
                 train_reply.iterrows()]

    train_reply['query_str'] = query_str
    train_reply = train_reply.fillna('腿汇')
    train_reply.to_csv('/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data/train_conjunction.csv', index=0)
    return train_reply


from tqdm import tqdm


def generate_testdata():
    data_path = '/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data'
    test_query = pd.read_csv(os.path.join(data_path, 'test.query.tsv'), encoding='gbk', sep='\t', header=None,
                             index_col=None,
                             names=['query_id', 'query_str'])
    test_reply = pd.read_csv(os.path.join(data_path, 'test.reply.tsv'), encoding='gbk', sep='\t', header=None,
                             index_col=None,
                             names=['query_id', 'reply_id', 'reply_str'])
    assert test_query.isna().sum()['query_str'] == 0

    query_str = [test_query[test_query['query_id'] == row['query_id']]['query_str'].iloc[0] for idx, row in
                 test_reply.iterrows()]

    test_reply['query_str'] = query_str
    test_reply = test_reply.fillna('腿汇')
    test_reply.to_csv('/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data/test_conjunction.csv', index=0)


def generate_retro_translate():
    from Preworks.retro_translate import retroTranslate, BaiduTranslate
    apis = [BaiduTranslate('zh', 'en'), BaiduTranslate('en', 'zh')]

    train_reply = pd.read_csv('/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data/train_conjunction.csv')

    for idx in tqdm(range(len(train_reply)), total=len(train_reply)):
        train_reply.loc[idx, 'query_str_translate'] = retroTranslate(apis, train_reply.loc[idx, 'query_str'])
        train_reply.loc[idx, 'reply_str_translate'] = retroTranslate(apis, train_reply.loc[idx, 'reply_str'])

    train_reply.to_csv('/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data/train_conjunction.csv', index=0)
    pass


if __name__ == "__main__":
    # generate_traindata()
    generate_retro_translate()
    # generate_testdata()
