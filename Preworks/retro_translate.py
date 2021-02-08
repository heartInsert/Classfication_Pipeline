import json
import random
import hashlib
from urllib import parse
import http.client
import pandas as pd


class BaiduTranslate:
    def __init__(self, fromLang, toLang):
        self.url = "/api/trans/vip/translate"
        self.appid = "20201108000611432"  # 申请的账号
        self.secretKey = 'PBlMgcV_LtFgbFfbidjT'  # 账号密码
        self.fromLang = fromLang
        self.toLang = toLang
        self.salt = random.randint(32768, 65536)

    def BdTrans(self, text):
        sign = self.appid + text + str(self.salt) + self.secretKey
        md = hashlib.md5()
        md.update(sign.encode(encoding='utf-8'))
        sign = md.hexdigest()
        myurl = self.url + \
                '?appid=' + self.appid + \
                '&q=' + parse.quote(text) + \
                '&from=' + self.fromLang + \
                '&to=' + self.toLang + \
                '&salt=' + str(self.salt) + \
                '&sign=' + sign
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            html = response.read().decode('utf-8')
            html = json.loads(html)
            if 'error_code' in html.keys():
                return False, ''
            dst = html["trans_result"]
            return True, dst
        except Exception as e:
            return False, e


def retroTranslate(apis, text):
    Tran_text = text
    for api in apis:
        flag, Tran_text = api.BdTrans(Tran_text)  # 要翻译的词组
        if flag is False:
            return text
    return Tran_text


if __name__ == '__main__':
    train_reply = pd.read_csv('/home/xjz/Desktop/Coding/DL_Data/house_bargin/train_data/train_conjunction.csv')
    text = retroTranslate([BaiduTranslate('zh', 'en'), BaiduTranslate('en', 'zh')], '这个19楼的\n这个也是\n')
    pass
