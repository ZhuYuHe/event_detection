# encoding:utf8
'''
@author: tonyzhu
@contact: tonyzhu@tencent.com
@file: seg.py
@time: 2018/7/11
@modify history: 
                2018/7/12: 修复运行速度过慢的问题
@run environment: python2
@desc: segment Chinese sentences
'''

"""Segment Chinese sentences using QQseg"""
import segmentor_4_python as sp
from tqdm import tqdm
import re
import codecs
import argparse

def add_arguments(parser):
    """Build ArgumentParser."""
    # input and output
    parser.add_argument('--input', help='input file path')
    parser.add_argument('--output', help='output path')
    # entity
    parser.add_argument('--entity', action="store_true", help="whether store entity or not")
    # UNK
    parser.add_argument('--UNK', action="store_true", help="whether build unkonwn words or not")
    parser.add_argument('--minn', type=int, default=0, help="min number word would be named UNK")
    
def load_data(path):
    with open(path, 'rb') as f:
        for line in f:
            yield line

def isdigit(text):
    try:
        float(text)
        return True
    except:
        pass

    try:
        import unicodedata
        unicodedata.numeric(text)
        return True
    except:
        pass

    return False

def init_seg():
    # 认证身份，文件路径为存放private_key.dat文件的目录
    if sp.TCInitSeg('/data/home/zhuyuhe/tools/QQseg-api/data/') != True:
        print "Init segmentor failed!"
        exit(1)
    else:
        print "segmentor Init Successed!"
    # 创建分词句柄, 这里开启了sp.TC_ORG_W，机构名会作为基本粒度词语输出
    handle = sp.TCCreateSegHandle(sp.TC_CRF | sp.TC_ORG_W | sp.TC_PER_W | sp.TC_S2D | sp.TC_T2S | sp.TC_CUT | sp.TC_POS | sp.TC_CUS | sp.TC_RUL)
    return handle

def extract_entity(handle):
    """extract entity and replace it"""
    phrase_len = sp.TCGetPhraseCnt(handle)
    person_set = set()
    org_set = set()
    for i in range(phrase_len):
        entity = sp.TCGetPhraseAt(handle,i)
        phrase_idx = sp.TCGetPhraseTokenAt(handle, i).cls
        if phrase_idx == sp.PHRASE_NAME_IDX or phrase_idx == sp.PHRASE_NAME_FR_IDX:
            person_set.add(entity)
        elif phrase_idx == sp.PHRASE_ORGANIZATION_IDX:
            org_set.add(entity)
    return person_set, org_set

def process_puncts(text):
    """make text clean"""
    rule = re.compile(ur"[^,[] :?\w\d\u4e00-\u9fa5]")
    text = text.replace('：', ':').replace('？', '?').replace('！', '!').replace('，', ',')
    text = text.replace('【', '[').replace('】', ']')
    text = re.sub('^ +', '', text)
    text = re.sub(' +', ' ', text)
    text = rule.sub('', text.decode('utf8'))
    return text

def segment(handle, sen, entity=True):
    """segment a sentence"""
    if sp.TCSegment(handle, sen, len(sen), sp.TC_UTF8) == False:
        print "Segment sentence failed!"
        exit(1)
    else:
        line = ''
        seg_len = sp.TCGetResultCnt(handle)
        if entity == False:
            person_set, org_set = extract_entity(handle)
        else:
            person_set, org_set = set(), set()
        for i in range(seg_len):
            now_word = sp.TCGetWordAt(handle, i)
            if isdigit(now_word):
                now_word = '0'

            if now_word in person_set:
                now_word = 'PER'
            elif now_word in org_set:
                now_word = 'ORG'

            if i == 0:
                line += now_word
            else:
                line = line + ' ' + now_word

        line = process_puncts(line)
        return line

def uninit_seg(handle):
    sp.TCCloseSegHandle(handle)
    sp.TCUnInitSeg()

def main():
    # read arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # init QQseg
    handle = init_seg()

    vocab2cnt = {}

    # segment sentence and save
    with codecs.open(args.output, 'w', encoding='utf8') as f:
        print "start segment..."
        for sen in tqdm(load_data('data_tech_w2v.txt')):
            line = segment(handle, sen, args.entity)
            f.write(line.replace('\n', '') + '\n')

    # replace rare word
    if args.UNK:
        print "building vocab..."
        for line in tqdm(load_data(args.output)):
            for word in line.split():
                if word in vocab2cnt.keys():
                    vocab2cnt[word] += 1
                else:
                    vocab2cnt[word] = 1

        print "replacing rare word..."
        with codecs.open(args.output, 'w', encoding='utf8') as f:
            for line in tqdm(load_data(path)):
                wordlist = line.split()
                for i in range(len(wordlist)):
                    if vocab2cnt[wordlist[i]] <= args.minn:
                        wordlist[i] = 'UNK'
                f.write(' '.join(wordlist) + '\n')

    uninit_seg(handle)

if __name__ == '__main__':
    main()
    
    

    