import codecs
import math
import os
import random
from collections import Counter

import numpy as np

from data_process.process4attention import process_line
from model.config import Config
from utils.model_utils import PAD, PAD_ID, TRIGGER_MODEL, UNK, UNK_ID


def read_data_sen(fname):
    config = Config()
    res = []
    with codecs.open(fname, 'r', 'utf8') as f:
        for line in f:
            res.append(process_line(line, config))
    return res

def sample(data, neg_num = 200):
    random.shuffle(data)
    res = []
    cnt = 0
    for word, context, label in data:
        if label != "__label__非事件":
            res.append((word, context, label))
        else:
            cnt += 1
            if cnt <= 200:
                res.append((word, context, label))
    return res

def read_data(fnames, model):
    need_trigger = model in TRIGGER_MODEL
    sentences = []
    if not isinstance(fnames, list):
        fnames = [fnames]
    for fname in fnames:
        sentence_num = 0
        num = 0
        print("read data from file {0} ".format(fname))
        with codecs.open(fname, 'r', 'utf8') as f:
            for line in f:
                num += 1
                line = line.rstrip().split('\t')
                # 每一行至少有两个元素，标题和标签
                assert len(line) >= 2, print(fname, num, line)
                if not need_trigger:
                    sen = line[0]
                    label = line[-1]
                    if not sen or len(sen) <= 2:
                        continue
                    else:
                        sample = (sen.split(), label)
                    sentences.append(sample)
                    sentence_num += 1
                else:
                    word, context, label = line
                    context = context.split()
                    sentences.append((word, context, label))
                    sentence_num += 1
            print("Got {0} sentences from file {1}".format(sentence_num, fname))
    print("Got all sentences from training files: {0} sentences.".format(len(sentences)))
    return sentences

def create_vocab_from_pretrained_w2v(path, vocab_path):
    print("Creating vocab...")
    vocab = []
    with codecs.open(path, 'r', 'utf8') as f:
        for line in f.readlines()[1:]:
            word = line.split()[0]
            vocab.append(word)
    unk_idx = vocab.index('UNK')
    tmp = vocab[UNK_ID]
    vocab[UNK_ID] = UNK
    vocab[unk_idx] = tmp 
    word2id = {word: idx for idx, word in enumerate(vocab)}

    if not os.path.exists(vocab_path):
        with codecs.open(vocab_path, 'w', 'utf8') as f:
            for word, idx in word2id.items():
                f.write(word + '\t' + str(idx) + '\n')

def create_tag_vocab_from_data(data, tag_path):
    vocab = set()
    print(data[0])
    if isinstance(data[0][1], str):
        for word_list, label in data:
            vocab.add(label)
    else:
        for word, context, label in data:
            vocab.add(label)
    tag2id = {tag: idx for idx, tag in enumerate(vocab)}
    id2tag = {idx: tag for idx, tag in enumerate(vocab)}

    if not os.path.exists(tag_path):
        with codecs.open(tag_path, 'w', 'utf8') as f:
            for tag, idx in tag2id.items():
                f.write(tag + '\t' + str(idx) + '\n')

def read_vocab(path):
    token2id = dict()
    with codecs.open(path, 'r', 'utf8') as f:
        for line in f:
            line = line.rstrip().split('\t')
            token2id[line[0]] = int(line[1])
    id2token = {idx: token for token, idx in token2id.items()}
    return token2id, id2token

def convert_sentence(sentence, word2id, model):
    """
    convert a sentence(list of word) to ids(list of id);
    """
    # assert isinstance(sentences[0], list) == True, print("sentence format error")
    res = []
    modelname = model.name
    config = model.config
    if modelname == "textcnn":
        word_id_list = [word2id.get(word, UNK_ID) for word in sentence]
        word_id_list = word_id_list[:config.sentence_length]
        padding = [PAD_ID] * (config.sentence_length - len(word_id_list))
        res.append(word_id_list + padding)
    if modelname == "attention":
        word_list = sentence
        if len(word_list) > config.sentence_length:
            word_list = word_list[:config.sentence_length]
        else:
            word_list = word_list + [PAD] * (config.sentence_length - len(word_list))
        word_id_list = [word2id.get(word, UNK_ID) for word in word_list]
        for word_idx in range(len(word_list)):
            word_id = word_id_list[word_idx]
            if word_id == PAD_ID:
                break
            context_id = word_id_list[:word_idx] + word_id_list[word_idx+1:]
            res.append((word_id, context_id))
    if modelname == "self-attention":
        word_id_list = [word2id.get(word, UNK_ID) for word in sentence]
        res.append(word_id_list)
    return res

def pad_data(sentences, max_len):
    sentences = [(sen[:max_len], label) if isinstance(label, str) 
                else (sen[:max_len], label[:max_len]) 
                for sen, label in sentences]
    padded_sen = []
    for sen, label in sentences:
        padding = [PAD] * (max_len - len(sen))
        # print(self.num_class, label)
        lable_padding = label if isinstance(label, str) \
                        else label + ['__label__非事件'] * (max_len - len(sen))
        padded_sen.append((sen + padding, lable_padding))
    return padded_sen

def one_hot(label, num_classes):
    tmp = np.eye(num_classes)
    assert isinstance(label, int)
    return tmp[label].tolist()

def convert_dataset_sen(data, word2id, tag2id, num_classes = 8, one_hot_label=False):
    res = []
    for sen in data:
        tmp = []
        for word, context, label in sen:
            word_id = word2id.get(word, UNK_ID)
            context_id = [word2id.get(word, UNK_ID) for word in context]
            label_id = tag2id[label] if not one_hot_label else one_hot(tag2id[label], num_classes)
            tmp.append((word_id, context_id, label_id))
        res.append(tmp)
    return res

def convert_dataset(data, word2id, tag2id, max_len, num_class, model='textcnn'):
    res = []
    assert isinstance(data[0], tuple) == True, print("data format error")
    if  model not in TRIGGER_MODEL:
        if model == 'textcnn':
            data = pad_data(data, max_len)
        for word_list, label in data:
            word_id_list = [word2id.get(word, UNK_ID) for word in word_list]
            label_id = tag2id[label]
            res.append((word_id_list, one_hot(label_id, num_class))) 
    elif model == 'attention':
        for word, context, label in data:
            word_id = word2id.get(word, UNK_ID)
            context_id = [word2id.get(word, UNK_ID) for word in context]
            label_id = tag2id[label]
            res.append((word_id, context_id, one_hot(label_id, num_class)))

    return res


class Batch_self_attention(object):
    def __init__(self, sentences, batch_size = 16):
        self.data_size = len(sentences)
        self.batch_size = batch_size
        # self.num_batch = int(math.ceil(self.data_size / self.batch_size))
        self.num_batch = self.data_size // self.batch_size

        self.sentences = sorted(sentences, key=lambda x: len(x[0]))
        self.batch_data = self.patch_to_batches()

    def patch_to_batches(self):
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad(self.sentences[i*self.batch_size: (i+1)*self.batch_size]))
        return batch_data

    def pad(self, sentences):
        max_len = max([len(sen[0]) for sen in sentences])
        padded_sentences = []
        for sen in sentences:
            word_id_list, label_id_onehot = sen
            padding = [PAD_ID] * (max_len - len(word_id_list))
            padded_sentences.append((word_id_list + padding, label_id_onehot))
        return padded_sentences

    def next_batch(self, shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)
        for batch in self.batch_data:
            yield batch if self.batch_size > 1 else batch[0]



class Batch(object):
    def __init__(self, sentences, batch_size = 16):
        self.data_size = len(sentences)
        self.batch_size = batch_size
        self.num_batch = int(math.ceil(self.data_size / self.batch_size))

        self.sentences = sentences
        self.batch_data = self.patch_to_batches()

    def patch_to_batches(self):
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.sentences[i*self.batch_size: (i+1)*self.batch_size])
        return batch_data

    def next_batch(self, shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)
        for batch in self.batch_data:
            yield batch if self.batch_size > 1 else batch[0]
