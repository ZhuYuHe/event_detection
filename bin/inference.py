import codecs
import os
import pickle

import tensorflow as tf

from bin.train import get_model
from model.config import Config
from utils.data_utils import convert_sentence, read_vocab
from utils.train_utils import get_config_proto


def infer_cmd(model, word2id, id2tag):
    while True:
        line = input(">>")
        line = line.strip().split()
        line = convert_sentence(line, word2id, model)
        print(line)
        pred = model.inference(line)
        print(id2tag[pred[0]])

def infer_file(model, word2id, id2tag, fname, save_path):
    res = []
    with codecs.open(fname, 'r', 'utf8') as f:
        for line in f:
            line = line.strip().split('\t')[0].split()
            line = convert_sentence(line, word2id, model)
            pred = model.inference(line)
            res.append(id2tag[pred[0]])
    with codecs.open(save_path, 'w', 'utf8') as f:
        for label in res:
            f.write(label + '\n')
    return res

def main():
    checkpoint_dir = "output/self_attention/multi_attention_0802/"
    # inference 用的还是文件中的config，并不是当初的config了。应该读取check_point中的config.pkl
    # TODO: read config from checkpoint/config.pkl
    if os.path.exists(checkpoint_dir + 'config.pkl'):
        config = pickle.load(open(checkpoint_dir+'config.pkl', 'rb'))
    else:
        config = Config()

    config.mode = 'inference'
    
    # 每次训练word_vocab和tag_vocab都会变化，而inference的时候用的是当初训练该模型时的词表；所以最好把词表定下来。
    # TODO：data_utils::fix vocab
    word2id, id2word = read_vocab(config.word_vocab_file)
    tag2id, id2tag = read_vocab(config.tag_vocab_file)

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = get_model(config.model, config, sess)
        model.build()
        model.restore_model(checkpoint_dir)
        infer_cmd(model, word2id, id2tag)

if __name__ == '__main__':
    main()
