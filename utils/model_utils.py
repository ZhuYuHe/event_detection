import tensorflow as tf 
import numpy as np

UNK = 'UNK'
UNK_ID = 1
PAD = '</s>'
PAD_ID = 0

TRIGGER_MODEL = ['attention']

def get_optimizer(opt):
    """
    A function to get optimizer.
    """
    if opt == 'adam':
        optfn = tf.train.AdamOptimizer
    elif opt == 'sgd':
        optfn = tf.train.GradientDescentOptimizer
    elif opt == 'adad':
        optfn = tf.train.AdadeltaOptimizer
    else:
        assert False
    return optfn

def load_embed_txt(embed_file):
    """
    Load embed_file into a python dictionary.
    """
    emb_dict = dict()
    emb_size = None
    with open(embed_file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            tokens = line.strip().split(' ')
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec)
            else:
                emb_size = len(vec)
    return emb_dict, emb_size

def load_pretrained_emb_from_txt(id2word, embed_file):
    emb_dict, emb_size = load_embed_txt(embed_file)
    embedding = np.zeros([len(id2word), emb_size], dtype=np.float32)
    for idx, word in id2word.items():
        embedding[idx] = emb_dict[word]
    return embedding
