import codecs
import itertools
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from bin.inference import infer_file
from bin.train import get_model
from model.config import Config
from utils.data_utils import read_vocab
from utils.train_utils import get_config_proto

# TODO: 最好直接给 checkpoint_dir 和 验证集，其他的根据checkpoint_dir获取

def compute_confuse_matrix(fname, classes):
    """
    Give a file, compute confuse matrix of y_true and y_pred.
    """
    print('im in')
    y_true = []
    with codecs.open(fname, 'r', 'utf8') as f:
        for line in f:
            line = line.strip().split('\t')[-1]
            y_true.append(line)

    checkpoint_dir = "output/self_attention/multi_attention_0802/"
    pred_path = "tmp/eval_y_self_attention.txt"
    if os.path.exists(checkpoint_dir + 'config.pkl'):
        config = pickle.load(open(checkpoint_dir+'config.pkl', 'rb'))
    else:
        config = Config()


    config.mode = 'inference'
    
    word2id, id2word = read_vocab(config.word_vocab_file)
    tag2id, id2tag = read_vocab(config.tag_vocab_file)

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = get_model(config.model, config, sess)
        model.build()
        model.restore_model(checkpoint_dir)
        y_pred = infer_file(model, word2id, id2tag, fname, pred_path)

    cmatrix = confusion_matrix(y_true, y_pred, classes)
    print(cmatrix)
    correct = [x == y for x,y in list(zip(y_true, y_pred))]
    print(correct.count(True) / len(correct))
    return cmatrix


def plot_confuse_matrix(cm, classes, 
                        normalize=False, 
                        title = "Confusion matrix",
                        cmap="Blues"):
    """
    Give confuse matrix, plot it's image.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()

def computeF(P,R):
    return (2*P*R) / (P+R)

def computePR(cm, classes, mode='label'):
    """
    compute P and R using confuse matrix;
    if mode == 'label', then compute PR by regarding every label as position label;
    else, compute PR by formula: P = 预测对的事件数/预测出的总事件数；R = 预测对的事件数/总事件数
    """
    if mode == 'label':
        P = dict()
        R = dict()
        for i in range(len(classes)):
            R[classes[i]] = cm[i,i] / sum(cm[i])
            P[classes[i]] = cm[i,i] / sum(cm[:,i])
    else:
        non_event_idx = classes.index("__label__非事件")
        pre_correct = 0
        for i in range(len(classes)):
            if i == non_event_idx:
                continue
            pre_correct += cm[i,i]
        events_pred = np.sum(cm[:, :non_event_idx]) + np.sum(cm[:, non_event_idx+1:])
        events_true = np.sum(cm[:non_event_idx, :]) + np.sum(cm[non_event_idx+1:, :])
        P = pre_correct / events_pred
        R = pre_correct / events_true
    return P, R


def compute_event_PR(cm, classes):
    non_event_idx = classes.index('__label__非事件')
    num_pos = sum(sum(cm[:non_event_idx, :]))# + sum(sum(cm[non_event_idx + 1 : , :]))
    pre_pos = sum(sum(cm[:, :non_event_idx]))# + sum(sum(cm[:, non_event_idx + 1 : -1]))
    hit_pos = sum(sum(cm)) - sum(cm[:, non_event_idx]) - sum(cm[non_event_idx, :]) + cm[non_event_idx, non_event_idx]
    P = hit_pos / pre_pos
    R = hit_pos / num_pos
    return P, R


def main():
    classes = ['__label__合作', '__label__发布产品', '__label__投/融资', '__label__人员变动', '__label__上市', '__label__发布消息', '__label__其他', '__label__非事件']
    cm = compute_confuse_matrix("data/data_tech_300.eval", classes)
    P, R = computePR(cm, classes, 'all') 
    plot_confuse_matrix(cm, classes)
    P_event, R_event = compute_event_PR(cm, classes)
    # print("Precision of each label - {0}".format(P))
    # print("Recall of each label - {0}".format(R))
    # print("Average Precision: {0}".format(np.mean(list(P.values()))))
    # print("Average Recall: {0}".format(np.mean(list(P.values()))))
    print("Precision: {}".format(P))
    print("Recall: {}".format(R))
    print("F: {}".format(computeF(P,R)))
    print("Event Presision = {0}".format(P_event))
    print("Event Recall - {0}".format(R_event))


if __name__ == '__main__':
    main()
