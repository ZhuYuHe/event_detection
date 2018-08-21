import argparse
import codecs
import os
import pickle

import tensorflow as tf

from model.attention import AttentionModel
from model.config import Config
from model.self_attention import SelfAttention
from model.textcnn import TextCNN
from utils.argument_utils import add_argument, config_from_args
from utils.data_utils import (Batch, convert_dataset, convert_dataset_sen,
                              create_tag_vocab_from_data,
                              create_vocab_from_pretrained_w2v, read_data,
                              read_data_sen, sample, Batch_self_attention,
                              read_vocab)
from utils.evaluate import evaluate, evaluate_attention
from utils.train_utils import get_config_proto


def get_model(model_name, config, sess):
    assert model_name in ['textcnn', 'attention', 'self-attention'], print("invalid model name")
    if model_name == 'textcnn':
        return TextCNN(config, sess)
    elif model_name == 'attention':
        return AttentionModel(config, sess)
    elif model_name == 'self-attention':
        return SelfAttention(config, sess)
    return None
    

def main():
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()

    config = Config()

    train_data = read_data(config.train_data_files, config.model)
    # 试试对负样本进行降采样
    # train_data = sample(train_data)
    eval_data = read_data(config.eval_data_files, config.model) 
    # train_data_sen = read_data_sen("data/data_tech.train")
    # eval_data_sen = read_data_sen("data/data_tech.eval")
    # 这里使用了预训练的词向量的词表作为了模型的词表
    create_vocab_from_pretrained_w2v(config.w2v_path, config.word_vocab_file)
    create_tag_vocab_from_data(train_data, config.tag_vocab_file) 

    word2id, id2word = read_vocab(config.word_vocab_file)
    tag2id, id2tag = read_vocab(config.tag_vocab_file)

    # convert word into ids
    train_data = convert_dataset(train_data, word2id, tag2id, config.sentence_length, config.num_classes, config.model)
    # train_data_sen = convert_dataset_sen(train_data_sen, word2id, tag2id, config.num_classes, one_hot_label=True)
    print(train_data[0])
    eval_data = convert_dataset(eval_data, word2id, tag2id, config.sentence_length, config.num_classes, config.model)
    # eval_data_sen = convert_dataset_sen(eval_data_sen, word2id, tag2id, config.num_classes, one_hot_label=True)
    print("train_data size: {0}".format(len(train_data)))

    if os.path.exists(os.path.join(config.checkpoint_dir, "config.pkl")):
        config = pickle.load(open(os.path.join(config.checkpoint_dir, "config.pkl"), 'rb'))
    else:
        pickle.dump(config, open(os.path.join(config.checkpoint_dir, "config.pkl"), 'wb'))

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = get_model(config.model, config, sess)
        model.build()
        model.init()

        batch_manager = Batch_self_attention(train_data, config.batch_size)
        batch_manager_eval = Batch_self_attention(eval_data, config.batch_size)
        # batch_manager = Batch(train_data, config.batch_size)
        # batch_manager_eval = Batch(eval_data, config.batch_size)
        epoches = config.epoch
        max_acc = 0
        for i in range(epoches):
            for batch in batch_manager.next_batch():
                # print(batch)
                loss, accuracy, global_step = model.train_one_step(*zip(*batch))
                # key_shape, query_shape = model.test(*zip(*batch))
                # print(key_shape, query_shape)
                # break
            train_accuracy = evaluate(model, batch_manager)
            eval_accuracy = evaluate(model, batch_manager_eval)
            # train_accuracy = evaluate_attention(model, train_data_sen, id2tag)
            # eval_accuracy = evaluate_attention(model, eval_data_sen, id2tag)
            print("epoch - {0}      step - {1}      loss - {2}      train_accuracy - {3}    eval_accuracy - {4}"\
                .format(i, global_step, loss, train_accuracy, eval_accuracy))

            # train_accuracy = evaluate_attention(model, train_data_sen, id2tag)
            # eval_accuracy = evaluate_attention(model, eval_data_sen, id2tag) 
            # print("epoch - {0}      step - {1}      loss - {2}      train_accuracy - {3}    eval_accuracy - {4}"\
            #         .format(i, global_step, loss, train_accuracy, eval_accuracy))

            if max_acc < eval_accuracy:
                max_acc = eval_accuracy
                model.save_model()


if __name__ == '__main__':
    main()
