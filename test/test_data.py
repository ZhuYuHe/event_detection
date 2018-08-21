import pytest
from unittest import TestCase
import numpy as np
import utils.data_utils as dutils
from model.config import Config
from utils.model_utils import UNK, UNK_ID

class Test_data(TestCase):

    def test_read_data(self):
        sentences = dutils.read_data('data/data_tech.train', 'textcnn')
        sen = ('NetS pee d 发布 SoC Builder 人工 智能 驱动 的 设计 与 集成 平台 , 加速 SoC 设计'.split(), '__label__发布产品')
        self.assertTrue(sen == sentences[7])

        sentences_1 = dutils.read_data('data/data_tech.train', 'attention')
        sen_1 = ('NetS pee d 发布 SoC Builder 人工 智能 驱动 的 设计 与 集成 平台 , 加速 SoC 设计'.split(), ['__label__非事件', '__label__非事件',
                '__label__非事件', '__label__发布产品', '__label__非事件', '__label__非事件', '__label__非事件', '__label__非事件', '__label__非事件',
                '__label__非事件', '__label__非事件','__label__非事件','__label__非事件','__label__非事件','__label__非事件','__label__非事件',
                '__label__非事件','__label__非事件',])
        self.assertTrue(sen_1 == sentences_1[7])

    def test_create_vocab_from_pretrained_w2v(self):
        vocab = dutils.create_vocab_from_pretrained_w2v('data/w2v_without_entity.vec', 'data/word_vocab.txt')
        self.assertTrue(vocab['UNK'] == 1)
        self.assertTrue(vocab['智能'] == 20)

    def test_convert_dataset(self):
        config = Config()
        sentences = dutils.read_data('data/data_tech.train', 'textcnn')
        word2id = dutils.create_vocab_from_pretrained_w2v('data/w2v_without_entity.vec', config.word_vocab_file)
        tag2id, id2tag = dutils.create_tag_vocab_from_data(sentences, config.tag_vocab_file)

        train_data = dutils.convert_dataset(sentences, word2id, tag2id)

        Batch = dutils.Batch(train_data, 8)
        test_data = train_data[7]
        self.assertTrue(test_data[0][0] == UNK_ID)
        self.assertTrue(test_data[0][3] == 54)
        self.assertTrue(test_data[1] == tag2id['__label__发布产品'])
        for batch in Batch.next_batch():
            print(batch[0])
            self.assertTrue(batch[0][1].__len__() == 8)
            break
        