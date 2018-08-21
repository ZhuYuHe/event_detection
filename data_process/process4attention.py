# 将数据处理为attention model的输入格式
# 即 word w   Context(w)    label
# 每行一个词语，Context(w) 为定长，过长截断，过短填充，然后进行打乱
import codecs
import random

from tqdm import tqdm

from model.config import Config
from utils.model_utils import PAD

TRAIN_PATH = 'data/data_tech.train'
EVAL_PATH = 'data/data_tech.eval'

def process_line(line, config):
    """
    process a line
    """
    res = []
    line = line.strip().split('\t')
    assert len(line) >= 2, print(line)
    word_list = line[0].split()
    trigger = line[1] if len(line) == 3 else "!@#$%^&"
    label = line[-1]
    word_list_len = len(word_list)

    # 确保数据格式正确
    if len(line) == 2:
        assert label == '__label__非事件', print(line)
    if len(line) == 3:
        assert label != '__label__非事件' and trigger in word_list, print(line)

    if word_list_len > config.sentence_length:
        word_list = word_list[:config.sentence_length]
    else:
        word_list += [PAD] * (config.sentence_length - word_list_len)
    
    for word_idx in range(len(word_list)):
        word = word_list[word_idx]

        if word == PAD:
            break

        word_label = '__label__非事件' if word != trigger else label

        context = word_list[:word_idx] + word_list[word_idx + 1:]
        
        res.append((word, context, word_label))
    return res

def process_file(fname, save_path, shuffle=True):
    """
    process a file and save result to save_path;
    return samples number.
    """
    config = Config()
    res = []
    with codecs.open(fname, 'r', 'utf8') as f:
        for line in tqdm(f):
            res.extend(process_line(line, config))
    if shuffle:
        random.shuffle(res)
    with codecs.open(save_path, 'w', 'utf8') as f_save:
        for line in res:
            word, context, label = line
            context = ' '.join(context)
            f_save.write(word + '\t' + context + '\t' + label + '\n')
                    
    return len(res)
                        


def main():
    save_train = "data/data_tech_attention.train"
    save_eval = "data/data_tech_attention.eval"
    train_size = process_file(TRAIN_PATH, save_train)
    eval_size = process_file(EVAL_PATH, save_eval)
    print("process done!")
    print("train data include {0} samples".format(train_size))
    print("eval data include {0} samples".format(eval_size))

if __name__ == '__main__':
    main()
