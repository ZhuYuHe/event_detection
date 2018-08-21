import codecs
import json
import struct

def main():
    path = "data/w2v_without_entity.vec"
    # f_w2v = codecs.open("word2vec.txt", 'w', 'utf8')
    f_w2i = codecs.open("word_vocab.txt", 'w', "utf8")
    with codecs.open(path, 'r', 'utf8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip().split()
            if len(line) < 3:
                # f_w2v.write(line[0] + ' ' + line[1] + '\n')
                continue
            word = line[0]
            # vec = ' '.join(line[1:])
            if i < len(lines) - 1:
                f_w2i.write(word + '\n')
            else:
                f_w2i.write(word)
            # f_w2v.write(vec + '\n')

    f_i2t = codecs.open("id2tag.txt", 'w', "utf8")
    with codecs.open("data/tag_vocab.txt", 'r', "utf8") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            tag = lines[i].strip('\n').split('\t')[0]
            if i < len(lines) - 1:
                f_i2t.write(tag + '\n')
            else:
                f_i2t.write(tag)
    f_i2t.close()

    f_w2i.close()
    # f_w2v.close()

    # with codecs.open("word2vec.bin", 'wb') as f_w:
    #     with codecs.open('word2vec.txt', 'r', 'utf8') as f:
    #         for line in f:
    #             for string in line.split():
    #                 if line.split().__len__() < 3:
    #                     f_w.write(struct.pack('i', int(string)))
    #                 else:
    #                     f_w.write(struct.pack('f', float(string)))

if __name__ == "__main__":
    main()
    

    
