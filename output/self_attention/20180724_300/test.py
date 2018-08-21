import tensorflow as tf 
from model.self_attention import SelfAttention
import json
import codecs

def main():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('output/self_attention/20180724_300/model.ckpt-1680.meta')
    saver.restore(sess, tf.train.latest_checkpoint('output/self_attention/20180724_300/'))

    weights = sess.run(["attention/query:0", "W:0", "output/b:0"])

    weight_dict = {
        'query': weights[0][0].tolist(),
        'W': weights[1].tolist(),
        'b': weights[2].tolist()
    }
    with codecs.open('weights.txt', 'w', 'utf8') as f:
        json.dump(weight_dict, f)
        

if __name__ ==  '__main__':
    main()