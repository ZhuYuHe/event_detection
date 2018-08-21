import tensorflow as tf 
import numpy as np

def test_argmax(sess):
    # tmp: 2 * 3
    tmp = [[1,2,3], [4,6,5]]
    a = tf.constant(tmp)
    pred = tf.argmax(a, 1)
    print(sess.run(pred))

def test_attention(sess, x, context):
    # [batch_size]
    input_x = tf.placeholder(tf.int32, [None], name = 'X')
    # batch_size * window_size
    context_x = tf.placeholder(tf.int32, [None, None], name = "Context_X")
    embedding = [[0.5, 0.6, 0.4], [0.4, 0.8, 0.2], [0.2,0.5,0.5],[0.4,0.2,0.3],[0.5,0.5,0.5],[0.2,0.3,0.5]]
    source_embedding = tf.get_variable("source_embedding", dtype=tf.float32, initializer=tf.constant(embedding), trainable=False)
    # embedding_size * W_dim
    W = np.eye(3, dtype=np.float32)
    W_w = tf.get_variable(
        "W_w",
        dtype=tf.float32,
        initializer=tf.constant(W), trainable=False)
    # vocab_size * W_dim
    hidden_embedding = tf.tanh(tf.matmul(source_embedding, W_w))
    # batch_size * embedding_size
    source_inputs = tf.nn.embedding_lookup(source_embedding, input_x)
    # batch_size * W_dim
    hidden_inputs = tf.nn.embedding_lookup(hidden_embedding, input_x)
    # batch_size * window_size * embedding_size
    source_context = tf.nn.embedding_lookup(source_embedding, context_x)
    # batch_size * window_size * W_dim
    hidden_context = tf.nn.embedding_lookup(hidden_embedding, context_x)
    # batch_size * 1 * W_dim
    hidden_inputs_expand = tf.expand_dims(hidden_inputs, 1)
    # batch_size * W_dim * window_size
    hidden_context_T = tf.transpose(hidden_context, perm=[0,2,1])
    # batch_size * 1 * window_size
    a = tf.nn.softmax(tf.einsum('aij,ajk->aik', hidden_inputs_expand, hidden_context_T))
    # batch_size * embedding_size
    c = tf.squeeze(tf.einsum('aij,ajk->aik', a, source_context))
    # batch_size * (2 * embedding_size)
    h = tf.concat([source_inputs, c], -1)

    sess.run(tf.global_variables_initializer())

    i_e, w, w_c, ak, ck, hk =  sess.run([source_inputs, hidden_inputs, hidden_context, a, c, h], feed_dict={input_x: x, context_x: context})

    print("输入embedding：{}".format(i_e))
    print("输入隐空间表示：{}".format(w))
    print("上下文隐空间表示： {}".format(w_c))
    print("注意力权重：{}".format(ak))
    print("上下文表示： {}".format(ck))
    print("事件分类模块输入：{}".format(hk))

def test_matmul(sess):
    a = [[1,2,3], [2,3,4]]
    b = [[[1,2],[2,3],[3,4]], [[1,2],[2,3],[3,4]]]
    tensor_a = tf.constant(a)
    tensor_b = tf.constant(b)
    tensor_a = tf.expand_dims(tensor_a, 1)
    res = tf.matmul(tensor_a, tensor_b)
    print(sess.run(res))

def test_einsum(sess):
    a = [[2,3,4]]
    b = [[[1,2],[2,3],[3,4]], [[1,2],[2,3],[3,4]]]
    tensor_a = tf.constant(a)
    tensor_b = tf.constant(b)
    # tensor_a = tf.expand_dims(tensor_a, 1)
    # res = tf.matmul(tensor_a, tensor_b)
    res = tf.einsum('ij,ajk->aik', tensor_a, tensor_b)
    # res_shape = tf.shape(res)
    print(sess.run(res))

def test_reduce_mean(sess):
    a = [4,5,6,7,8]
    b = [2,3,4,5,6]
    tensor_a = tf.constant(a)
    tensor_b = tf.constant(b)
    res = tf.reduce_mean([tensor_a,tensor_b], axis=0)
    print(res.eval(session=sess))


def main():
    sess = tf.Session()
    x = [0,1]
    context = [[2,3],[4,5]]
    # test_attention(sess, x, context)
    # test_matmul(sess)
    # test_einsum(sess)
    test_reduce_mean(sess)
    sess.close()

if __name__ == '__main__':
    main()