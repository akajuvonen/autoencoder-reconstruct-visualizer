import tensorflow as tf
import os


def autoencoder(x, output_size, outer_size=500, inner_size=100):
    encoder = tf.layers.dense(x, outer_size)
    code = tf.layers.dense(encoder, inner_size)
    decoder = tf.layers.dense(code, outer_size)
    output = tf.layers.dense(decoder, output_size)
    return output


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    learning_rate = 0.001
    input_size = 28*28
    batch_size = 100
    steps = 1000

    fig_dir = 'figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    x = tf.placeholder(tf.float32, [None, input_size], name='input')

    output = autoencoder(x, input_size)

    loss = tf.losses.mean_squared_error(x, output)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1, steps+1):
            batch, _ = mnist.train.next_batch(batch_size)
            _, l = sess.run([train_op, loss], feed_dict={x: batch})
            if i%100 == 0:
                print('Training loss at step {0}: {1}'.format(i, l))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
