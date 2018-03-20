import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


def autoencoder(x, output_size, outer_size=500, inner_size=100):
    encoder = tf.layers.dense(x, outer_size)
    code = tf.layers.dense(encoder, inner_size)
    decoder = tf.layers.dense(code, outer_size)
    output = tf.layers.dense(decoder, output_size)
    return output


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    example_fig = [mnist.train.images[10]]
    learning_rate = 0.001
    input_size = 28*28
    batch_size = 100
    steps = 200
    save_every = 10

    fig_dir = 'figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig = plt.figure(frameon=False)

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
            if i % save_every == 0:
                print('Training loss at step {0}: {1}'.format(i, l))
                result = sess.run([output], feed_dict={x: example_fig})
                reshaped = np.reshape(result, (28, 28))
                plt.clf()
                plt.axis('off')
                plt.imshow(reshaped, cmap='gray')
                plt.title('Step {}'.format(i))
                save_path = os.path.join(fig_dir, 'fig_step_{}'.format(i))
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
