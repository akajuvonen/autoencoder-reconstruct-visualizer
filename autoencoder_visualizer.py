import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


def save_figure(image, number, fig, fig_dir):
    """Saves a figure to disk as png.

    Arguments:
    image -- A 2-D matrix of pixels, values from 0 to 1, black and white
    number -- A number used for image title and filename
    fig -- Initialized pyplot figure, we do not create a new one every time
    fig_dir -- Directory in which to save figure
    """
    plt.clf()
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.title('Step {}'.format(number))
    save_path = os.path.join(fig_dir, 'fig_step_{:06d}'.format(number))
    # Additional arguments used to get rid of unneeded boundaries in the image
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


def autoencoder(x, output_size, outer_size=500, inner_size=100,
                name='autoencoder'):
    """Defines TensorFlow autoencoder model.

    Arguments:
    x -- Input tensor
    output_size -- Output layer size (should be the same as input)
    outer size -- Used for encoder and decoder layer sizes
    inner_size -- Code (latent) layer size

    Returns:
    output -- Model outputs
    """
    with tf.variable_scope(name):
        encoder = tf.layers.dense(x, outer_size, name='encoder')
        code = tf.layers.dense(encoder, inner_size, name='code')
        decoder = tf.layers.dense(code, outer_size, name='decoder')
        output = tf.layers.dense(decoder, output_size, name='output')
        return output


def main(unused_argv):
    # Load MNIST data, NOTE: Deprecated and removed in TF 1.7
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    # One example figure to reconstruct (for figures)
    example_fig = [mnist.train.images[10]]
    # Learning rate for Adam optimizer
    learning_rate = 0.001
    # Images are 28*28 pixels
    input_size = 28*28
    batch_size = 100
    # Total number of steps
    steps = 2000
    # Save training loss and image every [save_every] steps
    save_every = 100

    # Create figure directory if doesn't exist
    fig_dir = 'figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    # Initialize pyplot figure, we only create one
    fig = plt.figure(frameon=False)

    # Input placeholder
    x = tf.placeholder(tf.float32, [None, input_size], name='input')
    output = autoencoder(x, input_size)
    loss = tf.losses.mean_squared_error(x, output)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Writer for tensorboard
        writer = tf.summary.FileWriter('logdir', sess.graph)
        train_summary = tf.summary.scalar('train_loss', loss)

        for i in range(1, steps+1):
            # Take a new batch and train
            batch, _ = mnist.train.next_batch(batch_size)
            _, l, ts = sess.run([train_op, loss, train_summary],
                                feed_dict={x: batch})

            if i % save_every == 0:
                print('Training loss at step {0}: {1}'.format(i, l))
                writer.add_summary(ts, i)
                # Reconstruct example image
                result = sess.run([output], feed_dict={x: example_fig})
                # Reshape flat vector into an array
                reshaped = np.reshape(result, (28, 28))
                save_figure(reshaped, i, fig, fig_dir)

        # Use the whole test data for calculating final test loss
        test_data = mnist.test.images
        _, test_loss = sess.run([output, loss], feed_dict={x: test_data})
        print('Final test loss: {}'.format(test_loss))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
