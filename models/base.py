import tensorflow as tf


def build3DConvolution(x, nInputPlane, nOutputPlane, kW=1, kH=1, kD=1, dW=1, dH=1, dD=1, padW=0, padH=0, padD=0, name_scope='conv', padding='VALID', weights=None, biases=None):
    """
    Builds a 3-D convolutional layer acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        weights_init = tf.truncated_normal_initializer(0.0, stddev=0.01) if weights is None else tf.constant_initializer(weights)
        biases_init = tf.constant_initializer(0.0) if biases is None else tf.constant_initializer(biases)
        W = tf.get_variable('weights', shape=[kD, kH, kW, nInputPlane, nOutputPlane], initializer=weights_init)
        b = tf.get_variable('biases', shape=[nOutputPlane], initializer=biases_init)
        x_padded = tf.pad(x, [[0, 0], [padD, padD], [padH, padH], [padW, padW], [0, 0]], 'CONSTANT') if padding == 'VALID' else x
        return tf.nn.conv3d(x_padded, W, strides=[1, dD, dH, dW, 1], padding=padding, name='conv3d') + b


def build2DConvolution(x, nInputPlane, nOutputPlane, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, name_scope='conv',
                       padding='VALID', weights=None, biases=None):
    """
    Builds a 2-D convolutional layer acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        weights_init = tf.truncated_normal_initializer(0.0,
                                                       stddev=0.01) if weights is None else tf.constant_initializer(
            weights)
        biases_init = tf.constant_initializer(0.0) if biases is None else tf.constant_initializer(biases)
        W = tf.get_variable('weights', shape=[kH, kW, nInputPlane, nOutputPlane], initializer=weights_init)
        b = tf.get_variable('biases', shape=[nOutputPlane], initializer=biases_init)
        x_padded = tf.pad(x, [[0, 0], [padH, padH], [padW, padW], [0, 0]], 'CONSTANT') if padding == 'VALID' else x
        return tf.nn.conv2d(x_padded, W, strides=[1, dH, dW, 1], padding=padding, name='conv2d') + b


def buildMaxPooling(x, kW=1, kH=1, dW=1, dH=1, name_scope='conv'):
    """
    Builds a max pooling layer acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        return tf.nn.max_pool(x, ksize=[1, kH, kW, 1], strides=[1, dH, dW, 1], padding='VALID', name='max_pool')


def buildBatchNormalization(x, is_training, name_scope='conv', beta=None, gamma=None, mean=None, variance=None):
    """
    Builds a batch normalization layer acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        betaInit = tf.zeros_initializer() if beta is None else tf.constant_initializer(beta)
        gammaInit = tf.ones_initializer() if gamma is None else tf.constant_initializer(gamma)
        meanInit = tf.zeros_initializer() if mean is None else tf.constant_initializer(mean)
        varianceInit = tf.ones_initializer() if variance is None else tf.constant_initializer(variance)
        return tf.layers.batch_normalization(x, training=is_training, name='norm', beta_initializer=betaInit,
                                             gamma_initializer=gammaInit, moving_mean_initializer=meanInit,
                                             moving_variance_initializer=varianceInit)


def buildFullyConnected(x, nInputPlane, nOutputPlane, name_scope='full', weights=None, biases=None):
    """
    Builds a fully connected layer acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        weights_init = tf.truncated_normal_initializer(0.0,
                                                       stddev=0.01) if weights is None else tf.constant_initializer(
            weights)
        biases_init = tf.constant_initializer(0.0) if biases is None else tf.constant_initializer(biases)
        W = tf.get_variable('weights', shape=[nInputPlane, nOutputPlane], initializer=weights_init)
        b = tf.get_variable('biases', shape=[nOutputPlane], initializer=biases_init)
        x_flat = tf.reshape(x, [-1, nInputPlane])
        return tf.matmul(x_flat, W) + b


def buildReLU(x, name_scope='conv'):
    """
    Builds a rectifying linear unit layer acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        return tf.nn.relu(x, name='relu')


def buildDropout(x, keep_prob, name_scope='conv'):
    """
    Builds a dropout layer with probability 'keep_prob' acting over the given input 'x'.
    """
    with tf.variable_scope(name_scope):
        return tf.nn.dropout(x, keep_prob)


def buildCrossEntropyLoss(labels, logits, name_scope='cross_loss'):
    """
    Builds a loss layer that applies softmax over the logits and computes the cross-entropy between the result and the labels.
    """
    with tf.variable_scope(name_scope):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def l2_loss(labels, logits, name_scope='l2_loss', loss_collection=tf.GraphKeys.LOSSES):
    """
    Builds a loss layer that computes the L2 loss over the given 'x' and 'y'.
    """
    with tf.variable_scope(name_scope) as scope:
        diff = tf.squared_difference(logits, labels)
        loss = tf.reduce_mean(diff)
        tf.losses.add_loss(loss, loss_collection)
        return loss


def kl_loss(labels, logits, name_scope='kl_loss_stable', loss_collection=tf.GraphKeys.LOSSES):
    """
    Builds a loss layer that computes the Kullback-Leibler divergence coefficient for the given inputs. Implements the
    measure using the well known tf.nn.softmax_cross_entropy_with_logits which provides numerical stability.
    """
    with tf.variable_scope(name_scope):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(labels), logits=logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(labels), logits=labels)
        loss = tf.reduce_mean(cross_entropy - entropy)
        tf.losses.add_loss(loss, loss_collection)
        return loss


def buildKLLossUnstable(labels, logits, name_scope='kl_loss_unstable'):
    """
    Builds a loss layer that computes the Kullback-Leibler divergence coefficient for the given inputs. Implements the
    measure as the cross-entropy minus the entropy using the raw formulation of both which can be numerically unstable.
    """
    p = tf.nn.softmax(labels)
    q = tf.nn.softmax(logits)
    with tf.variable_scope(name_scope) as scope:
        cross_entropy = -tf.reduce_sum(p * tf.log(q), 1)
        entropy = tf.reduce_sum(p * tf.log(p), 1)
        return tf.reduce_mean(cross_entropy + entropy)

def buildAccuracyScalar(logits, labels, name_scope='accuracy'):
    """
    Builds a graph node to compute accuracy given 'logits' a probability distribution over the output and 'labels' a
    one-hot vector.
    """
    with tf.name_scope(name_scope) as scope:
        correct_prediction = tf.equal(logits, labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)
    
def buildAccuracy(logits, labels, name_scope='accuracy'):
    """
    Builds a graph node to compute accuracy given 'logits' a probability distribution over the output and 'labels' a
    one-hot vector.
    """
    with tf.name_scope(name_scope) as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)


class BaseModel:

    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
