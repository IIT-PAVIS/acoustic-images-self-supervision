import numpy as np
import tensorflow as tf

from base import build2DConvolution
from base import build3DConvolution
from base import buildBatchNormalization
from base import buildReLU
from base import buildMaxPooling
from base import buildFullyConnected
from base import buildDropout


def buildDualCamNetwork(x, keep_prob, name_scope='DualCamNet'):
    """
    Builds a DualCamNet network.
    """
    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        h_conv1 = build2DConvolution(x, 512, 32, 5, 5, name_scope='conv1', padding='SAME')
        h_relu1 = buildReLU(h_conv1, 'conv1')
        h_pool1 = buildMaxPooling(h_relu1, 2, 2, 2, 2, 'conv1')
        # ----------- 2nd layer group ---------------
        h_conv2 = build2DConvolution(h_pool1, 32, 64, 5, 5, name_scope='conv2', padding='SAME')
        h_relu2 = buildReLU(h_conv2, 'conv2')
        h_pool2 = buildMaxPooling(h_relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        h_full1 = buildFullyConnected(h_pool2, 9 * 12 * 64, 1024, name_scope='full1')
        h_drop1 = buildDropout(h_full1, keep_prob)
        # ----------- 4th layer group ---------------
        h_full2 = buildFullyConnected(h_drop1, 1024, 1000, name_scope='full2')

    return h_full2, {
        1: h_conv1,
        2: h_relu1,
        3: h_pool1,
        4: h_conv2,
        5: h_relu2,
        6: h_pool2,
        7: h_full1,
        8: h_drop1,
        9: h_full2
    }


def buildDualCamSimpleNetwork(x, keep_prob, name_scope='DualCamSimpleNet'):
    """
    Builds a DualCamNet network that works on channel aggregated data.
    """
    with tf.variable_scope(name_scope):
        h_full1 = buildFullyConnected(x, 512, 1024, name_scope='full1')
        h_relu1 = buildReLU(h_full1, 'full1')
        h_drop1 = buildDropout(h_relu1, keep_prob)
        h_full2 = buildFullyConnected(h_drop1, 1024, 1000, name_scope='full2')

    return h_full2, {
        1: h_full1,
        2: h_relu1,
        3: h_drop1,
        4: h_full2
    }


def buildDualCamClassNetwork(x, keep_prob, is_training, num_classes, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        h_conv1 = build2DConvolution(x, 512, 32, 5, 5, name_scope='conv1', padding='SAME')
        h_relu1 = buildReLU(h_conv1, 'conv1')
        h_pool1 = buildMaxPooling(h_relu1, 2, 2, 2, 2, 'conv1')
        # ----------- 2nd layer group ---------------
        h_conv2 = build2DConvolution(h_pool1, 32, 64, 5, 5, name_scope='conv2', padding='SAME')
        h_relu2 = buildReLU(h_conv2, 'conv2')
        h_pool2 = buildMaxPooling(h_relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        h_full1 = buildFullyConnected(h_pool2, 9 * 12 * 64, 1024, name_scope='full1')
        h_relu3 = buildReLU(h_full1, 'full1')
        h_drop1 = buildDropout(h_relu3, keep_prob)
        # ----------- 4th layer group ---------------
        h_full2 = buildFullyConnected(h_drop1, 1024, 1000, name_scope='full2')
        h_relu4 = buildReLU(h_full2, 'full2')
        # ----------- 5th layer group ---------------
        h_full3 = buildFullyConnected(h_relu4, 1000, num_classes, name_scope='full3')

    return h_full3, {
        1: h_conv1,
        2: h_relu1,
        3: h_pool1,
        5: h_conv2,
        4: h_relu2,
        6: h_pool2,
        7: h_full1,
        8: h_relu3,
        9: h_drop1,
        10: h_full2,
        11: h_relu4,
        12: h_full3
    }


def buildDualCamClassGapNetwork(x, keep_prob, is_training, num_classes, name_scope='DualCamClassGapNet'):
    """
    Builds a DualCamNet Global Average Pooling network.
    """

    # Build the network graph
    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        conv1 = build2DConvolution(x, 512, 32, 5, 5, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        pool1 = buildMaxPooling(relu1, 2, 2, 2, 2, 'conv1')
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(pool1, 32, 64, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv1')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(relu2, 64, 1024, 3, 3, name_scope='conv3')
        relu3 = buildReLU(conv3, 'conv1')
        gap1 = tf.reduce_mean(relu3, [1, 2], name='gap')
        drop1 = buildDropout(gap1, keep_prob)
        # ----------- 4th layer group ---------------
        full1 = buildFullyConnected(drop1, 1024, num_classes, name_scope='full1')

    return full1, {
        1: conv1,
        2: relu1,
        3: pool1,
        4: conv2,
        5: relu2,
        6: conv3,
        7: gap1,
        8: full1
    }


def buildDualCamClassNetworkV2(x, keep_prob, is_training, num_classes, name_scope='DualCamClassNetV2'):
    """
    Builds a DualCamNet network for classification based on AlexNet.
    """

    # Build the network graph
    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        conv1 = build2DConvolution(x, 512, 96, 11, 11, 1, 1, 5, 5, name_scope='conv1', padding='VALID')
        relu1 = buildReLU(conv1, 'conv1')
        pool1 = buildMaxPooling(relu1, 2, 2, 2, 2, name_scope='conv1')
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(pool1, 96, 256, 5, 5, 1, 1, 2, 2, name_scope='conv2', padding='VALID')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, name_scope='conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 256, 384, 3, 3, 1, 1, 1, 1, name_scope='conv3')
        relu3 = buildReLU(conv3, 'conv3')
        # ----------- 4th layer group ---------------
        conv4 = build2DConvolution(relu3, 384, 384, 3, 3, 1, 1, 1, 1, name_scope='conv4')
        relu4 = buildReLU(conv4, 'conv4')
        # ----------- 5th layer group ---------------
        conv5 = build2DConvolution(relu4, 384, 256, 3, 3, 1, 1, 1, 1, name_scope='conv5')
        relu5 = buildReLU(conv5, 'conv5')
        pool5 = buildMaxPooling(relu5, 2, 3, 2, 2, name_scope='conv5')
        # ----------- 6th layer group ---------------
        full1 = buildFullyConnected(pool5, 4 * 6 * 256, 4096, name_scope='full1')
        relu6 = buildReLU(full1, 'full1')
        # ----------- 7th layer group ---------------
        full2 = buildFullyConnected(relu6, 4096, 4096, name_scope='full2')
        relu7 = buildReLU(full2, 'full2')
        # ----------- 8th layer group ---------------
        full3 = buildFullyConnected(relu7, 4096, num_classes, name_scope='full3')

    return full3, {
        1: conv1,
        2: relu1,
        3: pool1,
        4: conv2,
        5: relu2,
        6: pool2,
        7: conv3,
        8: relu3,
        9: conv4,
        10: relu4,
        11: conv5,
        12: relu5,
        13: pool5,
        14: full1,
        15: relu6,
        16: full2,
        17: relu7,
        18: full3
    }


def buildDualCamClassNetworkV3(x, keep_prob, is_training, num_classes, name_scope='DualCamClassNetV3'):
    """
    Builds a DualCamNet network for classification using less aggressive filters.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        conv1 = build2DConvolution(x, 512, 512, 1, 1, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 32, 64, 5, 5, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full4 = buildFullyConnected(pool3, 9 * 12 * 64, 1024, name_scope='full1')
        relu4 = buildReLU(full4, 'full1')
        drop4 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full5 = buildFullyConnected(drop4, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full5, 'full2')
        # ----------- 6th layer group ---------------
        full6 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full6, {
        1: conv1,
        2: relu1,
        3: conv2,
        4: relu2,
        5: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full4,
        10: relu4,
        11: drop4,
        12: full5,
        13: relu5,
        14: full6
    }


def buildDualCamClassNetworkV4(x, keep_prob, is_training, num_classes, name_scope='DualCamClassNetV4'):
    """
    Builds a DualCamNet network for classification using less aggressive filters.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        conv1 = build2DConvolution(x, 512, 512, 1, 1, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1, 512, 256, 3, 3, 1, 1, 1, 1, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 256, 128, 3, 3, 1, 1, 1, 1, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full4 = buildFullyConnected(pool3, 9 * 12 * 128, 1024, name_scope='full1')
        relu4 = buildReLU(full4, 'full1')
        drop4 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full5 = buildFullyConnected(drop4, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full5, 'full2')
        # ----------- 6th layer group ---------------
        full6 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full6, {
        1: conv1,
        2: relu1,
        3: conv2,
        4: relu2,
        5: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full4,
        10: relu4,
        11: drop4,
        12: full5,
        13: relu5,
        14: full6
    }


def buildDualCamClassNetworkV5(x, keep_prob, is_training, num_classes, name_scope='DualCamClassNetV5'):
    """
    Builds a DualCamNet network for classification using less aggressive filters.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        conv1 = build2DConvolution(x, 512, 128, 1, 1, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1, 128, 256, 3, 3, 1, 1, 1, 1, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 256, 512, 3, 3, 1, 1, 1, 1, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full4 = buildFullyConnected(pool3, 9 * 12 * 512, 1024, name_scope='full1')
        relu4 = buildReLU(full4, 'full1')
        drop4 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full5 = buildFullyConnected(drop4, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full5, 'full2')
        # ----------- 6th layer group ---------------
        full6 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full6, {
        1: conv1,
        2: relu1,
        3: conv2,
        4: relu2,
        5: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full4,
        10: relu4,
        11: drop4,
        12: full5,
        13: relu5,
        14: full6
    }


def buildDualCamClassNetworkV6(x, keep_prob, is_training, num_classes, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification with batch normalization.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        h_conv1 = build2DConvolution(x, 512, 32, 5, 5, name_scope='conv1', padding='SAME')
        h_norm1 = buildBatchNormalization(h_conv1, is_training, 'conv1')
        h_relu1 = buildReLU(h_norm1, 'conv1')
        h_pool1 = buildMaxPooling(h_relu1, 2, 2, 2, 2, 'conv1')
        # ----------- 2nd layer group ---------------
        h_conv2 = build2DConvolution(h_pool1, 32, 64, 5, 5, name_scope='conv2', padding='SAME')
        h_norm2 = buildBatchNormalization(h_conv2, is_training, 'conv2')
        h_relu2 = buildReLU(h_norm2, 'conv2')
        h_pool2 = buildMaxPooling(h_relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        h_full1 = buildFullyConnected(h_pool2, 9 * 12 * 64, 1024, name_scope='full1')
        h_relu3 = buildReLU(h_full1, 'full1')
        h_drop1 = buildDropout(h_relu3, keep_prob)
        # ----------- 4th layer group ---------------
        h_full2 = buildFullyConnected(h_drop1, 1024, 1000, name_scope='full2')
        h_relu4 = buildReLU(h_full2, 'full2')
        # ----------- 5th layer group ---------------
        h_full3 = buildFullyConnected(h_relu4, 1000, num_classes, name_scope='full3')

    return h_full3, {
        1: h_conv1,
        2: h_norm1,
        3: h_relu1,
        4: h_pool1,
        5: h_conv2,
        6: h_norm2,
        7: h_relu2,
        8: h_pool2,
        9: h_full1,
        10: h_relu3,
        11: h_drop1,
        12: h_full2,
        13: h_relu4,
        14: h_full3
    }


def buildDualCamClassNetworkV7(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 3x1x1 filters.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 3, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        relu1_reshaped = tf.reshape(relu1, shape=(-1, 36, 48, 512))
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1_reshaped, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 32, 64, 5, 5, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full1 = buildFullyConnected(pool3, 9 * 12 * 64, 1024, name_scope='full1')
        relu4 = buildReLU(full1, 'full1')
        drop1 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full2 = buildFullyConnected(drop1, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full2, 'full2')
        # ----------- 6th layer group ---------------
        full3 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full3, {
        1: conv1,
        2: relu1,
        3: conv2,
        5: relu2,
        4: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full1,
        10: relu4,
        11: drop1,
        12: full2,
        13: relu5,
        14: full3
    }


def buildDualCamClassNetworkV8(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 3x1x1 filters before
    the first 2D spatial convolutional layer and default weights.
    """

    weights = np.eye(512, 512)
    weights = np.expand_dims(weights, 0)
    weights = np.expand_dims(weights, 0)
    weights = np.stack([np.zeros(weights.shape), weights, np.zeros(weights.shape)])

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 3, name_scope='conv1', padding='SAME', weights=weights)
        relu1 = buildReLU(conv1, 'conv1')
        relu1_reshaped = tf.reshape(relu1, shape=(-1, 36, 48, 512))
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1_reshaped, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 32, 64, 5, 5, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full1 = buildFullyConnected(pool3, 9 * 12 * 64, 1024, name_scope='full1')
        relu4 = buildReLU(full1, 'full1')
        drop1 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full2 = buildFullyConnected(drop1, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full2, 'full2')
        # ----------- 6th layer group ---------------
        full3 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full3, {
        1: conv1,
        2: relu1,
        3: conv2,
        5: relu2,
        4: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full1,
        10: relu4,
        11: drop1,
        12: full2,
        13: relu5,
        14: full3
    }


def buildDualCamClassNetworkV9(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 5x1x1 filters.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 5, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        relu1_reshaped = tf.reshape(relu1, shape=(-1, 36, 48, 512))
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1_reshaped, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 32, 64, 5, 5, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full1 = buildFullyConnected(pool3, 9 * 12 * 64, 1024, name_scope='full1')
        relu4 = buildReLU(full1, 'full1')
        drop1 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full2 = buildFullyConnected(drop1, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full2, 'full2')
        # ----------- 6th layer group ---------------
        full3 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full3, {
        1: conv1,
        2: relu1,
        3: conv2,
        5: relu2,
        4: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full1,
        10: relu4,
        11: drop1,
        12: full2,
        13: relu5,
        14: full3
    }


def buildDualCamClassNetworkV10(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 7, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        relu1_reshaped = tf.reshape(relu1, shape=(-1, 36, 48, 512))
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1_reshaped, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(pool2, 32, 64, 5, 5, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        full1 = buildFullyConnected(pool3, 9 * 12 * 64, 1024, name_scope='full1')
        relu4 = buildReLU(full1, 'full1')
        drop1 = buildDropout(relu4, keep_prob)
        # ----------- 5th layer group ---------------
        full2 = buildFullyConnected(drop1, 1024, 1000, name_scope='full2')
        relu5 = buildReLU(full2, 'full2')
        # ----------- 6th layer group ---------------
        full3 = buildFullyConnected(relu5, 1000, num_classes, name_scope='full3')

    return full3, {
        1: conv1,
        2: relu1,
        3: conv2,
        5: relu2,
        4: pool2,
        6: conv3,
        7: relu3,
        8: pool3,
        9: full1,
        10: relu4,
        11: drop1,
        12: full2,
        13: relu5,
        14: full3
    }


def buildDualCamClassNetworkV10a(x, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
    """

    # Create the data placeholders
    acoustic_images = tf.placeholder(tf.float32, [None, 36, 48, 512])
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    output, network = buildDualCamClassNetworkV10(x, keep_prob, num_classes, num_frames, name_scope)

    network.update({
        'is_training': is_training,
        'keep_prob': keep_prob,
        0: acoustic_images})

    return output, network


def buildDualCamClassNetworkV11(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 3x1x1 filters before
    and after the first 2D spatial convolutional layer.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 3, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        relu1_reshaped = tf.reshape(relu1, shape=(-1, 36, 48, 512))
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1_reshaped, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        pool2_reshaped = tf.reshape(pool2, shape=(-1, num_frames, 18, 24, 32))
        # ----------- 3rd layer group ---------------
        conv3 = build3DConvolution(pool2_reshaped, 32, 32, 1, 1, 3, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        relu3_reshaped = tf.reshape(relu3, shape=(-1, 18, 24, 32))
        # ----------- 4th layer group ---------------
        conv4 = build2DConvolution(relu3_reshaped, 32, 64, 5, 5, name_scope='conv4', padding='SAME')
        relu4 = buildReLU(conv4, 'conv4')
        pool4 = buildMaxPooling(relu4, 2, 2, 2, 2, 'conv4')
        # ----------- 5th layer group ---------------
        full5 = buildFullyConnected(pool4, 9 * 12 * 64, 1024, name_scope='full1')
        relu5 = buildReLU(full5, 'full1')
        drop5 = buildDropout(relu5, keep_prob)
        # ----------- 6th layer group ---------------
        full6 = buildFullyConnected(drop5, 1024, 1000, name_scope='full2')
        relu6 = buildReLU(full6, 'full2')
        # ----------- 7th layer group ---------------
        full7 = buildFullyConnected(relu6, 1000, num_classes, name_scope='full3')

    return full7, {
        1: conv1,
        2: relu1,
        3: conv2,
        5: relu2,
        4: pool2,
        6: conv3,
        7: relu3,
        8: conv4,
        9: relu4,
        10: pool4,
        11: full5,
        12: relu5,
        13: drop5,
        14: full6,
        15: relu6,
        16: full7
    }


def buildDualCamClassNetworkV12(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 3x1x1 filters before
    and after every 2D spatial convolutional layer.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 3, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        relu1_reshaped = tf.reshape(relu1, shape=(-1, 36, 48, 512))
        # ----------- 2nd layer group ---------------
        conv2 = build2DConvolution(relu1_reshaped, 512, 32, 5, 5, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        pool2 = buildMaxPooling(relu2, 2, 2, 2, 2, 'conv2')
        pool2_reshaped = tf.reshape(pool2, shape=(-1, num_frames, 18, 24, 32))
        # ----------- 3rd layer group ---------------
        conv3 = build3DConvolution(pool2_reshaped, 32, 32, 1, 1, 3, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        relu3_reshaped = tf.reshape(relu3, shape=(-1, 18, 24, 32))
        # ----------- 4th layer group ---------------
        conv4 = build2DConvolution(relu3_reshaped, 32, 64, 5, 5, name_scope='conv4', padding='SAME')
        relu4 = buildReLU(conv4, 'conv4')
        pool4 = buildMaxPooling(relu4, 2, 2, 2, 2, 'conv4')
        pool4_reshaped = tf.reshape(pool4, shape=(-1, num_frames, 9, 12, 64))
        # ----------- 5th layer group ---------------
        conv4a = build3DConvolution(pool4_reshaped, 64, 64, 1, 1, 3, name_scope='conv5', padding='SAME')
        relu4a = buildReLU(conv4a, 'conv5')
        relu4a_reshaped = tf.reshape(relu4a, shape=(-1, 9, 12, 64))
        # ----------- 6th layer group ---------------
        full5 = buildFullyConnected(relu4a_reshaped, 9 * 12 * 64, 1024, name_scope='full1')
        relu5 = buildReLU(full5, 'full1')
        drop5 = buildDropout(relu5, keep_prob)
        # ----------- 7th layer group ---------------
        full6 = buildFullyConnected(drop5, 1024, 1000, name_scope='full2')
        relu6 = buildReLU(full6, 'full2')
        # ----------- 8th layer group ---------------
        full7 = buildFullyConnected(relu6, 1000, num_classes, name_scope='full3')

    return full7, {
        1: conv1,
        2: relu1,
        3: conv2,
        5: relu2,
        4: pool2,
        6: conv3,
        7: relu3,
        8: conv4,
        9: relu4,
        10: pool4,
        11: full5,
        12: relu5,
        13: drop5,
        14: full6,
        15: relu6,
        16: full7
    }


def buildDualCamClassNetworkV13(x, keep_prob, is_training, num_classes, num_frames, name_scope='DualCamClassNet'):
    """
    Builds a DualCamNet network for classification using a set of 3D temporal convolutional layers with 13x1x1 filters
    at the beginning to compress the time information followed by two 2D spatial convolutional layers.
    """

    with tf.variable_scope(name_scope):
        # ----------- 1st layer group ---------------
        x_reshaped = tf.reshape(x, shape=(-1, num_frames, 36, 48, 512))
        conv1 = build3DConvolution(x_reshaped, 512, 512, 1, 1, 13, name_scope='conv1', padding='SAME')
        relu1 = buildReLU(conv1, 'conv1')
        # ----------- 2nd layer group ---------------
        conv2 = build3DConvolution(relu1, 512, 512, 1, 1, 13, name_scope='conv2', padding='SAME')
        relu2 = buildReLU(conv2, 'conv2')
        relu2_reshaped = tf.reshape(relu2, shape=(-1, 36, 48, 512))
        # ----------- 3rd layer group ---------------
        conv3 = build2DConvolution(relu2_reshaped, 512, 32, 5, 5, name_scope='conv3', padding='SAME')
        relu3 = buildReLU(conv3, 'conv3')
        pool3 = buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
        # ----------- 4th layer group ---------------
        conv4 = build2DConvolution(pool3, 32, 64, 5, 5, name_scope='conv4', padding='SAME')
        relu4 = buildReLU(conv4, 'conv4')
        pool4 = buildMaxPooling(relu4, 2, 2, 2, 2, 'conv4')
        # ----------- 5th layer group ---------------
        full1 = buildFullyConnected(pool4, 9 * 12 * 64, 1024, name_scope='full1')
        relu5 = buildReLU(full1, 'full1')
        drop1 = buildDropout(relu5, keep_prob)
        # ----------- 6th layer group ---------------
        full2 = buildFullyConnected(drop1, 1024, 1000, name_scope='full2')
        relu6 = buildReLU(full2, 'full2')
        # ----------- 7th layer group ---------------
        full3 = buildFullyConnected(relu6, 1000, num_classes, name_scope='full3')

    return full3, {
        1: conv1,
        2: relu1,
        3: conv2,
        4: relu2,
        5: conv3,
        6: relu3,
        7: pool3,
        8: conv4,
        9: relu4,
        10: pool4,
        11: full1,
        12: relu5,
        13: drop1,
        14: full2,
        15: relu6,
        16: full3
    }
