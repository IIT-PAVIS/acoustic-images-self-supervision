import tensorflow as tf
import tensorflow.contrib.slim as slim
from models import base


def shared_net(inputs, num_classes=None, is_training=True, keep_prob=0.5, spatial_squeeze=True, scope='shared_net', embedding=1):
    """
    Builds a three-layer fully-connected modality agnostic network.
    """

    with tf.variable_scope(scope, [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for convolution2d and max_pool2d
        with slim.arg_scope([slim.layers.conv2d], padding='VALID', outputs_collections=[end_points_collection],
                            weights_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01),
                            biases_initializer=tf.constant_initializer(0.0)):
            # Use convolution2d instead of fully_connected layers
            net = slim.layers.conv2d(inputs, 1000, 1, scope='fc1')
            end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
            netsqueeze1 = tf.squeeze(net, [1, 2], name='fc/squeezed')
            end_points[sc.name + '/fc1'] = netsqueeze1
            if embedding:
                net = slim.layers.conv2d(net, 128, 1, activation_fn=None, normalizer_fn=None, scope='fc2')
                net2 = tf.squeeze(net, [1, 2], name='fc2/squeezed')
                # Add squeezed fc1 to the collection of end points
                end_points[sc.name + '/fc2'] = net2
            else:
                net11 = slim.layers.conv2d(net, 128, 1, scope='fc2')
                net22 = slim.layers.conv2d(net11, num_classes, 1, activation_fn=None, normalizer_fn=None, scope='fc3')
                net1 = tf.squeeze(net11, [1, 2], name='fc2/squeezed')
                # Add squeezed fc1 to the collection of end points
                end_points[sc.name + '/fc2'] = net1
                net2 = tf.squeeze(net22, [1, 2], name='fc3/squeezed')
                # Add squeezed fc1 to the collection of end points
                end_points[sc.name + '/fc3'] = net2

    return net2, end_points


def shared_net_legacy(inputs, num_classes=None, spatial_squeeze=True, scope='shared_net'):
    """
    Builds a three-layer fully-connected modality agnostic network using legacy functions.
    """

    with tf.variable_scope(scope, [inputs]):

        conv1 = base.build2DConvolution(inputs, 1024, 1024, 1, 1, 1, 1, 0, 0, 'fc1')
        relu1 = base.buildReLU(conv1, 'fc1')

        conv2 = base.build2DConvolution(relu1, 1024, 1000, 1, 1, 1, 1, 0, 0, 'fc2')
        relu2 = base.buildReLU(conv2, 'fc2')

        conv3 = base.build2DConvolution(relu2, 1000, num_classes, 1, 1, 1, 1, 0, 0, 'fc3')

        if spatial_squeeze:
            output = tf.squeeze(conv3, [1, 2], name='fc3/squeezed')
        else:
            output = conv3

    return output, {
        scope + '/fc1': relu1,
        scope + '/fc2': output
    }