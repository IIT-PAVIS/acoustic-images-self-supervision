import models.shared as shared
import tensorflow as tf
import tensorflow.contrib.slim as slim
import models.base as base
from collections import OrderedDict

flags = tf.app.flags
FLAGS = flags.FLAGS

class HearModel(object):

    def __init__(self, input_shape=None, num_classes=14, embedding=1):

        self.scope = 'hear_net'
        self.num_classes = num_classes
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.embedding = embedding
        
    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        variables_to_restore = slim.get_model_variables(self.scope)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

    def _build_arg_scope(self, weight_decay=0.0005):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                            activation_fn=slim.nn_ops.relu,
                            weights_regularizer=slim.regularizers.l2_regularizer(weight_decay),
                            biases_initializer=slim.init_ops.zeros_initializer()):
            with slim.arg_scope([slim.layers.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def _build_network(self, inputs, num_classes=14, is_training=True, keep_prob=0.5, scope='hear_net'):
        """
        Builds a three-layer network that operates over a spectrogram.
        """
    
        with tf.variable_scope(scope, 'hear_net', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
        
            # Collect outputs for convolution2d and max_pool2d
            with slim.arg_scope([slim.layers.conv2d, slim.layers.max_pool2d],
                                outputs_collections=[end_points_collection]):
                net = slim.layers.conv2d(inputs, 128, [11, 1], [1, 1], scope='conv1')
                net = slim.layers.max_pool2d(net, [5, 1], [5, 1], scope='pool1')
                net = slim.layers.conv2d(net, 256, [5, 1], [1, 1], scope='conv2')
                net = slim.layers.max_pool2d(net, [5, 1], [5, 1], scope='pool2')
                net = slim.layers.conv2d(net, 256, [3, 1], [1, 1], scope='conv3')
                net = slim.layers.max_pool2d(net, [5, 1], [5, 1], scope='pool3')
                net = slim.layers.conv2d(net, 1024, [1, 1], padding='VALID', scope='conv4')
                net = slim.layers.conv2d(net, 1024, 1, scope='conv5')
    
        # Convert end_points_collection into a end_point dictionary
        end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
    
        return net, end_points

    def _build_model(self, spectrogram):
        """
        Builds the model using slim.
        """

        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': spectrogram,
            'is_training': is_training,
            'keep_prob': keep_prob
        })
        with slim.arg_scope(self._build_arg_scope(weight_decay=5e-4)):
            hearnet_output, hearnet_end_points = self._build_network(spectrogram,
                                                                     num_classes=self.num_classes,
                                                                     is_training=is_training,
                                                                     keep_prob=keep_prob,
                                                                     scope=self.scope)

        end_points.update(hearnet_end_points)

        shared_net_output, shared_net_end_points = shared.shared_net(hearnet_output,
                                                                     num_classes=self.num_classes,
                                                                     is_training=is_training,
                                                                     keep_prob=keep_prob,
                                                                     spatial_squeeze=True,
                                                                     scope=self.scope, embedding=self.embedding)

        end_points.update(shared_net_end_points)
        self.output = shared_net_output
        self.network = end_points
        self.train_vars = slim.get_trainable_variables(self.scope)


class DualCamHybridModel(object):

    def __init__(self, input_shape=None, num_classes=14, num_frames=12, embedding=1):

        self.scope = 'DualCamNet'

        self.num_classes = num_classes
        self.num_frames = num_frames

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.embedding = embedding
        #self.output, self.network = self._build_model()

        #self.train_vars = slim.get_trainable_variables(self.scope)

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCam-Net network parameters.
        """

        # Restore all the model layers
        variables_to_restore = slim.get_variables(self.scope) #+ slim.get_variables('shared_net')

        # Initialization operation of the pre-trained weights
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)

        # Load the pre-trained weights
        init_fn(session)

    def _build_network_slim(self, inputs, spatial_squeeze=False, scope='DualCamNet'):
        """
        Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
        """

        with tf.variable_scope(scope, 'DualCamNet', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for convolution2d and max_pool2d
            with slim.arg_scope([slim.layers.conv2d, slim.layers.max_pool2d], outputs_collections=[end_points_collection]):
                # ----------- 1st layer group ---------------
                net = tf.reshape(inputs, shape=(-1, self.num_frames, self.height, self.width, self.channels))
                net = slim.conv3d(net, self.channels, [7, 1, 1], scope='conv1', padding='SAME')
                net = tf.reshape(net, shape=(-1, self.height, self.width, self.channels))
                # ----------- 2nd layer group ---------------
                net = slim.conv2d(net, 32, [5, 5], scope='conv2', padding='SAME')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # ----------- 3rd layer group ---------------
                net = slim.conv2d(net, 64, [5, 5], scope='conv3', padding='SAME')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # ----------- 4th layer group ---------------
                # Use convolution2d instead of fully_connected layers
                net = slim.conv2d(net, 1024, 9, 12, scope='fc1', padding='VALID')

                # Convert end_points_collection into a end_point dictionary
                end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc1/squeezed')
                    end_points[sc.name + '/fc1'] = net

                return net, end_points

    def _build_network(self, inputs, scope='DualCamNet'):
        """
        Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
        """
    
        with tf.variable_scope(scope, 'DualCamNet', [inputs]):
            # ----------- 1st layer group ---------------
            inputs_reshaped = tf.reshape(inputs, shape=(-1, self.num_frames, self.height, self.width, self.channels))
            conv1 = base.build3DConvolution(inputs_reshaped, self.channels, self.channels, 1, 1, 7, name_scope='conv1',
                                            padding='SAME')
            relu1 = base.buildReLU(conv1, 'conv1')
            relu1_reshaped = tf.reshape(relu1, shape=(-1, self.height, self.width, self.channels))
            # ----------- 2nd layer group ---------------
            conv2 = base.build2DConvolution(relu1_reshaped, self.channels, 32, 5, 5, name_scope='conv2', padding='SAME')
            relu2 = base.buildReLU(conv2, 'conv2')
            pool2 = base.buildMaxPooling(relu2, 3, 3, 3, 3, 'conv2')  # 2 2 2 2
            # ----------- 3rd layer group ---------------
            conv3 = base.build2DConvolution(pool2, 32, 128, 5, 5, name_scope='conv3', padding='SAME')  # 64
            relu3 = base.buildReLU(conv3, 'conv3')
            if self.embedding == False:
                pool3 = tf.reduce_sum(relu3, axis=[1, 2])
                full1 = base.buildFullyConnected(pool3, 128, self.num_classes, name_scope='full1')
                # pool3 = base.buildMaxPooling(relu3, 2, 2, 2, 2, 'conv3')
                # ----------- 4th layer group ---------------
                # full1 = base.build2DConvolution(pool3, self.num_classes, 1024, 8, 6, name_scope='full1', padding='VALID')#12 9
                return full1, OrderedDict({  # full1
                    1: conv1,
                    2: relu1,
                    3: conv2,
                    5: relu2,
                    4: pool2,
                    6: conv3,
                    7: relu3,
                    8: pool3,
                    9: full1
                })
        return relu3, OrderedDict({  # full1
            1: conv1,
            2: relu1,
            3: conv2,
            5: relu2,
            4: pool2,
            6: conv3,
            7: relu3  # ,
            # 8: pool3,
            # 9: full1
        })

    def _build_model(self, acoustic_images):
        """
        Builds the hybrid model using slim and base functions.
        """

        # acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='acoustic_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': acoustic_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        dualcam_net_output, dualcam_net_end_points = self._build_network(acoustic_images,
                                                                         scope=self.scope)

        end_points.update(dualcam_net_end_points)

        # shared_net_output, shared_net_end_points = shared.shared_net(dualcam_net_output,
        # num_classes=self.num_classes,
        # is_training=is_training,
        # keep_prob=keep_prob,
        # spatial_squeeze=True,
        # scope=self.scope)

        # end_points.update(shared_net_end_points)

        self.output = dualcam_net_output  # shared_net_output
        self.network = end_points

        self.train_vars = slim.get_trainable_variables(self.scope)
        # return shared_net_output, end_points