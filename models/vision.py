import tensorflow as tf
import tensorflow.contrib.slim as slim
import models.resnet18_v1 as resnet18_v1

class ResNet18_v1(object):
    
    def __init__(self, input_shape=None, num_classes=14, map =True):

        self.scope = 'resnet_v1'
        self.num_classes = num_classes
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.map = map
    def init_model(self, session, checkpoint_file):
        """
        Initializes ResNet-18 network parameters using slim.
        """
        
        # Restore only the layers up to logits (excluded)
        model_variables = slim.get_model_variables(self.scope)
        variables_to_restore = slim.filter_variables(model_variables)#, exclude_patterns=['logits']
        
        # Initialization operation of the pre-trained weights
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        
        # Load the pre-trained weights
        init_fn(session)
        
    def _build_model(self, visual_images):
        """
        Builds a ResNet-18 network using slim.
        """

        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with slim.arg_scope(resnet18_v1.resnet_arg_scope(weight_decay=5e-4)):
            output, network = resnet18_v1.resnet_v1_18(visual_images, num_classes=self.num_classes,
                                                          map=self.map,
                                                          is_training=is_training)
        if not self.map:
            output = tf.squeeze(output, [1, 2])
        
        network.update({
            'input': visual_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })
        self.output = output
        self.network = network
        # we train all network because we don't have a checkpoint
        self.train_vars = slim.get_trainable_variables(self.scope)
