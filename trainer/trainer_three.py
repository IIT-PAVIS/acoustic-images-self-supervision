from datetime import datetime
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn_ops


flags = tf.app.flags
FLAGS = flags.FLAGS
_FRAMES_PER_SECOND = 12


class Trainer(object):
    
    def __init__(self, model_1, model_2, logger=None, display_freq=1,
                 learning_rate=0.0001, num_classes=128, num_epochs=1, nr_frames=12, temporal_pooling=False):
        
        self.model_1 = model_1
        self.model_2 = model_2
        self.logger = logger
        
        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.nr_frames = nr_frames
        self.temporal_pooling = temporal_pooling
        self.acoustic = False
        if FLAGS.model_1 == 'DualCamHybridNet' or FLAGS.model_2 == 'DualCamHybridNet':
            self.acoustic = True
        # # Extract input model shape
        self.shape_1 = [self.model_1.height, self.model_1.width, self.model_1.channels]
        self.shape_2 = [self.model_2.height, self.model_2.width, self.model_2.channels]
    
    def _build_functions(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        self.epoch = tf.placeholder(tf.int32, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        self.data_1, self.data_2, labels, scenario = self._retrieve_batch(next_batch)
        self.labels = tf.argmax(labels, axis=1)
        self.scenario = tf.argmax(scenario, axis=1)
        # build model with tensor data next batch
        with tf.device('/gpu:0'):
            self.model_1._build_model(self.data_1)  # positive_outputANDnegative_output
        with tf.device('/gpu:1'):
            self.model_2._build_model(self.data_2)
        # find logits after defining next batch and iterator
        
        # temporal pooling gives one predition for nr_frames, if it is not we have one predicition for frame
        if FLAGS.model_1 == 'ResNet18_v1' and self.temporal_pooling:
            expanded_shape = [-1, self.nr_frames, 12, 16, self.num_classes]
            self.logits_1 = tf.reduce_mean(tf.reshape(self.model_1.output, shape=expanded_shape), axis=1)
        else:  # FLAGS.model_1 == 'DualCamHybridNet' and self.temporal_pooling:
            logits_1_multiple = self.model_1.output  # network[7]
            expanded_shape = [-1, FLAGS.sample_length * _FRAMES_PER_SECOND, 12, 16, self.num_classes]
            self.logits_1 = tf.reduce_mean(tf.reshape(logits_1_multiple, shape=expanded_shape), axis=1)
        logits_1 = self.logits_1
        
        if FLAGS.model_2 == 'DualCamHybridNet' and self.temporal_pooling:
            logits_2_multiple = self.model_2.output  # network[7]
            expanded_shape = [-1, FLAGS.sample_length * _FRAMES_PER_SECOND, 12, 16, self.num_classes]
            logits_2_reshape = tf.reduce_mean(tf.reshape(logits_2_multiple, shape=expanded_shape), axis=1)
        else:
            logits_2 = self.model_2.output
            
            # normalize vector of audio with positive and then negative
            logits_2_multiple = tf.tile(logits_2, [1, 12 * 16])
            # reshape in order to have same dimension of video feature map
            logits_2_reshape = tf.reshape(logits_2_multiple, [-1, 12, 16, self.num_classes])
            logits_2_reshape = nn_ops.relu(logits_2_reshape)
        # expand dims to have all combinations of audio and video
        logits_audio_infl = tf.expand_dims(logits_2_reshape, 1)
        logits_video_infl = tf.expand_dims(logits_1, 0)

        # inner product between frame feature map and positive audio
        innerdot = logits_audio_infl * logits_video_infl  # order matters!!!!

        # reply video to multiply inner product for each video
        video_repl = tf.tile(tf.expand_dims(logits_1, 0), [tf.shape(logits_1)[0], 1, 1, 1, 1])
        product = innerdot * video_repl
        # sum along 12 16
        videoweighted = tf.reduce_sum(product, [2, 3])
        # sum audio along 12 16 and tile horizontal to have a 64 64 map and compute distance
        # between each audio and video computed for each audio
        audio_repl_sum = tf.tile(tf.expand_dims(tf.reduce_sum(logits_2_reshape, [1, 2]), 1),
                                 [1, tf.shape(logits_1)[0], 1])
        self.productvectnorm = tf.nn.l2_normalize(videoweighted, dim=-1)
        self.productvectnorm2 = tf.nn.l2_normalize(audio_repl_sum, dim=-1)

        # triplet loss between inner product, positive and negative audio
        # remove if using random couples
        self.tripletloss, _ = tf.cond(self.epoch > 3,
                                      lambda: self.mix_data_hard(self.productvectnorm, self.productvectnorm2,
                                                                 self.labels, self.scenario,
                                                                 FLAGS.margin),
                                      lambda: self.mix_all(self.productvectnorm, self.productvectnorm2, self.labels,
                                                           self.scenario,
                                                           FLAGS.margin))
        self.loss = self.tripletloss + tf.losses.get_regularization_loss()  # + self.reg_strength*self.l1_loss
        # compute prediction if distance neg is greater than positive plus margin
        
        # Initialize counters and stats
        self.global_step = tf.train.create_global_step()
        
        # Define optimizer
        # before different
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_vars = self.model_1.train_vars + self.model_2.train_vars
        with tf.control_dependencies(update_ops):
            with tf.device('/gpu:0'):
                # Compute the gradients for acoustic variables.
                grads_and_vars = self.optimizer.compute_gradients(self.loss,
                                                                  self.model_1.train_vars)
                # Ask the optimizer to apply the gradients.
                self.train_op_0 = self.optimizer.apply_gradients(grads_and_vars)
            with tf.device('/gpu:1'):
                # Compute the gradients for visual variables.
                grads_and_vars = self.optimizer.compute_gradients(self.loss, self.model_2.train_vars)
                # Ask the optimizer to apply the gradients.
                self.train_op_1 = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        # Initialize model saver
        self.saver = tf.train.Saver(max_to_keep=None)
        return iterat
    
    def _get_optimizer_variables(self, optimizer):
        
        optimizer_vars = [optimizer.get_slot(var, name)
                          for name in optimizer.get_slot_names() for var in self.train_vars if var is not None]
        
        optimizer_vars.extend(list(optimizer._get_beta_accumulators()))
        
        return optimizer_vars
    
    def _init_models(self, session):
        
        # if there is restore checkpoint restore all not single model
        if FLAGS.restore_checkpoint is not None:
            # Restore session from checkpoint
            self._restore_model(session)
        # initialize two stream
        elif FLAGS.visual_init_checkpoint is not None or FLAGS.acoustic_init_checkpoint is not None:
            # not to have uninitialized value if only one model is initialized
            session.run(tf.global_variables_initializer())
            # Initialize global step
            print('{}: {} - Initializing global step'.format(datetime.now(), FLAGS.exp_name))
            session.run(self.global_step.initializer)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
            
            # don't initialize with sgd and momentum
            # Initialize optimizer variables
            print('{}: {} - Initializing optimizer variables'.format(datetime.now(), FLAGS.exp_name))
            optimizer_vars = self._get_optimizer_variables(self.optimizer)
            optimizer_init_op = tf.variables_initializer(optimizer_vars)
            session.run(optimizer_init_op)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
            if FLAGS.acoustic_init_checkpoint is not None:
                # Initialize acoustic model
                print('{}: {} - Initializing model'.format(datetime.now(), FLAGS.exp_name))
                self.model_2.init_model(session, FLAGS.acoustic_init_checkpoint)
            if FLAGS.visual_init_checkpoint is not None:
                # Initialize visual model
                print('{}: {} - Initializing model'.format(datetime.now(), FLAGS.exp_name))
                self.model_1.init_model(session, FLAGS.visual_init_checkpoint)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
        else:
            # Initialize all variables
            print('{}: {} - Initializing full model'.format(datetime.now(), FLAGS.exp_name))
            session.run(tf.global_variables_initializer())
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
    
    def _restore_model(self, session):
        
        print('{}: {} - Restoring session'.format(datetime.now(), FLAGS.exp_name))
        # Restore model
        session.run(tf.global_variables_initializer())
        # FLAGS.model == 'DualCamHybridNet':
        to_exclude = [i.name for i in tf.global_variables()
                      if
                      'beta' in i.name or 'hear_net' in i.name or 'global_step' in i.name]  # or 'resnet_v1' in i.name
        var_list = slim.get_variables_to_restore(exclude=to_exclude)
        
        # else:
        #     var_list = slim.get_model_variables(self.model.scope)
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(session, FLAGS.restore_checkpoint)
        print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
    
    def train(self, train_data=None, valid_data=None):
        
        # Assert training and validation sets are not None
        assert train_data is not None
        assert valid_data is not None
        
        # compute loss and minimize
        train_iterat = self._build_functions(train_data)
        eval_iterat = valid_data.data.make_initializable_iterator()
        # Add the losses to summary
        self.logger.log_scalar('triplet_loss', self.tripletloss)
        self.logger.log_scalar('train_loss', self.loss)
        
        # Add the accuracy to the summary
        # self.logger.log_scalar('train_accuracy', self.accuracy)
        
        # Merge all summaries together
        self.logger.merge_summary()
        
        # Start training session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                              gpu_options=tf.GPUOptions(
                                                  allow_growth=True), )) as session:
            train_handle = session.run(train_iterat.string_handle())
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint
            self._init_models(session)
            
            # Add the model graph to TensorBoard
            self.logger.write_graph(session.graph)
            # Save model
            self._save_checkpoint(session, 'random')
            start_epoch = int(tf.train.global_step(session, self.global_step) / train_data.total_batches)
            best_epoch = -1
            best_accuracy = -1.0
            best_loss = 10000.0
            
            # For each epoch
            for epoch in range(start_epoch, start_epoch + self.num_epochs):
                # Initialize counters and stats
                step = 0
                
                # Initialize iterator over the training set
                # session.run(training_init_op)  , feed_dict={train_data.seed: epoch})
                session.run(train_iterat.initializer)
                # For each mini-batch
                while True:
                    try:
                        tripletlossvalue, train_loss, train_summary, _, _ = session.run(  # train_accuracy,
                            [self.tripletloss, self.loss, self.logger.summary_op, self.train_op_0, self.train_op_1],  #
                            feed_dict={self.handle: train_handle,  # self.accuracy,
                                       self.epoch: epoch,
                                       self.model_1.network['is_training']: 1,
                                       self.model_2.network['is_training']: 1,
                                       self.model_1.network['keep_prob']: 0.5,
                                       self.model_2.network['keep_prob']: 0.5})
                        
                        # Compute mini-batch error
                        if step % self.display_freq == 0:
                            print('{}: {} - Iteration: [{:3}]\t Training_Loss: {:6f}\t Triplet_Loss: {:6f}'.format(
                                # \t Training_Accuracy: {:6f}
                                datetime.now(), FLAGS.exp_name, step, train_loss, tripletlossvalue))  # , train_accuracy
                            self.logger.write_summary(train_summary, tf.train.global_step(session, self.global_step))
                            self.logger.flush_writer()
                        # Update counters and stats
                        step += 1
                    
                    except tf.errors.OutOfRangeError:
                        break
                
                # Evaluate model on validation set
                session.run(eval_iterat.initializer)
                total_loss, total_accuracy = self._evaluate(session, 'validation', evaluation_handle)  #
                
                print('{}: {} - Epoch: {}\t Triplet_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                     FLAGS.exp_name,
                                                                                                     epoch,
                                                                                                     total_loss,
                                                                                                     total_accuracy))
                
                self.logger.write_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=total_loss),
                    tf.Summary.Value(tag="valid_accuracy", simple_value=total_accuracy)
                ]), epoch)
                
                self.logger.flush_writer()
                # if multiple of 10 epochs save model
                if epoch % 1 == 0:
                    best_epoch = epoch
                    best_accuracy = total_accuracy
                    best_loss = tripletlossvalue
                    # Save model
                    self._save_checkpoint(session, epoch)
                    with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/model_{}.txt".format(epoch),
                              "w") as outfile:
                        outfile.write(
                            '{}: {} - Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(
                                datetime.now(),
                                FLAGS.exp_name,
                                best_epoch,
                                best_loss, best_accuracy))
                
                # if accuracy or loss decrease save model
                if total_accuracy >= best_accuracy and round(tripletlossvalue, 3) <= round(best_loss, 3):
                    best_epoch = epoch
                    best_accuracy = total_accuracy
                    best_loss = tripletlossvalue
                    # Save model
                    name = 'best'
                    self._save_checkpoint(session, name)
                    with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/model_{}.txt".format(name),
                              "w") as outfile:
                        outfile.write(
                            '{}: {} - Best Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(
                                datetime.now(),
                                FLAGS.exp_name,
                                best_epoch,
                                best_loss, best_accuracy))  #
            print('{}: {} - Best Epoch: {}\t Triplet_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                      FLAGS.exp_name,
                                                                                                      best_epoch,
                                                                                                      best_loss,
                                                                                                      best_accuracy))
    
    def _save_checkpoint(self, session, epoch):
        
        checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
        model_name = 'model_{}.ckpt'.format(epoch)
        print('{}: {} - Saving model to {}/{}'.format(datetime.now(), FLAGS.exp_name, checkpoint_dir, model_name))
        
        self.saver.save(session, '{}/{}'.format(checkpoint_dir, model_name))
    
    def _pairwise_distances(self, embeddings0, embeddings1, squared=True):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product0 = tf.matmul(embeddings0, tf.transpose(embeddings0))
        dot_product1 = tf.matmul(embeddings1, tf.transpose(embeddings1))
        dot_productab = tf.matmul(embeddings0, tf.transpose(embeddings1))
        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm0 = tf.diag_part(dot_product0)
        square_norm1 = tf.diag_part(dot_product1)
        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm0, 0) - 2.0 * dot_productab + tf.expand_dims(square_norm1, 1)
        
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)
        
        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.to_float(tf.equal(distances, 0.0))
            distances = distances + mask * 1e-16
            
            distances = tf.sqrt(distances)
            
            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)
        
        return distances
    
    def _get_anchor_positive_and_negative_triplet_mask(self, labels, scenario):
        """Return a 2D mask where mask[a, p] is True iff a and p have same label and scenario.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Also if i and j are not distinct is ok because we are considering audio and video embeddings
        # indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        # indices_not_equal = tf.logical_not(indices_equal)
        
        # Check if labels[i] == labels[j] and scenario are equal
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        scenario_equal = tf.equal(tf.expand_dims(scenario, 0), tf.expand_dims(scenario, 1))
        
        # Combine the two masks
        mask = tf.logical_and(scenario_equal, labels_equal)
        
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels or scenario.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k] or different scenario
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        maskneg0 = tf.logical_not(labels_equal)
        maskneg1 = tf.logical_not(scenario_equal)
        maskneg = tf.logical_or(maskneg0, maskneg1)
        return mask, maskneg
    
    def _get_triplet_mask(self, labels, scenario):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - video[i] == video[j] and  video[i] != video[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check if video[i] == video[j] and video[i] != video[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        scenario_equal = tf.equal(tf.expand_dims(scenario, 0), tf.expand_dims(scenario, 1))
        # Combine the three masks
        same_video = tf.logical_and(scenario_equal, label_equal)
        i_equal_j = tf.expand_dims(same_video, 2)
        i_equal_k = tf.expand_dims(same_video, 1)
        
        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
        
        return valid_labels
    
    # compute closest embedding  negative and furthest positive for each batch
    # negative has different label, or person or location
    # positive has same label, person and location
    def mix_data_hard(self, data0, data1, labels, scenario, margin):  # acoustic_data and video_data
        # compute distances
        pairwise_dist = tf.reduce_sum(tf.square(data0 - data1), -1)
        # pairwise_dist = self._pairwise_distances(data0, data1)
        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label, person and location)
        mask_anchor_positive, mask_anchor_negative = self._get_anchor_positive_and_negative_triplet_mask(labels,
                                                                                                         scenario)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)
        # We put to 0 any element where (a, p) is not valid (valid if is the same video)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keep_dims=True)
        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels, or location, or person)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n) or location, or person)
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keep_dims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        
        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keep_dims=True)
        
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        
        mask = self._get_triplet_mask(labels, scenario)
        mask = tf.to_float(mask)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)
        
        return triplet_loss, fraction_positive_triplets
    
    def mix_all(self, data0, data1, labels, scenario, margin):
        """Build the triplet loss over a batch of embeddings.

        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        # pairwise_dist = self._pairwise_distances(data0, data1)
        # compute distances
        pairwise_dist = tf.reduce_sum(tf.square(data0 - data1), -1)
        
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        
        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        
        # Put to zero the invalid triplets
        # (where video(a) != video(p) or video(n) == video(a))
        mask = self._get_triplet_mask(labels, scenario)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)
        
        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
        
        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        
        return triplet_loss, fraction_positive_triplets
    
    def _valid(self, session, evaluation_handle):
        return self._evaluate(session, 'validation', evaluation_handle)
    
    def _evaluate(self, session, mod, eval_handle):
        _, acc = self.mix_all(self.productvectnorm, self.productvectnorm2, self.labels,
                              self.scenario, FLAGS.margin)
        # Initialize counters and stats
        loss_sum = 0.0
        accuracy_sum = 0.0
        data_set_size = 0  # 0.0
        label = []
        pred = []
        # For each mini-batch
        while True:
            try:
                # Compute batch loss and accuracy
                # compute accuracy with corresponding vectors
                batch_loss, batch_accuracy, labels_data = session.run(  # labels_data, batch_pred, batch_accuracy,
                    [self.loss, acc, self.labels],
                    # self.labels, self.batch_pred, self.accuracy,
                    feed_dict={self.handle: eval_handle,
                               self.epoch: 0,
                               self.model_1.network['is_training']: 0,
                               self.model_2.network['is_training']: 0,
                               self.model_1.network['keep_prob']: 1.0,
                               self.model_2.network['keep_prob']: 1.0})

                batch_accuracy = 1.0 - batch_accuracy
                # Update counters
                data_set_size += np.shape(labels_data)[0]  # 1 labels_data.shape[0]
                loss_sum += batch_loss * np.shape(labels_data)[0]  # labels_data.shape[0]
                accuracy_sum += batch_accuracy * np.shape(labels_data)[0]
            except tf.errors.OutOfRangeError:
                break
        # print (data_set_size)
        print (loss_sum)
        # print (accuracy_sum)
        total_loss = loss_sum / float(data_set_size)
        total_accuracy = accuracy_sum / float(data_set_size)
        return total_loss, total_accuracy
    
    def _retrieve_batch(self, next_batch):
        
        if FLAGS.model_1 == 'ResNet18_v1':
            data_1 = tf.reshape(next_batch[2], shape=[-1, self.shape_1[0], self.shape_1[1], self.shape_1[2]])
        elif FLAGS.model_1 == 'DualCamHybridNet':
            data_1 = tf.reshape(next_batch[0], shape=[-1, self.shape_1[0], self.shape_1[1], self.shape_1[2]])
        else:
            raise ValueError('Unknown model type')

        if FLAGS.model_2 == 'HearNet':
            data_2 = tf.reshape(next_batch[1], shape=[-1, self.shape_2[0], self.shape_2[1], self.shape_2[2]])
        elif FLAGS.model_2 == 'DualCamHybridNet':
            data_2 = tf.reshape(next_batch[0], shape=[-1, self.shape_2[0], self.shape_2[1], self.shape_2[2]])
        else:
            raise ValueError('Unknown model type')
        labels = tf.reshape(next_batch[3], shape=[-1, 10])
        scenario = tf.reshape(next_batch[4], shape=[-1, 61])
        return data_1, data_2, labels, scenario
    
    def test(self, test_data=None):
        
        # Assert testing set is not None
        assert test_data is not None
        eval_iterat = self._build_functions(test_data)
        # Start training session
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:  # allow_growth
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint if given
            self._restore_model(session)
            session.run(eval_iterat.initializer)
            # Evaluate model over the testing set
            test_loss, test_accuracy = self._evaluate(session, 'test', evaluation_handle)
        
        print('{}: {} - Testing_Loss: {:6f}\t Testing_Accuracy: {:6f}'.format(datetime.now(),
                                                                              FLAGS.exp_name,
                                                                              test_loss, test_accuracy))
        
        return test_loss, test_accuracy
