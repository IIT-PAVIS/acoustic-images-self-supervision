from datetime import datetime

from models.base import buildAccuracy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

flags = tf.app.flags
FLAGS = flags.FLAGS
_FRAMES_PER_SECOND = 12

class Trainer(object):
    
    def __init__(self, model, logger=None, display_freq=1,
                 learning_rate=0.0001, num_classes=14, num_epochs=1, nr_frames=12, temporal_pooling=False):
        
        self.model = model
        self.logger = logger
        
        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.nr_frames = nr_frames
        self.temporal_pooling = temporal_pooling
        self.shape = [self.model.height, self.model.width, self.model.channels]

    def _build_functions(self, data):
        self.handle = tf.placeholder(tf.string, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle, data.data.output_types,
                                                       data.data.output_shapes)
        iterat = data.data.make_initializable_iterator()
        next_batch = iterator.get_next()
        # give directly batch tensor depending on the network reshape
        in_data, self.labels = self._retrieve_batch(next_batch)
        self.model._build_model(in_data)
        if FLAGS.model == 'ResNet18_v1' and self.temporal_pooling:
            # temporal pooling gives one predition for nr_frames, if it is not we have one predicition for frame
            expanded_shape = [-1, self.nr_frames, self.num_classes]
            self.logits = tf.reduce_mean(tf.reshape(self.model.output, shape=expanded_shape), axis=1)
        elif FLAGS.model == 'DualCamHybridNet' and self.temporal_pooling:
            expanded_shape = [-1, FLAGS.sample_length*_FRAMES_PER_SECOND, self.num_classes]
            self.logits = tf.reduce_mean(tf.reshape(self.model.output, shape=expanded_shape), axis=1)
        else:
            self.logits = self.model.output
        # Define loss
        self.cross_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.labels,
            logits=self.logits,
            scope='cross_loss'
        )
        self.loss = tf.losses.get_total_loss()
        
        # Define accuracy
        self.accuracy = buildAccuracy(self.logits, self.labels)
        
        # Initialize counters and stats
        self.global_step = tf.train.create_global_step()
        
        # Define optimizer
        #before different
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(loss=self.loss,
                                                  global_step=self.global_step,
                                                  var_list=self.model.train_vars)
        
        # Initialize model saver
        self.saver = tf.train.Saver(max_to_keep=None)
        return iterat
    
    def _get_optimizer_variables(self, optimizer):
        
        optimizer_vars = [optimizer.get_slot(var, name)
                          for name in optimizer.get_slot_names() for var in self.model.train_vars if var is not None]
        
        optimizer_vars.extend(list(optimizer._get_beta_accumulators()))
        
        return optimizer_vars
    
    def _init_model(self, session):
        
        if FLAGS.init_checkpoint is not None:
            # not to have uninitialized value
            #session.run(tf.global_variables_initializer())
            # Initialize global step
            print('{}: {} - Initializing global step'.format(datetime.now(), FLAGS.exp_name))
            session.run(self.global_step.initializer)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
            
            #don't initialize with sgd and momentum
            # Initialize optimizer variables
            print('{}: {} - Initializing optimizer variables'.format(datetime.now(), FLAGS.exp_name))
            optimizer_vars = self._get_optimizer_variables(self.optimizer)
            optimizer_init_op = tf.variables_initializer(optimizer_vars)
            session.run(optimizer_init_op)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
            
            # Initialize model
            print('{}: {} - Initializing model'.format(datetime.now(), FLAGS.exp_name))
            self.model.init_model(session, FLAGS.init_checkpoint)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
        elif FLAGS.restore_checkpoint is not None:
            # Restore session from checkpoint
            self._restore_model(session)
        else:
            # Initialize all variables
            print('{}: {} - Initializing full model'.format(datetime.now(), FLAGS.exp_name))
            session.run(tf.global_variables_initializer())
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
    
    def _restore_model(self, session):
        
        # Restore model
        print('{}: {} - Restoring session'.format(datetime.now(), FLAGS.exp_name))
        #
        session.run(tf.global_variables_initializer())
        if FLAGS.mode == "train":
            if FLAGS.model == 'ResNet18_v1':
                to_exclude = [i.name for i in tf.global_variables()
                              if '/Adam' in i.name
                              or '/logits' in i.name or 'beta' in i.name
                              ]
            elif FLAGS.model == 'HearNet':
                to_exclude = [i.name for i in tf.global_variables()
                          if '/Adam' in i.name or 'beta' in i.name or 'fc3' in i.name]
            else: #FLAGS.model == 'DualCamHybridNet':
                to_exclude = [i.name for i in tf.global_variables()
                              if '/Adam' in i.name or 'beta' in i.name or '/full1' in i.name]
                         #only to finetune dualcamhybridmodel
        else:
            to_exclude = [i.name for i in tf.global_variables()
                              if '/Adam' in i.name or 'beta' in i.name]
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

        train_iterat = self._build_functions(train_data)
        eval_iterat = valid_data.data.make_initializable_iterator()
        # Add the losses to summary
        self.logger.log_scalar('cross_entropy_loss', self.cross_loss)
        self.logger.log_scalar('train_loss', self.loss)
        
        # Add the accuracy to the summary
        self.logger.log_scalar('train_accuracy', self.accuracy)
        
        # Merge all summaries together
        self.logger.merge_summary()
        
        # Start training session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              gpu_options=tf.GPUOptions(allow_growth=True))) as session:
            train_handle = session.run(train_iterat.string_handle())
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint
            self._init_model(session)
            
            # Add the model graph to TensorBoard
            self.logger.write_graph(session.graph)
            self._save_checkpoint(session, 'random')
            start_epoch = int(tf.train.global_step(session, self.global_step) / train_data.total_batches)
            best_epoch = -1
            best_accuracy = -1.0
            best_loss = -1.0
            
            # For each epoch
            for epoch in range(start_epoch, start_epoch + self.num_epochs):
                # Initialize counters and stats
                step = 0
                
                # Initialize iterator over the training set
                session.run(train_iterat.initializer)
                
                # For each mini-batch
                while True:
                    try:
                        
                        # Forward batch through the network
                        train_loss, train_accuracy, train_summary, _ = session.run([self.loss, self.accuracy, self.logger.summary_op, self.train_step], feed_dict={self.handle: train_handle,
                                                                self.model.network['keep_prob']: 0.5,
                                                                self.model.network['is_training']: 1})
                        
                        # Compute mini-batch error
                        if step % self.display_freq == 0:

                            print('{}: {} - Iteration: [{:3}]\t Training_Loss: {:6f}\t Training_Accuracy: {:6f}'.format(
                                datetime.now(), FLAGS.exp_name, step, train_loss, train_accuracy))
                            
                            self.logger.write_summary(train_summary, tf.train.global_step(session, self.global_step))
                        
                        # Update counters and stats
                        step += 1
                    
                    except tf.errors.OutOfRangeError:
                        break
                        
                session.run(eval_iterat.initializer)
                # Evaluate model on validation set
                total_loss, total_accuracy = self._evaluate(session, 'validation', evaluation_handle)
                
                print('{}: {} - Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                        FLAGS.exp_name,
                                                                                                        epoch,
                                                                                                        total_loss,
                                                                                                        total_accuracy))
                
                self.logger.write_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=total_loss),
                    tf.Summary.Value(tag="valid_accuracy", simple_value=total_accuracy)
                ]), epoch)
                
                self.logger.flush_writer()
                #if accuracy or loss decrease save model
                if total_accuracy > best_accuracy or (total_accuracy == best_accuracy and total_loss <= best_loss)\
                        or epoch == 2 or epoch == 10 or epoch == 20:
                    best_epoch = epoch
                    best_accuracy = total_accuracy
                    best_loss = total_loss
                    # Save model
                    self._save_checkpoint(session, epoch)
                    with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/model.txt", "w") as outfile:
                        outfile.write(
                            '{}: {}\nBest Epoch: {}\nValidation_Loss: {:6f}\nValidation_Accuracy: {:6f}\n'.format(
                                datetime.now(),
                                FLAGS.exp_name,
                                best_epoch,
                                best_loss,
                                best_accuracy))
            print('{}: {} - Best Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                         FLAGS.exp_name,
                                                                                                         best_epoch,
                                                                                                         best_loss,
                                                                                                         best_accuracy))
    
    def _save_checkpoint(self, session, epoch):
        
        checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
        model_name = 'epoch_{}.ckpt'.format(epoch)
        print('{}: {} - Saving model to {}/{}'.format(datetime.now(), FLAGS.exp_name, checkpoint_dir, model_name))
        
        self.saver.save(session, '{}/{}'.format(checkpoint_dir, model_name))
    
    def _valid(self, session,  evaluation_handle):
        return self._evaluate(session, 'validation', evaluation_handle)
    
    def _evaluate(self, session, mod, eval_handle):
        
        # Initialize counters and stats
        loss_sum = 0
        accuracy_sum = 0
        data_set_size = 0
        label = []
        pred = []
        # For each mini-batch
        while True:
            try:

                # Compute batch loss and accuracy
                one_hot_labels, logits, batch_loss, batch_accuracy = session.run(
                    [self.labels, self.logits, self.loss, self.accuracy],
                    feed_dict={self.handle: eval_handle,
                               self.model.network['keep_prob']: 1.0,
                               self.model.network['is_training']: 0})
                # find argmax of one hot labels
                label1 = np.argmax(one_hot_labels, 1)
                label = np.concatenate((label, label1), axis=0)
                # find argmax of logit
                pred1 = np.argmax(logits, 1)
                pred = np.concatenate((pred, pred1), axis=0)

                # Update counters
                data_set_size += one_hot_labels.shape[0]
                loss_sum += batch_loss * one_hot_labels.shape[0]
                accuracy_sum += batch_accuracy * one_hot_labels.shape[0]
            
            except tf.errors.OutOfRangeError:
                break
        
        total_loss = loss_sum / data_set_size
        total_accuracy = accuracy_sum / data_set_size
        # if mod == 'test':
        #     self.plot_confusion_matrix(pred, label)
        return total_loss, total_accuracy
    
    def _retrieve_batch(self, next_batch):
        
        if FLAGS.model == 'ResNet18_v1':
            data = tf.reshape(next_batch[2], shape=[-1, self.shape[0], self.shape[1], self.shape[2]])
            if self.temporal_pooling:
                labels = tf.reshape(next_batch[3], shape=[-1, self.num_classes])
            else:
                # Replicate labels to match the number of frames
                multiples = [1, self.nr_frames]
                labels = tf.reshape(tf.tile(next_batch[3], multiples), shape=[-1, self.num_classes])
        elif FLAGS.model == 'DualCamHybridNet':
            data = tf.reshape(next_batch[0], shape=[-1, self.shape[0], self.shape[1], self.shape[2]])
            if self.temporal_pooling:
                labels = tf.reshape(next_batch[3], shape=[-1, self.num_classes])
            else:
                # Replicate labels to match the number of frames
                multiples = [1, FLAGS.sample_length * _FRAMES_PER_SECOND]
                labels = tf.reshape(tf.tile(next_batch[3], multiples), shape=[-1, self.num_classes])
        elif FLAGS.model == 'HearNet' or FLAGS.model == 'SoundNet5':
            data = tf.reshape(next_batch[1], shape=[-1, self.shape[0], self.shape[1], self.shape[2]])
            labels = tf.reshape(next_batch[3], shape=[-1, self.num_classes])
        else:
            raise ValueError('Unknown model type')
        
        return data, labels
    
    def test(self, test_data=None):
        
        # Assert testing set is not None
        assert test_data is not None
        eval_iterat = self._build_functions(test_data)
        
        # Start training session
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:
            evaluation_handle = session.run(eval_iterat.string_handle())
            # Initialize model either randomly or with a checkpoint if given
            self._restore_model(session)
            session.run(eval_iterat.initializer)
            # Evaluate model over the testing set
            test_loss, test_accuracy = self._evaluate(session, 'test', evaluation_handle)
        
            with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/test_accuracy.txt", "w") as outfile:
                    outfile.write('{}: {} - Testing_Loss: {:6f}\nTesting_Accuracy: {:6f}'.format(datetime.now(),
                                                                              FLAGS.exp_name,
                                                                              test_loss,
                                                                              test_accuracy))
        print('{}: {} - Testing_Loss: {:6f}\nTesting_Accuracy: {:6f}'.format(datetime.now(),
                                                                              FLAGS.exp_name,
                                                                              test_loss,
                                                                              test_accuracy))
        
        return test_loss, test_accuracy
    
    def plot_confusion_matrix(self, pred, label, normalize=True,
                              title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        counter = 0
        cmap = plt.cm.Blues
        cm = confusion_matrix(label, pred)
        percentage2 = label.shape[0]
        for i in range(percentage2):
            if (pred[i] == label[i]):
                counter += 1
        
        perc = counter / float(percentage2)
        print(perc)
        # classes = ['Clapping', 'Snapping fingers', 'Speaking', 'Whistling', 'Playing kendama', 'Clicking', 'Typing',
        #            'Knocking', 'Hammering', 'Peanut breaking', 'Paper ripping', 'Plastic crumpling', 'Paper shaking',
        #            'Stick dropping']
        classes = ['Train', 'Boat', 'Drone', 'Fountain', 'Drill',
                   'Razor', 'Hair dryer', 'Vacuumcleaner', 'Cart', 'Traffic']
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        print(cm)
        # cmap = plt.cm.get_cmap('Blues')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + '/confusion_matrix.png')