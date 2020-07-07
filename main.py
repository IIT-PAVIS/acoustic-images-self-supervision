import tensorflow as tf
from datetime import datetime
from logger.logger import Logger
from models.vision import ResNet18_v1
from models.audition import HearModel
from models.audition import DualCamHybridModel
from trainer.trainer import Trainer as Trainer
from trainer.trainer_andres import Trainer as TrainerDistillation
from trainer.trainer_three import Trainer as TrainerTripletLoss
from trainer.trainer_audio import Trainer as TrainerAudio
from dataloader.actions_data import ActionsDataLoader as DataLoader

flags = tf.app.flags
flags.DEFINE_string('mode', None, 'Execution mode, it can be either \'train\' or \'test\'')
flags.DEFINE_string('model', None, 'Model type, it can be one of \'SeeNet\', \'ResNet50\', \'TemporalResNet50\', '
                                   '\'DualCamNet\', \'DualCamHybridNet\', \'SoundNet5\', or \'HearNet\'')
flags.DEFINE_string('train_file', None, 'Path to the plain text file for the training set')
flags.DEFINE_string('valid_file', None, 'Path to the plain text file for the validation set')
flags.DEFINE_string('test_file', None, 'Path to the plain text file for the testing set')
flags.DEFINE_string('exp_name', None, 'Name of the experiment')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_string('visual_init_checkpoint', None, 'Checkpoint file for visual model initialization')
flags.DEFINE_string('acoustic_init_checkpoint', None, 'Checkpoint file for acoustic model initialization')
flags.DEFINE_string('restore_checkpoint', None, 'Checkpoint file for session restoring')
flags.DEFINE_integer('batch_size', 4, 'Size of the mini-batch')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('display_freq', 1, 'How often must be shown training results')
flags.DEFINE_integer('num_epochs', 100, 'Number of iterations through dataset')
flags.DEFINE_integer('total_length', 30, 'Length in seconds of a full sequence')
#sample length is 1 s for dualcamnet and 5 s for hearnet and soundnet
flags.DEFINE_integer('sample_length', 1, 'Length in seconds of a sequence sample')
#number of crops 30 for 1 s and 6 for 5 s
flags.DEFINE_integer('number_of_crops', 30, 'Number of crops')
flags.DEFINE_integer('buffer_size', 1, 'Size of pre-fetch buffer')
flags.DEFINE_string('tensorboard', None, 'Directory for storing logs')
flags.DEFINE_string('checkpoint_dir', None, 'Directory for storing models')
flags.DEFINE_integer('temporal_pooling', 1, 'Flag to indicate whether to use average pooling over time')
flags.DEFINE_integer('embedding', 1, 'Say if you are training 128 vectors')
flags.DEFINE_string('model_1', None, 'Model type, it can be one of \'ResNet18_v1\', \'ResNet18\', \'ResNet50\', \'TemporalResNet50\' \'DualCamNet\', \'DualCamHybridNet\', ')
flags.DEFINE_string('model_2', None, 'Model type, it can be one of \'DualCamNet\', \'DualCamHybridNet\', \'SoundNet5\', or \'HearNet\'')
flags.DEFINE_float('margin', 0.2, 'margin') # between 0 and 11 for 128 vector
flags.DEFINE_integer('transfer', 0, 'Say if you are doing transfer')
flags.DEFINE_integer('distillation', 0, 'Say if you are doing distillation')
# in temporal models TemporalResNet50 and ResNet18
flags.DEFINE_integer('block_size', 1, 'Number of frames to pick randomly for each second') #12
flags.DEFINE_string('loss', 'Triplet', 'Loss type, it can be one of \'Triplet\', \'Contrastive\'')
flags.DEFINE_integer('num_class', 128, 'Classes')
# Disable number of channels
# flags.DEFINE_integer('num_channels', 512, 'Number of channels from the acoustic images')
# Disable version
# flags.DEFINE_integer('version', None, 'Network version')
flags.DEFINE_float('alpha', 0.1, 'How much weighting the loss')
FLAGS = flags.FLAGS


def main(_):
    # Instantiate logger
    if FLAGS.transfer == 0:
        transfer = False
    else:
        transfer = True
    logger = Logger('{}/{}'.format(FLAGS.tensorboard, FLAGS.exp_name))

    # Create data loaders according to the received program arguments
    print('{}: {} - Creating data loaders'.format(datetime.now(), FLAGS.exp_name))

    # random_pick = (FLAGS.model == 'TemporalResNet50' or FLAGS.model_1 == 'TemporalResNet50') or (FLAGS.model == 'ResNet18' or FLAGS.model_1 == 'ResNet18')
    # if we are randomly picking total number of frames, we can set random pick to False
    nr_frames = FLAGS.block_size * FLAGS.sample_length
    if (FLAGS.model == 'ResNet18_v1' or FLAGS.model == 'ResNet50' or FLAGS.model_1 == 'ResNet18_v1' or FLAGS.model_1 == 'ResNet50') and nr_frames < 12*FLAGS.sample_length:
        random_pick = True
    else:
        random_pick = False
    build_spectrogram = (FLAGS.model_2 == 'HearNet' or FLAGS.model == 'HearNet')
    normalize = (FLAGS.model_2 == 'HearNet' or FLAGS.model == 'HearNet')
   
    modalities = []

    if FLAGS.embedding:
        # model 1 is video
        # model2 is audio or acoustic images
        if transfer:
            modalities.append(0)
        if FLAGS.model_2 == 'DualCamNet' or FLAGS.model_2 == 'DualCamHybridNet' or FLAGS.model_1 == 'DualCamNet' or FLAGS.model_1 == 'DualCamHybridNet':
            modalities.append(0)
        if FLAGS.model_2 == 'SoundNet5' or FLAGS.model_2 == 'HearNet':
            modalities.append(1)
        if FLAGS.model_1 == 'ResNet50' or FLAGS.model_1 == 'ResNet18' or FLAGS.model_1 == 'ResNet18_v1' or FLAGS.model_1 == 'TemporalResNet50':
            modalities.append(2)
            
    else:
        if FLAGS.model == 'DualCamNet' or FLAGS.model == 'DualCamHybridNet':
            modalities.append(0)
        elif FLAGS.model == 'SoundNet5' or FLAGS.model == 'HearNet':
            modalities.append(1)
        elif FLAGS.model == 'SeeNet' or FLAGS.model == 'ResNet50'or FLAGS.model == 'ResNet18' or FLAGS.model == 'ResNet18_v1' or FLAGS.model == 'TemporalResNet50':
            modalities.append(2)
        elif FLAGS.model == 'AVModel':
            modalities.append(0)
            modalities.append(2)
    
            
    with tf.device('/cpu:0'):

        if FLAGS.train_file is None:
            train_data = None
        else:
            train_data = DataLoader(FLAGS.train_file, 'training', FLAGS.batch_size, num_epochs=1,
                                    total_length=FLAGS.total_length, sample_length=FLAGS.sample_length,
                                    number_of_crops=FLAGS.number_of_crops, buffer_size=FLAGS.buffer_size,
                                    shuffle=True, normalize=normalize, random_pick=random_pick,
                                    build_spectrogram=build_spectrogram, modalities=modalities, nr_frames=nr_frames)

        if FLAGS.valid_file is None:
            valid_data = None
        else:
            valid_data = DataLoader(FLAGS.valid_file, 'inference', FLAGS.batch_size, num_epochs=1,
                                    total_length=FLAGS.total_length, sample_length=FLAGS.sample_length,
                                    buffer_size=FLAGS.buffer_size, shuffle=False, normalize=normalize,
                                    random_pick=random_pick, build_spectrogram=build_spectrogram, modalities=modalities, nr_frames=nr_frames)

        if FLAGS.test_file is None:
            test_data = None
        else:
            test_data = DataLoader(FLAGS.test_file, 'inference', FLAGS.batch_size, num_epochs=1,
                                   total_length=FLAGS.total_length, sample_length=FLAGS.sample_length,
                                   buffer_size=FLAGS.buffer_size, shuffle=False, normalize=normalize,
                                   random_pick=random_pick, build_spectrogram=build_spectrogram, modalities=modalities, nr_frames=nr_frames)

    # Build model
    print('{}: {} - Building model'.format(datetime.now(), FLAGS.exp_name))
    
    if FLAGS.embedding:
        with tf.device('/gpu:0'):
        #for d in ['/gpu:1', '/gpu:0']:
            #with tf.device(d):
            if FLAGS.distillation:
                model_1 = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class, embedding=0)
                model_2 = HearModel(input_shape=[200, 1, 257], num_classes=FLAGS.num_class, embedding=0)
            elif transfer:
                model_1 = ResNet18_v1(input_shape=[224, 298, 3], num_classes=FLAGS.num_class, map=True)
                model_2 = HearModel(input_shape=[200, 1, 257], num_classes=FLAGS.num_class, embedding=FLAGS.embedding)
                model_transfer = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class, embedding=FLAGS.embedding)
            else:
                #visual model
                if FLAGS.model_1 == 'ResNet50':
                    model_1 = ResNet50Model(input_shape=[224, 298, 3], num_classes=FLAGS.num_class)
                elif FLAGS.model_1 == 'ResNet18_v1':
                    #map=True map of features, otherwise 10 classes
                    model_1 = ResNet18_v1(input_shape=[224, 298, 3], num_classes=FLAGS.num_class, map=True)
                elif FLAGS.model_1 == 'ResNet18':
                    model_1 = ResNet18Model(input_shape=[224, 298, 3], num_classes=FLAGS.num_class, nr_frames=nr_frames)
                elif FLAGS.model_1 == 'TemporalResNet50':
                    model_1 = ResNet50TemporalModel(input_shape=[224, 298, 3], num_classes=FLAGS.num_class, nr_frames=nr_frames)
                elif FLAGS.model_1 == 'DualCamHybridNet':
                    model_1 = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class,
                                                 embedding=FLAGS.embedding)
                elif FLAGS.model_1 == 'DualCamNet':
                    model_1 = DualCamModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class, version=10)
                else:
                     model_1 = ResNet50Model(input_shape=[224, 298, 3], num_classes=FLAGS.num_class)
                
                #audio or acoustic model
                if FLAGS.model_2 == 'DualCamHybridNet':
                    model_2 = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class, embedding=FLAGS.embedding)
                elif FLAGS.model_2 == 'DualCamNet':
                    model_2 = DualCamModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class, version=10)
                elif FLAGS.model_2 == 'HearNet':
                    model_2 = HearModel(input_shape=[200, 1, 257], num_classes=FLAGS.num_class, embedding=FLAGS.embedding)
                elif FLAGS.model_2 == 'SoundNet5':
                    #, checkpoint_file=FLAGS.checkpoint_dir+'/soundnet/soundnet5_final.t7'
                    model_2 = SoundNet5Model(input_shape=[22050 * 2, 1, 1], num_classes=FLAGS.num_class)
                else:
                    model_2 = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class)

        # Build trainer
        print('{}: {} - Building trainer'.format(datetime.now(), FLAGS.exp_name))
        if transfer:
            trainer = TrainerAudio(model_1, model_2, model_transfer, logger, display_freq=FLAGS.display_freq,
                                         learning_rate=FLAGS.learning_rate,
                                         num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling,
                                         nr_frames=nr_frames, num_classes=FLAGS.num_class)
        elif FLAGS.distillation:
            trainer = TrainerDistillation(model_1, model_2, logger,
                                   learning_rate=FLAGS.learning_rate,
                                   num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling,
                                   nr_frames=nr_frames, num_classes=FLAGS.num_class)
        elif FLAGS.loss == 'Triplet':
            trainer = TrainerTripletLoss(model_1, model_2, logger, display_freq=FLAGS.display_freq, learning_rate=FLAGS.learning_rate,
                          num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames, num_classes=FLAGS.num_class)
        else:
            raise ValueError('Unknown loss')
        
        if FLAGS.mode == 'train':
            checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
            if not tf.gfile.Exists(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
            # Train model
            with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/configuration.txt", "w") as outfile:
                
                outfile.write('Experiment: {} \nVisual model: {} \nAudio model: {} \nLearning_rate: {}\n'.format(FLAGS.exp_name, FLAGS.model_1, FLAGS.model_2,
                                                                                 FLAGS.learning_rate))
                outfile.write(
                    'Num_epochs: {} \nTotal_length: {} \nSample_length: {}\n'.format(FLAGS.num_epochs, FLAGS.total_length,
                                                                                       FLAGS.sample_length))
                outfile.write(
                    'Number_of_crops: {} \nMargin: {}\nNumber of classes: {}\n'.format(FLAGS.number_of_crops, FLAGS.margin, FLAGS.num_class))
                outfile.write(
                    'Block_size: {} \nLoss: {} \nEmbedding: {}\n'.format(FLAGS.block_size,
                                                                                FLAGS.loss,
                                                                                FLAGS.embedding))
                outfile.write(
                    'Train_file: {} \nValid_file: {} \nTest_file: {}\n'.format(FLAGS.train_file,
                                                                                     FLAGS.valid_file,
                                                                                     FLAGS.test_file))
                outfile.write(
                    'Mode: {} \nVisual_init_checkpoint: {} \nAcoustic_init_checkpoint: {} \nRestore_checkpoint: {}\n'.format(FLAGS.mode,
                                                                                       FLAGS.visual_init_checkpoint,
                                                                                       FLAGS.acoustic_init_checkpoint,
                                                                                       FLAGS.restore_checkpoint))
                outfile.write('Checkpoint_dir: {} \nLog dir: {} \nBatch_size: {}\n'.format(FLAGS.checkpoint_dir,   FLAGS.tensorboard, FLAGS.batch_size))
                
            print('{}: {} - Training started'.format(datetime.now(), FLAGS.exp_name))
            trainer.train(train_data=train_data, valid_data=valid_data)
        elif FLAGS.mode == 'test':
            # Test model
            print('{}: {} - Testing started'.format(datetime.now(), FLAGS.exp_name))
            trainer.test(test_data=test_data)
        else:
            raise ValueError('Unknown execution mode')
        
    else:
        with tf.device('/gpu:0'):
        #for d in ['/gpu:1', '/gpu:0']:
            #with tf.device(d):
            if FLAGS.model == 'ResNet50':
                model = ResNet50Model(input_shape=[224, 298, 3], num_classes=FLAGS.num_class)
            elif FLAGS.model == 'ResNet18_v1':
                model = ResNet18_v1(input_shape=[224, 298, 3], num_classes=FLAGS.num_class, map=False)
            elif FLAGS.model == 'ResNet18':
                model = ResNet18Model(input_shape=[224, 298, 3], num_classes=128, nr_frames=nr_frames)#64 or 128
            elif FLAGS.model == 'TemporalResNet50':
                model = ResNet50TemporalModel(input_shape=[224, 298, 3], num_classes=FLAGS.num_class, nr_frames=nr_frames)
            elif FLAGS.model == 'DualCamHybridNet':
                model = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=FLAGS.num_class, embedding=FLAGS.embedding)
            elif FLAGS.model == 'HearNet':
                model = HearModel(input_shape=[200, 1, 257], num_classes=FLAGS.num_class, embedding=FLAGS.embedding)
            elif FLAGS.model == 'SoundNet5':
                model = SoundNet5Model(input_shape=[22050 * 2, 1, 1], num_classes=FLAGS.num_class, checkpoint_file=FLAGS.checkpoint_dir+'/soundnet/soundnet5_final.t7')
            else:
                # Not necessary but set model to None to avoid warning about using unassigned local variable
                model = None
                raise ValueError('Unknown model type')

        # Build trainer
        print('{}: {} - Building trainer'.format(datetime.now(), FLAGS.exp_name))
        trainer = Trainer(model, logger, display_freq=FLAGS.display_freq, learning_rate=FLAGS.learning_rate, num_classes=FLAGS.num_class,
                                 num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling, nr_frames=nr_frames)
    
        if FLAGS.mode == 'train':
            checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
            if not tf.gfile.Exists(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
            # Train model
            with open('{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name) + "/configuration.txt", "w") as outfile:
                outfile.write('Experiment: {} \nBatch_size: {}\n'.format(FLAGS.exp_name,
                                                                                              FLAGS.batch_size))
                outfile.write(
                    'Model: {} \nLearning_rate: {}\nNumber of classes: {}\n'.format(FLAGS.model, FLAGS.learning_rate, FLAGS.num_class))
                outfile.write(
                    'Num_epochs: {} \nTotal_length: {} \nSample_length: {}\n'.format(FLAGS.num_epochs,
                                                                                     FLAGS.total_length,
                                                                                     FLAGS.sample_length))
                outfile.write(
                    'Number_of_crops: {} \nCheckpoint_dir: {} \nLog dir: {}\n'.format(FLAGS.number_of_crops, FLAGS.checkpoint_dir,
                                                                              FLAGS.tensorboard))
                outfile.write(
                    'Train_file: {} \nValid_file: {} \nTest_file: {}\n'.format(FLAGS.train_file,
                                                                                     FLAGS.valid_file,
                                                                                     FLAGS.test_file))
                outfile.write(
                    'Mode: {} \nInit_checkpoint: {} \nRestore_checkpoint: {}\n'.format(FLAGS.mode,
                                                                               FLAGS.init_checkpoint,
                                                                               FLAGS.restore_checkpoint))
            # Train model
            print('{}: {} - Training started'.format(datetime.now(), FLAGS.exp_name))
            trainer.train(train_data=train_data, valid_data=valid_data)
        elif FLAGS.mode == 'test':
            # Test model
            print('{}: {} - Testing started'.format(datetime.now(), FLAGS.exp_name))
            trainer.test(test_data=test_data)
        else:
            raise ValueError('Unknown execution mode')


if __name__ == '__main__':
    flags.mark_flags_as_required(['mode', 'exp_name'])
    tf.app.run()

    # --mode
    # train
    # --model
    # DualCamHybridNet
    # --train_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/training.txt"
    # --valid_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/validation.txt"
    # --test_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/testing.txt"
    # --exp_name
    # train_resnet
    # --batch_size
    # 16
    # --total_length
    # 30
    # --number_of_crops
    # 15
    # --sample_length
    # 2
    # --buffer_size
    # 10
    # --init_checkpoint
    # /data/vsanguineti/checkpoints/Nuno/model.ckpt-98277 /resnet/resnet_v1_50.ckpt
    # --tensorboard
    # /data/vsanguineti/tensorboard/
    # --checkpoint_dir
    # /data/vsanguineti/checkpoints2/
    # --num_epochs
    # 300
    # --learning_rate
    # 0.000001
    # --restore_checkpoint
    # /data/vsanguineti/checkpoints2/embeddingAcousticScalar2MapFarDifferentDot0.00001savemodel/model_100.ckpt
    # --temporal_pooling
    # True
    # --embedding
    # False

    # --mode
    # train
    # --model
    # ResNet18_v1
    # --train_file
    # / media / vsanguineti / TOSHIBAEXT / tfrecords / lists / training.txt
    # --valid_file
    # / media / vsanguineti / TOSHIBAEXT / tfrecords / lists / validation.txt
    # --test_file
    # / media / vsanguineti / TOSHIBAEXT / tfrecords / lists / testing.txt
    # --exp_name
    # ResNettfrecords
    # --batch_size
    # 2
    # --total_length
    # 2
    # --number_of_crops
    # 1
    # --sample_length
    # 2
    # --buffer_size
    # 10
    # --tensorboard
    # / home / vsanguineti / Documents / Code / audio - video / tensorboard /
    # --checkpoint_dir
    # / home / vsanguineti / Documents / Code / audio - video / checkpoints /
    # --num_epochs
    # 100
    # --learning_rate
    # 0.0001
    # --embedding
    # 0
    # --temporal_pooling
    # 1
    # --num_class
    # 10
    
    # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #Adam occupies more variables space
    # self.optimizer == 'Momentum'
    # lr_fn = learning_rate_with_decay(
    #     batch_size= FLAGS.batch_size, batch_denom=FLAGS.batch_size,
    #     # [30, 60, 80, 90],
    #     num_images=train_data.data_size, boundary_epochs=[20, 120, 160, 170],
    #     decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=True, base_lr=.0128)
    # learning_rate = lr_fn(self.global_step)
    #
    # self.optimizer = tf.train.MomentumOptimizer(
    #     learning_rate=learning_rate,
    #     momentum=0.9
    # )
    #learning rate can be a number

    #How to use with two models
    # --mode
    # train
    # --train_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/training.txt"
    # --valid_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/validation.txt"
    # --test_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/testing.txt"
    # --model_1
    # ResNet18_v1
    # --model_2
    # HearNet
    # --exp_name
    # train_resnet
    # --batch_size
    # 2
    # --total_length
    # 30
    # --number_of_crops
    # 15
    # --sample_length
    # 2
    # --buffer_size
    # 1
    # --learning_rate
    # 0.0001
    # --tensorboard
    # /data/vsanguineti/tensorboard/
    # --checkpoint_dir
    # /data/vsanguineti/checkpoints2/
    # --embedding
    # True
    # --temporal_pooling
    # True
    # --num_class
    # 128
    
    #How to use transfer
    # --mode
    # train
    # --train_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/training.txt"
    # --valid_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/validation.txt"
    # --test_file
    # "/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/testing.txt"
    # --model_1
    # ResNet18_v1
    # --model_2
    # HearNet
    # --exp_name
    # train_resnet
    # --batch_size
    # 1
    # --total_length
    # 30
    # --number_of_crops
    # 15
    # --sample_length
    # 2
    # --buffer_size
    # 1
    # --learning_rate
    # 0.0001
    # --tensorboard
    # / data / vsanguineti / tensorboard /
    # --checkpoint_dir
    # / data / vsanguineti / checkpoints2 /
    # --embedding
    # True
    # --temporal_pooling
    # True
    # --num_class
    # 128
    # --restore_checkpoint
    # / data / vsanguineti / checkpoints2 / embeddingAcousticScalar2MapFarDifferentDot0
    # .00001
    # vers2_1 / model_100.ckpt
    # --transfer
    # 1
    #--alpha
    #0.1