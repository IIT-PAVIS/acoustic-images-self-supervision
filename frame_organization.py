# sys is used for extracting script arguments from bash
import sys
# glob is used to list files in directories using regex
import glob
# os is used to obtain script directory
import os
from traceback import print_exc
from operator import itemgetter
import argparse

# extracting data location (path)
# if no argument is passed, then there' s only script name in sys.argv
parser = argparse.ArgumentParser()
parser.add_argument('root_dir', help='directory where classes are located', type=str)
#The directory contains sync and mat files
#parser.add_argument('out_dir', help='directory containing training, testing and validation video', type=str)

parsed_args = parser.parse_args()
#path of dataset
path = os.path.abspath(parsed_args.root_dir)
#directory where to save lists
out_dir = path + '/' + 'lists'

try:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
except:
    print('error during folder creation:')
    print_exc()

classes_dir = glob.glob(path + '/class_*/')

video_time_filename = 'video_time.txt'
#read minimum time, all videos have same length
class_time_filename = 'class_time.txt'# 'timedataset.txt'
training_filename = 'training_percentage.txt'
validation_filename = 'validation_percentage.txt'
testing_filename = 'testing_percentage.txt'
training_tfrecord_filename = 'training.txt'
validation_tfrecord_filename = 'validation.txt'
testing_tfrecord_filename = 'testing.txt'
total_filename = 'total_percentage.txt'
sum_training = 0
sum_testing = 0
sum_validation = 0

total = 0
tfrecord=1
try:
    #files containing folders
    f_training = open(out_dir + '/' + training_filename, 'w')
    f_validation = open(out_dir + '/' + validation_filename, 'w')
    f_testing = open(out_dir + '/' + testing_filename, 'w')
    #files containing tfrecords
    if tfrecord:
        f_tfrecord_training = open(out_dir + '/' + training_tfrecord_filename, 'w')
        f_tfrecord_validation = open(out_dir + '/' + validation_tfrecord_filename, 'w')
        f_tfrecord_testing = open(out_dir + '/' + testing_tfrecord_filename, 'w')
    #files containing duration and percentage
    f_total = open(out_dir + '/' + total_filename, 'w')
except:
    print('error while opening file: ')
    print_exc()

info_time = '' #temp string to store time extracted from files

for c in classes_dir:
    time = {'avaiable':0, 'training':0, 'validation':0, 'testing':0}
    training_max = 0 #max training time for each class

    training_list = []
    validation_list = []
    testing_list = []
    video_list = []
    training_tfrecords_list = []
    validation_tfrecord_list = []
    testing_tfrecord_list = []

    try:
        fc = open(c + '/' + class_time_filename, 'r')
        info_time = fc.readline().split(':')[1]
        info_time = info_time.split(' in')[0]
        #time available is minimum time for all classes
        time['avaiable'] = int(info_time.strip())
        total += time['avaiable']
        fc.close()
    except:
        print('error during reading class_time.txt for ' + c)
    #70% for training, 15% validation and 15% testing
    #maximum number of seconds for training
    training_max = int((time['avaiable'] / 100) * 75)
    #maximum number of seconds for validation, testing
    validation_max = int((time['avaiable'] - training_max) / 2)

    data_dir = glob.glob(c+'/data_*/')
    try:
        for d in data_dir:
                fd = open(d + video_time_filename, 'r')
                info_time = fd.readline().split(':')[1]
                fd.close()
                video_time = int(info_time.strip())
                data_name = d.split('/')[-2]
                video_list.append((data_name, video_time))
    except:
        print('error during reading video_time.txt for ' + c)
        print_exc()

    video_list.sort(key=itemgetter(1), reverse=True)

    for v in video_list:
        if (time['training'] + v[1] < training_max or len(training_list) == 0):
            time['training'] = time['training'] + v[1]
            #append name of folder and duration
            training_list.append((v[0], v[1]))
            #append all names of all tfrecords for every second of folder and path
            if tfrecord:
                tot_frames = glob.glob(c + v[0] + '/*.tfrecord')
                tot_frames.sort()
                training_tfrecords_list += tot_frames

        elif (time['validation'] + v[1] < validation_max or len(validation_list) == 0):
            time['validation'] = time['validation'] + v[1]
            #append name of folder and duration
            validation_list.append((v[0], v[1]))
            #append all names of all tfrecords for every second of folder and path
            if tfrecord:
                tot_frames = glob.glob(c + v[0] + '/*.tfrecord')
                tot_frames.sort()
                validation_tfrecord_list += tot_frames
                
        else:#elif (time['testing'] + v[1] < validation_max or len(testing_list) == 0):
            time['testing'] = time['testing'] + v[1]
            #append name of folder and duration
            testing_list.append((v[0], v[1]))
            #append all names of all tfrecords for every second of folder and path
            if tfrecord:
                tot_frames = glob.glob(c + v[0] + '/*.tfrecord')
                tot_frames.sort()
                testing_tfrecord_list += tot_frames
            
    sum_training_class = 0
    sum_testing_class = 0
    sum_validation_class = 0
    
    for v in training_list:
        f_training.write(c + v[0] +'\n')
        sum_training_class += v[1]

    for v in validation_list:
        f_validation.write(c + v[0] +'\n')
        sum_validation_class += v[1]

    for v in testing_list:
        f_testing.write(c + v[0] +'\n')
        sum_testing_class += v[1]
    if tfrecord:
        for v in training_tfrecords_list:
            f_tfrecord_training.write(v +'\n')
    
        for v in validation_tfrecord_list:
            f_tfrecord_validation.write(v +'\n')
    
        for v in testing_tfrecord_list:
            f_tfrecord_testing.write(v +'\n')
        
    f_total.write(c + 'training_class' + ' ' + str(sum_training_class) + ' ' + str(sum_training_class*1.0/time['avaiable']) +' ' +
                  'testing_class' + ' ' + str(sum_testing_class) + ' ' + str(sum_testing_class*1.0/time['avaiable']) +' ' +
                  'validation_class' + ' ' + str(sum_validation_class) + ' ' + str(sum_validation_class*1.0/time['avaiable']) + ' '+
                  str(sum_training_class + sum_testing_class +sum_validation_class ) +' '+'\n' )
    sum_validation += sum_validation_class
    sum_testing += sum_testing_class
    sum_training += sum_training_class
f_total.write('\n' + 'training' + ' ' + str(sum_training) + ' ' + str(sum_training*1.0/total) +' ' +
                  'testing' + ' ' + str(sum_testing) + ' ' + str(sum_testing*1.0/total) +' ' +
                  'validation' + ' ' + str(sum_validation) + ' ' + str(sum_validation*1.0/total) + '\n')

#debug
'''print(video_list)f_info.write('testing:\n')
        for v in testing_list:
            f_info.write('\t' + v +'\n')
print(training_list)
print(validation_list)
print(testing_list)'''