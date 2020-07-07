# ''' contare il numero di frame dentro la cartella per trovare
# quanto lungo il video in secondi, salvare un file txt sulla durata
# in secondi del video arrotondata per difetto (int) '''
#
# ''' nella cartella radice (la cartella class) salvare un file che contenga
# la somma delle durate dei video dei file salvati nella cartella data '''

# sys is used for extracting script arguments from bash
import sys
# glob is used to list files in directories using regex
import glob
# os is used to obtain script directory
import os

FRAMERATE = 12 #number of frames

# extracting data location (path)
# if no argument is passaed, then there' s only script name in sys.argv
if len(sys.argv) == 1:
    path = os.path.dirname(os.path.abspath(__file__))
    print('no arguments passed: using {}'.format(path))
elif sys.argv[1] == '-c':
    path = sys.argv[2]
else:
    path = sys.argv[1]

tfrecord = 1
#list containing classes' directories
classes_dir = glob.glob(path + '/class_*/')

# '''files that will contain single video length and sum of each video length
# of the class'''
video_time_filename = 'video_time.txt'
class_time_filename = 'class_time.txt'
datadir_filename = 'datadir.txt'
timedataset_filename = 'timedataset.txt'
video_filename = 'videodataset.txt'
num_datadir_max = 0
total_data_dir = 0
num_datadir_min = 1000
timemin = 32*60 #32 minutes
timemax = 0
totaltime = 0

timevideomin = 32*60
timevideomax = 0
totalvideotime = 0

for c in classes_dir:
    # seconds counter for each class
    class_seconds = 0

    # list containing data directory for each class
    data_dir = glob.glob(c+'/data_*/')
    num_datadir = len(data_dir)
    #max number
    if num_datadir > num_datadir_max:
        num_datadir_max = num_datadir
        class_datadir_max = c
    #minimum number
    if num_datadir < num_datadir_min:
        num_datadir_min = num_datadir
        class_datadir_min = c
    #total number of videos
    total_data_dir = total_data_dir + num_datadir

    for d in data_dir:
        
        video_dir = d + '/video'
        if tfrecord:
            tot_frames = len((glob.glob(d + '/*.tfrecord')))
            # using integer division
            video_seconds = tot_frames #// FRAMERATE
        else:
            tot_frames = len((glob.glob(video_dir + '/*.bmp')))
            # using integer division
            video_seconds = tot_frames // FRAMERATE
        #video seconds has o be miminum 2 seconds or divisible for 2
        #if it is 0 we don't add
        #if it is odd we subtract one
        if video_seconds % 2 != 0:
            video_seconds -= 1

        if video_seconds < timevideomin and video_seconds >= 2:
            timevideomin = video_seconds
            videotimemin = d
            classvideotimemin = c
        if video_seconds > timevideomax:
            timevideomax = video_seconds
            videotimemax = d
            classvideotimemax = c
        totalvideotime = totalvideotime + video_seconds

        class_seconds = class_seconds + video_seconds

        # exception management
        try:
            fv = open(d + video_time_filename, 'w')
            fv.write('video seconds: {}'.format(video_seconds))
            fv.close()
        except:
            print('error during writing video_time.txt for ' + d)
    try:
        fc = open(c + '/' + class_time_filename, 'w')
        fc.write('class seconds: {}'.format(class_seconds))
        fc.close()
    except:
        print('error during writing class_time.txt for ' + c)
    if class_seconds < timemin:
        timemin = class_seconds
        classtimemin = c
    if class_seconds > timemax:
        timemax = class_seconds
        classtimemax = c
    totaltime = totaltime + class_seconds
        
try:
    fd = open(path + '/' + datadir_filename, 'w')
    fd.write('Number of minimum datadir : {} in class {}\n Number of average datadir : {} \n '
             'Number of total datadir : {} \n Number of maximum datadir : {} in class {}\n'.format(

        num_datadir_min, class_datadir_min, total_data_dir/10,  total_data_dir, num_datadir_max, class_datadir_max))
    fd.close()
except:
    print('error during writing data dir maximum ')

try:
    fm = open(path + '/' + timedataset_filename, 'w')
    fm.write('Number of average seconds : {} \n Number of maximum seconds : {} in class {}\n '
             'Number of minimum seconds : {} in class {}'.format(totaltime/10, timemax, classtimemax, timemin, classtimemin))
    fm.close()
except:
    print('error during writing class length')

try:
    fm = open(path + '/' + video_filename, 'w')
    fm.write('Total {} Number of average seconds : {} \n Number of maximum seconds : {} in video {} in class {}\n '
             'Number of minimum seconds : {} in video {} in class {}'.format(totalvideotime,
              totalvideotime/total_data_dir, timevideomax, videotimemax, classvideotimemax,
              timevideomin, videotimemin, classvideotimemin))
    fm.close()
except:
    print('error during writing minimum class ')
