#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:/home/vsanguineti/models/research/slim"

python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds"

python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_location_1/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_location_1"

python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_location_2/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_location_2"

python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_location_3/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_location_3"

python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_locations_1_and_2/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_locations_1_and_2"
 
python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_locations_1_and_3/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_locations_1_and_3"

python spectrogramStatistic.py --train_file "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_locations_2_and_3/lists/training.txt" --batch_size 8 --num_channels 257 "/data/vsanguineti/dualcam-net/30_seconds_lists/30_seconds_locations_2_and_3"

