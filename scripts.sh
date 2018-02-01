#!/bin/bash

# PATH=/home/garbade/anaconda2/bin:$PATH

python train_msc.py --batch-size 2 \
	--data-dir /home/garbade/datasets/nyu_depth_v2/ \
	--data-list /home/garbade/datasets/nyu_depth_v2/filelists/train_drsleep.txt \
	--snapshot-dir /home/garbade/models_tf/08_nyu_depth_v2/01_02_firstTry_nyu11 \
	--grad-update-every 10 \
	--ignore-label 0 \
	--not-restore-last \
	--num-classes 11 \
	--num-steps 20001 \
	--random-mirror \
	--random-scale \
	--restore-from /home/garbade/models_tf/dr_sleep_models/tf_v0.12/deeplab_resnet.ckpt \
	--save-num-images 1 \
	--learning-rate 1.0e-4
	
	
	

python evaluate_msc.py --data-dir /home/garbade/datasets/nyu_depth_v2/ \
	--data-list /home/garbade/datasets/nyu_depth_v2/filelists/test_drsleep.txt \
	--ignore-label 0 \
	--num-classes 11 \
	--num-steps 1 \
	--restore-from /home/garbade/models_tf/08_nyu_depth_v2/01_02_firstTry_nyu11/model.ckpt-10000 \
	--save-dir /home/garbade/models_tf/08_nyu_depth_v2/01_02_firstTry_nyu11/res \
	--save-dir-ind /home/garbade/models_tf/08_nyu_depth_v2/01_02_firstTry_nyu11/res_ind