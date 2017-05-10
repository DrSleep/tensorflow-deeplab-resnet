"""Test preparation script for the DeepLab-ResNet network on the test subset
   of PASCAL VOC dataset.

This script tests the model on 1456 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from voc_colour_map import voc_colour_map
from PIL import Image as PILImage

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label

n_classes = 21

SAVE_DIR = './output'
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/test.txt'
NUM_STEPS = 1456 # Number of images in the validation set.
RESTORE_FROM = './deeplab_resnet.ckpt'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted masks.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            coord)
        image, label = reader.image, reader.label

    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch075 = tf.image.resize_images(image_batch, tf.pack([tf.to_int32(tf.mul(h_orig, 0.75)), tf.to_int32(tf.mul(w_orig, 0.75))]))
    image_batch05 = tf.image.resize_images(image_batch, tf.pack([tf.to_int32(tf.mul(h_orig, 0.5)), tf.to_int32(tf.mul(w_orig, 0.5))]))
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, is_training=False)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    raw_output05 = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    
    raw_output = tf.reduce_max(tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
    pred = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
   
    # Get the color palette
    palette = voc_colour_map()


    f = open(args.data_list, 'r')
    image_names = []
    for line in f:
        image_names.append(line)

    # Iterate over training steps.
    for step in range(args.num_steps):
        preds = sess.run(pred)
        preds = np.argmax(preds, axis=3).squeeze().astype(np.uint8)
        im = PILImage.fromarray(preds)
        im.putpalette(palette)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        mask_name = image_names[step].strip("\n").rsplit('/', 1)[1].replace('jpg', 'png')
        im.save(args.save_dir + "/" + mask_name)

    print('The segmentation masks have been saved to {}'.format(args.save_dir))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
