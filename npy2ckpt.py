"""Conversion of the .npy weights into the .ckpt ones.

This script converts the weights of the DeepLab-ResNet model
from the numpy format into the TensorFlow one.
"""

from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel

SAVE_DIR = './'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NPY to CKPT converter.")
    parser.add_argument("npy_path", type=str,
                        help="Path to the .npy file, which contains the weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save the converted .ckpt file.")
    return parser.parse_args()

def save(saver, sess, logdir):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, write_meta_graph=False)
    print('The weights have been converted to {}.'.format(checkpoint_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    # Default image.
    image_batch = tf.constant(0, tf.float32, shape=[1, 321, 321, 3]) 
    # Create network.
    net = DeepLabResNetModel({'data': image_batch})
    var_list = tf.global_variables()
          
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
          init = tf.global_variables_initializer()
          sess.run(init)
          
          # Loading .npy weights.
          net.load(args.npy_path, sess)
          
          # Saver for converting the loaded weights into .ckpt.
          saver = tf.train.Saver(var_list=var_list, write_version=1)
          save(saver, sess, args.save_dir)

if __name__ == '__main__':
    main()
