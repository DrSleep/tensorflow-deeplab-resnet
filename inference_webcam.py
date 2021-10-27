"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label
import cv2


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference(webcam).")

    parser.add_argument("--model_weights", type=str,
                        help="Path to the file with model weights.", default='./deeplab_resnet.ckpt')

    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")

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

    cv2.namedWindow("preview")
    cv2.namedWindow("legend")
    cv2.namedWindow("SemanticSegmentation")
    leg = cv2.imread('./images/colour_scheme.png')
    cv2.imshow('legend', leg)

    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    # TODO : get resolution of webcam using opencv
    img_input = tf.placeholder(dtype=tf.uint8, shape=[480,640,3])
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img_input)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)
    
    # Perform inference.


    while True:
        preds = sess.run(pred,feed_dict={img_input:frame})

        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])

        open_cv_image = np.array(im)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        cv2.imshow("SemanticSegmentation", open_cv_image)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
if __name__ == '__main__':
    main()
