"""Training script with multi-scale inputs for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label

n_classes = 21

BATCH_SIZE = 1
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
GRAD_UPDATE_EVERY = 10
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--grad-update-every", type=int, default=GRAD_UPDATE_EVERY,
                        help="Number of steps after which gradient update is applied.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to update the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
        image_batch075 = tf.image.resize_images(image_batch, [int(h * 0.75), int(w * 0.75)])
        image_batch05 = tf.image.resize_images(image_batch, [int(h * 0.5), int(w * 0.5)])
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, is_training=args.is_training)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, is_training=args.is_training)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = net075.layers['fc1_voc12']
    raw_output05 = net05.layers['fc1_voc12']
    raw_output = tf.reduce_max(tf.stack([raw_output100,
                                         tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                         tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, n_classes])
    raw_prediction100 = tf.reshape(raw_output100, [-1, n_classes])
    raw_prediction075 = tf.reshape(raw_output075, [-1, n_classes])
    raw_prediction05 = tf.reshape(raw_output05, [-1, n_classes])
    
    label_proc = prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]), one_hot=False) # [batch_size, h, w]
    label_proc075 = prepare_label(label_batch, tf.pack(raw_output075.get_shape()[1:3]), one_hot=False)
    label_proc05 = prepare_label(label_batch, tf.pack(raw_output05.get_shape()[1:3]), one_hot=False)
    
    raw_gt = tf.reshape(label_proc, [-1,])
    raw_gt075 = tf.reshape(label_proc075, [-1,])
    raw_gt05 = tf.reshape(label_proc05, [-1,])
    
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, n_classes - 1)), 1)
    indices075 = tf.squeeze(tf.where(tf.less_equal(raw_gt075, n_classes - 1)), 1)
    indices05 = tf.squeeze(tf.where(tf.less_equal(raw_gt05, n_classes - 1)), 1)
    
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
    gt05 = tf.cast(tf.gather(raw_gt05, indices05), tf.int32)
    
    prediction = tf.gather(raw_prediction, indices)
    prediction100 = tf.gather(raw_prediction100, indices)
    prediction075 = tf.gather(raw_prediction075, indices075)
    prediction05 = tf.gather(raw_prediction05, indices05)
                                                  
                                                  
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    loss100 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction100, labels=gt)
    loss075 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction075, labels=gt075)
    loss05 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction05, labels=gt05)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss100) + tf.reduce_mean(loss075) + tf.reduce_mean(loss05) + tf.add_n(l2_losses)
    
    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(2, [images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    # Define a variable to accumulate gradients.
    accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
                               trainable=False) for v in conv_trainable + fc_w_trainable + fc_b_trainable]

    # Define an operation to clear the accumulated gradients for next batch.
    zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

    # Compute gradients.
    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
   
    # Accumulate and normalise the gradients.
    accum_grads_op = [accum_grads[i].assign_add(grad / args.grad_update_every) for i, grad in
                       enumerate(grads)]

    grads_conv = accum_grads[:len(conv_trainable)]
    grads_fc_w = accum_grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = accum_grads[(len(conv_trainable) + len(fc_w_trainable)):]

    # Apply the gradients.
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=10)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        loss_value = 0

        # Clear the accumulated gradients.
        sess.run(zero_op, feed_dict=feed_dict)
       
        # Accumulate gradients.
        for i in range(args.grad_update_every):
            _, l_val = sess.run([accum_grads_op, reduced_loss], feed_dict=feed_dict)
            loss_value += l_val

        # Normalise the loss.
        loss_value /= args.grad_update_every

        # Apply gradients.
        if step % args.save_pred_every == 0:
            images, labels, summary, _ = sess.run([image_batch, label_batch, total_summary, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step)
        else:
            sess.run(train_op, feed_dict=feed_dict)

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
