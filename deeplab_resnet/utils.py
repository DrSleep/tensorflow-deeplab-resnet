from PIL import Image
import numpy as np
import tensorflow as tf

n_classes = 21
# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    
def decode_labels(mask):
    """Decode batch of segmentation masks.
    
    Args:
      label_batch: result of inference after taking argmax.
    
    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < n_classes:
                pixels[k_,j_] = label_colours[k]
    return np.array(img)

def prepare_label(input_batch, new_size):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch

