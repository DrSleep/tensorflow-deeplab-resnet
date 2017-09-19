# DeepLab-ResNet-TensorFlow

[![Build Status](https://travis-ci.org/DrSleep/tensorflow-deeplab-resnet.svg?branch=master)](https://travis-ci.org/DrSleep/tensorflow-deeplab-resnet)

This is an (re-)implementation of [DeepLab-ResNet](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

## Frequently Asked Questions

If you encounter some problems and would like to create an issue, please read this first. If the guide below does not cover your question, please use search to see if a similar issue has already been solved before. Finally, if you are unable to find an answer, please fill in the issue with details of your problem provided.

#### Which `python` version should I use?

All the experiments are been done using `python2.7`. `python3` will likely require some minor modifications.

#### After training, I have multiple files that look like `model.ckpt-xxxx.index`, `model.ckpt-xxxx.dataxxxx` and `model.ckpt-xxxx.meta`. Which one of them should I use to restore the model for inference?

Instead of providing a path to one of those files, you must provide just `model.ckpt-xxxx`. It will fetch other files.

#### My model is not learning anything. What should I do?

First, check that your images are being read correctly. The setup implies that segmentation masks are saved without a colour map, i.e., each pixel contains a class index, not an RGB value.
Second, tune your hyperparameters. As there are no general strategies that work for each case, the design of this procedure is up to you.

#### I want to use my own dataset. What should I do?

Please refer to this [topic](https://github.com/DrSleep/tensorflow-deeplab-resnet#using-your-dataset).

## Updates

**29 Jan, 2017**:
* Fixed the implementation of the batch normalisation layer: it now supports both the training and inference steps. If the flag `--is-training` is provided, the running means and variances will be updated; otherwise, they will be kept intact. The `.ckpt` files have been updated accordingly - to download please refer to the new link provided below.
* Image summaries during the training process can now be seen using TensorBoard.
* Fixed the evaluation procedure: the 'void' label (<code>255</code>) is now correctly ignored. As a result, the performance score on the validation set has increased to <code>80.1%</code>.

**11 Feb, 2017**:
* The training script `train.py` has been re-written following the original optimisation setup: SGD with momentum, weight decay, learning rate with polynomial decay, different learning rates for different layers, ignoring the 'void' label (<code>255</code>).
* The training script with multi-scale inputs `train_msc.py` has been added: the input is resized to <code>0.5</code> and <code>0.75</code> of the original resolution, and <code>4</code> losses are aggregated: loss on the original resolution, on the <code>0.75</code> resolution, on the <code>0.5</code> resolution, and loss on the all fused outputs.
* Evaluation of a single-scale converted pre-trained model on the PASCAL VOC validation dataset (using ['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)) leads to <code>86.9%</code> mIoU (as trainval was likely to be used for final training). This is confirmed by [the official PASCAL VOC server](http://host.robots.ox.ac.uk/anonymous/FIQPRH.html). The score on the test dataset is [<code>75.8%</code>](http://host.robots.ox.ac.uk/anonymous/EPBIGU.html).

**22 Feb, 2017**:
* The training script with multi-scale inputs `train_msc.py` now supports gradients accumulation: the relevant parameter `--grad-update-every` effectively mimics the behaviour of `iter_size` of Caffe. This allows to use batches of bigger sizes with less GPU memory being consumed. (Thanks to @arslan-chaudhry for this contribution!)
* The random mirror and random crop options have been added. (Again big thanks to @arslan-chaudhry !)

**23 Apr, 2017**:
* TensorFlow 1.1.0 is now supported.
* Three new flags `--num-classes`, `--ignore-label` and `--not-restore-last` are added to ease the usability of the scripts on new datasets. Check out [these instructions](https://github.com/DrSleep/tensorflow-deeplab-resnet#using-your-dataset) on how to set up the training process on your dataset.

## Model Description

The DeepLab-ResNet is built on a fully convolutional variant of [ResNet-101](https://github.com/KaimingHe/deep-residual-networks) with [atrous (dilated) convolutions](https://github.com/fyu/dilation), atrous spatial pyramid pooling, and multi-scale inputs (not implemented here).

The model is trained on a mini-batch of images and corresponding ground truth masks with the softmax classifier at the top. During training, the masks are downsampled to match the size of the output from the network; during inference, to acquire the output of the same size as the input, bilinear upsampling is applied. The final segmentation mask is computed using argmax over the logits.
Optionally, a fully-connected probabilistic graphical model, namely, CRF, can be applied to refine the final predictions.
On the test set of PASCAL VOC, the model achieves <code>79.7%</code> of mean intersection-over-union.

For more details on the underlying model please refer to the following paper:


    @article{CP2016Deeplab,
      title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs},
      author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
      journal={arXiv:1606.00915},
      year={2016}
    }



## Requirements

TensorFlow needs to be installed before running the scripts.
TensorFlow v1.1.0 is supported; for TensorFlow v0.12 please refer to this [branch](https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/tf-0.12); for TensorFlow v0.11 please refer to this [branch](https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/tf-0.11). Note that those branches may not have the same functional as the current master.

To install the required python packages (except TensorFlow), run
```bash
pip install -r requirements.txt
```
or for a local installation
```bash
pip install -user -r requirements.txt
```

## Caffe to TensorFlow conversion

To imitate the structure of the model, we have used `.caffemodel` files provided by the [authors](http://liangchiehchen.com/projects/DeepLabv2_resnet.html). The conversion has been performed using [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow) with an additional configuration for atrous convolution and batch normalisation (since the batch normalisation provided by Caffe-tensorflow only supports inference).
There is no need to perform the conversion yourself as you can download the already converted models - `deeplab_resnet.ckpt` (pre-trained) and `deeplab_resnet_init.ckpt` (the last layers are randomly initialised) - [here](https://drive.google.com/open?id=0B_rootXHuswsZ0E4Mjh1ZU5xZVU).

Nevertheless, it is easy to perform the conversion manually, given that the appropriate `.caffemodel` file has been downloaded, and [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow) dependencies have been installed. The Caffe model definition is provided in `misc/deploy.prototxt`.
To extract weights from `.caffemodel`, run the following:
```bash
python convert.py /path/to/deploy/prototxt --caffemodel /path/to/caffemodel --data-output-path /where/to/save/numpy/weights
```
As a result of running the command above, the model weights will be stored in `/where/to/save/numpy/weights`. To convert them to the native TensorFlow format (`.ckpt`), simply execute:
```bash
python npy2ckpt.py /where/to/save/numpy/weights --save-dir=/where/to/save/ckpt/weights
```

## Dataset and Training

To train the network, one can use the augmented PASCAL VOC 2012 dataset with <code>10582</code> images for training and <code>1449</code> images for validation.

The training script allows to monitor the progress in the optimisation process using TensorBoard's image summary. Besides that, one can also exploit random scaling and mirroring of the inputs during training as a means for data augmentation. For example, to train the model from scratch with random scale and mirroring turned on, simply run:
```bash
python train.py --random-mirror --random-scale
```

<img src="images/summary.png"></img>

To see the documentation on each of the training settings run the following:

```bash
python train.py --help
```

An additional script, `fine_tune.py`, demonstrates how to train only the last layers of the network. The script `train_msc.py` with multi-scale inputs fully resembles the training setup of the original model.


## Evaluation

The single-scale model shows <code>86.9%</code> mIoU on the Pascal VOC 2012 validation dataset (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)). No post-processing step with CRF is applied.

The following command provides the description of each of the evaluation settings:
```bash
python evaluate.py --help
```

## Inference

To perform inference over your own images, use the following command:
```bash
python inference.py /path/to/your/image /path/to/ckpt/file
```
This will run the forward pass and save the resulted mask with this colour map:
<img src="images/colour_scheme.png" height="75"></img>
<img src="images/mask.png"></img>

## Using your dataset

In order to apply the same scripts using your own dataset, you would need to follow the next steps:

0. Make sure that your segmentation masks are in the same format as the ones in the DeepLab setup (i.e., without a colour map). This means that if your segmentation masks are RGB images, you would need to convert each 3-D RGB vector into a 1-D label. For example, take a look [here](https://gist.github.com/DrSleep/4bce37254c5900545e6b65f6a0858b9c);
1. Create a file with instances of your dataset in the same format as in files [here](https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/master/dataset);
2. Change the flags `data-dir` and `data-list` accordingly in thehttps://gist.github.com/DrSleep/4bce37254c5900545e6b65f6a0858b9c); script file that you will be using (e.g., `python train.py --data-dir /my/data/dir --data-list /my/data/list`);
3. Change the `IMG_MEAN` vector accordingly in the script file that you will be using;
4. For visualisation purposes, you will also need to change the colour map [here](https://github.com/DrSleep/tensorflow-deeplab-resnet/blob/master/deeplab_resnet/utils.py);
5. Change the flags `num-classes` and `ignore-label` accordingly in the script that you will be using (e.g., `python train.py --ignore-label 255 --num-classes 21`).
6. If restoring weights from the `PASCAL` models for your dataset with a different number of classes, you will also need to pass the `--not-restore-last` flag, which will prevent the last layers of size <code>21</code> from being restored.


## New Features

###  Training and fine-tuning

These features are included in the [training](https://github.com/naldeborgh7575/tensorflow-deeplab-resnet/blob/master/train.py) and [fine-tuning](https://github.com/naldeborgh7575/tensorflow-deeplab-resnet/blob/master/fine_tune.py) scripts.

1. <b>--class-weights </b> [float ...]: This argument allows training with an asymmetric loss function (helpful with class imbalance). This flag is followed by a float for each class, representing the relative weight that the logit will be multiplied by. Order of classes is the same as `label_colours` in [deeplab_resnet/utils.py](https://github.com/naldeborgh7575/tensorflow-deeplab-resnet/blob/master/deeplab_resnet/utils.py#L7).

  ```bash
  python train.py --class-weights 1. 2.7
  ```

2.  <b>--val-list </b> [str]: Location of list of validation data. This instructs the script to report the [Jaccard loss](https://en.wikipedia.org/wiki/Jaccard_index) of validation data each time a checkpoint is created. Note that this is a streaming value and is simply  updated as each checkpoint is created. *This flag should be accompanied by a --val-size flag, indicating how many validation chips there are (defaults to 500)*

  ```bash
  python train.py --val-list ../footprints/splits/validation_tf.txt --val-size 750
  ```

3. <b>--val-size</b> [int]: Number of images to validate on each time validation is performed.

### Evaluation and prediction

The following update can be found in the [evaluation](https://github.com/naldeborgh7575/tensorflow-deeplab-resnet/blob/master/evaluate.py) and [inference](https://github.com/naldeborgh7575/tensorflow-deeplab-resnet/blob/master/inference.py) scripts.

1. <b>--crf</b>: Use a [CRF](https://arxiv.org/abs/1210.5644) to 'clean up' the output of the network.

2. <b>--augment</b>: Use prediction-time augmentation. Predictions will be calculated at 0, 90, 180 and 270 degrees, and averaged for a final prediction.

### Sample run

1. Fine tune final layers.
  ```bash
  python2 fine_tune.py --not-restore-last --batch-size 2 --num-steps 3500 \
    --data-dir ../footprints/ --data-list ../footprints/splits/train_tf.txt \
    --val-list ../footprints/splits/validation_tf.txt --val-size 750 \
    --class-weights 1.  91.82. --ignore-label 128 --input-size 256,256 \
    --learning-rate 1e-4 --num-classes 2 --restore-from ./deeplab_resnet.ckpt
  ```

2. Retrain entire net

  ```bash
  python2 train.py --batch-size 10  --num-steps 15000 --data-dir ../footprints/ \
    --data-list ../footprints/splits/train_tf.txt --class-weights 1.  91.84 \
    --val-list ../footprints/splits/validation_tf.txt --val-size 750 \
    --ignore-label 128 --input-size 256,256 --learning-rate 2.5e-4 --num-classes 2 \
    --restore-from ./snapshots_finetune/model_0.474_viou.ckpt-4500
  ```

## Other implementations
* [DeepLab-LargeFOV in TensorFlow](https://github.com/DrSleep/tensorflow-deeplab-lfov)
