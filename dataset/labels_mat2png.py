#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Converts the .mat segmentation labels in the Augmented Pascal VOC
dataset to color-coded .png images.

Download the dataset from:

http://home.bharathh.info/pubs/codes/SBD/download.html
'''

import argparse
import glob
from os import path
import scipy.io
import PIL
import numpy as np

PASCAL_PALETTE = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (128, 128, 0),
    4: (0, 0, 128),
    5: (128, 0, 128),
    6: (0, 128, 128),
    7: (128, 128, 128),
    8: (64, 0, 0),
    9: (192, 0, 0),
    10: (64, 128, 0),
    11: (192, 128, 0),
    12: (64, 0, 128),
    13: (192, 0, 128),
    14: (64, 128, 128),
    15: (192, 128, 128),
    16: (0, 64, 0),
    17: (128, 64, 0),
    18: (0, 192, 0),
    19: (128, 192, 0),
    20: (0, 64, 128),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, help='Input folder',
                        required=True)
    parser.add_argument('--out-dir', type=str, help='Output folder',
                        required=True)
    args = parser.parse_args()

    from IPython import embed
    files = sorted(glob.glob(path.join(args.in_dir, '*.mat')))

    assert len(files), 'no matlab files found in the input folder'

    # BOUNDARIES_IDX = 0
    SEGMENTATION_IDX = 1
    # CATEGORIES_PRESENT_IDX  = 2

    for fname in files:
        mat = scipy.io.loadmat(fname, mat_dtype=True)
        seg_data = mat['GTcls'][0][0][SEGMENTATION_IDX]
        img_data = np.zeros(seg_data.shape + (3, ), dtype=np.uint8)

        for i in range(img_data.shape[0]):
            for j in range(img_data.shape[1]):
                img_data[i, j, :] = PASCAL_PALETTE[seg_data[i, j]]

        img = PIL.Image.fromarray(img_data)
        img_name = str.replace(path.basename(fname), '.mat', '.png')
        img.save(path.join(args.out_dir, img_name), 'png')


if __name__ == '__main__':
    main()
