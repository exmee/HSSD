import os
from skimage.transform import resize as imresize

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.ops.metrics_impl import mean_iou

import logging
from vgg import VGG
from voc_loader import VOCLoader

from boxer import PriorBoxGrid
from config import args, train_dir
from paths import CKPT_ROOT, EVAL_DIR, RESULTS_DIR
from utils import decode_bboxes, batch_iou

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

log = logging.getLogger()
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']

#************************************#
#cow 0.96 (214, 108) (283, 148)
#cow 0.96 (277, 104) (346, 146)
#cow 0.97 (465, 102) (500, 147)
#cow 0.97 (103, 95) (143, 133)
#cow 0.99 (130, 102) (221, 149)
#************************************#


def draw_box():
    img = "/home/ubuntu2/exme/papercode/HSSD/Demo/test/2007_001764.jpg"
    img = Image.open(img)
    cats = [10, 10, 10, 10, 10]
    dets = [[214, 108, 283, 148], [277, 104, 346, 146], [465, 102, 500, 147],
            [103, 95, 143, 133], [130, 102, 221, 149]]
    scores = [0.96, 0.96, 0.97, 0.97, 0.99]
    name = "2007_001764"
    draw(img, dets, cats, scores, name, None, None)

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline=color)

def draw(img, dets, cats, scores, name, gt_bboxes, gt_cats):
    """Visualize objects detected by the network by putting bounding boxes"""
    #colors = np.load('Extra/colors.npy').tolist()
    outpath = "/home/ubuntu2/exme/papercode/HSSD/demodemo/test/output1"
    colors = STANDARD_COLORS
    font = ImageFont.truetype("Extra/FreeSansBold.ttf", 18)

    #h, w = img.shape[:2]
    #image = Image.fromarray((img * 255).astype('uint8'))
    image = img
    dr = ImageDraw.Draw(image)
    #if not args.segment:
    #    image.save(outpath + '/%s.jpg' % name, 'JPEG')

    for i in range(len(cats)):
        cat = cats[i]
        score = scores[i]
        bbox = np.array(dets[i])

        #bbox[[2, 3]] += bbox[[0, 1]]
        # color = 'green' if matched_det[i] else 'red'
        color = colors[cat]
        draw_rectangle(dr, bbox, color, width=5)
        # ********************************************************#
        x, y = bbox[:2]
        dr.text((x, y - 25), VOC_CATS[cat] + ' ' + str(score),
                # dr.text(bbox[:2], self.loader.ids_to_cats[cat] + ' ' + str(score),
                fill=color, font=font)
    image.save(outpath + '/%s.jpg' % name, 'JPEG')
    image.show()
    #image.save(outpath + 'output.jpg')
    #image.save(outpath + '/%s_det_%i.jpg' % (name, int(100 * args.eval_min_conf)), 'JPEG')



draw_box()
