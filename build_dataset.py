# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
from PIL import Image
import PIL
import scipy
import scipy.ndimage
import imageio
import pickle
import upscale


# LIBRARY_PATH = 'library'
# SAVE_PATH = 'art_face_save.pth'

output_size = 256
LIBRARY_PATH = 'library'
# LIBRARY_PATH = 'images'
# SAVE_PATH = 'save_cio.pth'
SAVE_PATH = 'art_face_save.pth'
DST_PATH = 'k-y-h-u_%sx%s' % (output_size, output_size)
use_1024 = True
max_upscale = 4
FILTER = 'k-y-h-u'

up = upscale.Waifu2x()

# lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
# 69	111	108	111	88	136	72	152	105	152
# 44	51	83	51	63	76	47	92	80	92


def align(img, parts, dst_dir='realign1024x1024', output_size=1024, item_idx=0, enable_padding=True):
    transform_size = 4 * output_size
    # Parse landmarks.
    lm = np.array(parts)

    # Calculate auxiliary vectors.
    eye_left = lm[0]
    eye_right = lm[1]
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm[2]
    mouth_right = lm[3]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)

    if use_1024:
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    else:
        x *= (np.hypot(*eye_to_eye) * 1.6410 + np.hypot(*eye_to_mouth) * 1.560) / 2.0

    y = np.flipud(x) * [-1, 1]

    if use_1024:
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    else:
        c = eye_avg + eye_to_mouth * 0.317
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

    qsize = np.hypot(*x) * 2

    img = Image.fromarray(img)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    if output_size / ((quad[0, 0] - quad[1, 0]) ** 2 + (quad[0, 1] - quad[1, 1]) ** 2) ** 0.5 > max_upscale:
        return False

    for i in range(3):
        if img.size[0] < output_size or img.size[1] < output_size:
            img = up.process(img)
            quad *= 2.0

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BICUBIC)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    dst_subdir = dst_dir
    os.makedirs(dst_subdir, exist_ok=True)
    img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))
    return True


item_idx = 0

pickles = [SAVE_PATH]

paths = []
for dirName, subdirList, fileList in os.walk(LIBRARY_PATH):
    paths += [os.path.relpath(os.path.join(dirName, x), LIBRARY_PATH) for x in fileList if x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')]


for p in pickles:
    with open(p, 'rb') as f:
        annotation = pickle.load(f)

    for filename in paths:
        if not os.path.isfile(os.path.join(LIBRARY_PATH, filename)):
            continue
        if FILTER is not None and FILTER not in filename:
            continue
        if filename in annotation:
            img = imageio.imread(os.path.join(LIBRARY_PATH, filename))
            try:
                if len(img.shape) == 3:
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    elif img.shape[2] == 1:
                        img = np.concatinate([img, img, img], axis=2)
                elif len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=2)

            except IndexError as e:
                print('error %s' % filename)
                raise e

            l = annotation[filename]
            n = 4

            dets = x = [l[i:i + n] for i in range(0, len(l), n)]

            print("Number of faces detected: {}".format(len(dets)))

            try:
                for i, d in enumerate(dets):
                    if len(d) != 4:
                        print('Error with annotation: %s' % filename)
                        continue
                    res = align(img, d, dst_dir=DST_PATH, output_size=output_size, item_idx=item_idx)

                    if res:
                        item_idx += 1
                    else:
                        print('Skipped annotation in %s' % filename)
            except IndexError as e:
                print("error with %s" % filename)
                raise e
