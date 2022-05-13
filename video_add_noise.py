# -*- coding: utf-8 -*-
# @Time : 2022/4/28 4:47 PM
# @Author : zihua.zeng
# @File : video_add_noise.py

import os
import cv2
import torch
import numpy as np


def video_add_noise(inp_path_vn, out_path, sigma=20):
    noise_sigma = sigma / 255.

    cap = cv2.VideoCapture(inp_path_vn)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fn = os.path.basename(inp_path_vn)

    writer = cv2.VideoWriter(os.path.join(out_path, "%d_%s" % (sigma, fn)),
                             cv2.VideoWriter_fourcc(*"MP4V"),
                             fps,
                             (width, height))

    for _ in range(frame_count):
        _, frame = cap.read()

        if frame is None:
            break
        frame = frame.astype(np.float)
        noise = np.random.normal(0, noise_sigma, (height, width, 3))
        frame /= 255.
        frame += noise
        frame *= 255.
        frame = np.clip(frame, 0, 255)
        writer.write(frame.astype(np.uint8))

    cap.release()
    writer.release()


if __name__ == '__main__':
    # base_path = "data/fastdvdnet/training/mp4/"
    base_path = "/Users/zihua.zeng/Workspace/fastdvdnet/data/fastdvdnet/test_sequences/derf_540p_seqs"
    fns = ["park_joy.mp4", "sunflower.mp4", "tractor.mp4"]
    for vp in fns:
        for sigma in range(10, 40, 10):
            video_add_noise(os.path.join(base_path, vp), "data/noising", sigma)
