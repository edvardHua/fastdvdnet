#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 10:39
# @Author  : zengzihua
# @File    : test.py

import torch
from models_e2e_edz import TemporalUNet


def test_network():
    from thop import profile

    model_temp = TemporalUNet(num_input_frames=5, each_frame_channel=1)
    model_temp.eval()
    # 假设是 5 帧
    inp_frames = torch.randn(1, 5, 1, 284, 256)

    flops, params = profile(model_temp, (inp_frames,))
    print(flops / 1e9, "G")
    print(params / 1e6, "M")
    pass


if __name__ == "__main__":
    test_network()
