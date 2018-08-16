import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
import random
import argparse
import chainer
from chainer import serializers

from importlib import import_module
import os


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--data_type', '-d', type=str, default='mnist')
    parser.add_argument('--model_type', '-m', type=str, default='DNN')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--cluster', '-c', type=int, default=2)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0005)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--mu', '-mu', type=float, default=6.0)
    args = parser.parse_args()

    model_file = 'models/ResNet.py'
    model_name = 'ResNet50'
    model_path = ""  # TODO: write model_path

    # Load model
    ext = os.path.splitext(model_file)[1]
    mod_path = '.'.join(os.path.split(model_file)).replace(ext, '')
    mod = import_module(mod_path)
    model = getattr(mod, model_name)(args.cluster)

    model = L.ResNet50Layers()
    serializers.load_npz(model_path, model)



if __name__ == '__main__':
    main()