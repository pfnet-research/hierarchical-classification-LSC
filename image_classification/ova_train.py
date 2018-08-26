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
import ova_network
from importlib import import_module
import sys, os
import numpy as np
import hierarchy_network as h_net

import cifar
import mnist

import separate
import dataset
import updater
import accuracy


def data_generate():
    instance, labels = np.empty((400, 2), np.float32), np.empty(400, np.int32)
    var = 0.5

    instance[0:100] = np.random.randn(100, 2) * var + np.array([1, 1])
    instance[100:200] = np.random.randn(100, 2) * var + np.array([1, -1])
    instance[200:300] = np.random.randn(100, 2) * var + np.array([-1, 1])
    instance[300:400] = np.random.randn(100, 2) * var + np.array([-1, -1])

    labels[0:100] = np.zeros(100, np.int32)
    labels[100:200] = np.zeros(100, np.int32) + 1
    labels[200:300] = np.zeros(100, np.int32) + 2
    labels[300:400] = np.zeros(100, np.int32) + 3

    return instance, labels


class MyNpzDeserializer(serializers.NpzDeserializer):
    def load(self, obj, not_load_list=None):
        obj.serialize(self, not_load_list)


def load_npz(file, obj, path='', strict=True, not_load_list=None):
    with np.load(file) as f:
        d = MyNpzDeserializer(f, path=path, strict=strict)
        d.load(obj, not_load_list)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--data_type', '-d', type=str, default='mnist')
    parser.add_argument('--model_type', '-m', type=str, default='linear')
    parser.add_argument('--model_path', '-mp', type=str,
                        default='./models/ResNet50_model_500.npz')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0005)
    parser.add_argument('--epoch', '-e', type=int, default=3)
    parser.add_argument('--out', '-o', type=str, default='results')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--optimizer', '-op', type=str, default='Adam')
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)


    gpu = args.gpu
    data_type = args.data_type
    model_type = args.model_type
    initial_lr = args.initial_lr
    lr_decay_rate = args.lr_decay_rate
    lr_decay_epoch = args.lr_decay_epoch
    weight_decay = args.weight_decay
    model_path = args.model_path
    opt = args.optimizer

    if data_type == 'toy':
        num_classes = 4

        train = chainer.datasets.TupleDataset(data_generate())
        test = chainer.datasets.TupleDataset(data_generate())

        model = ova_network.LinearModel(2, num_classes)
    elif data_type == 'mnist':
        num_classes = 10
        if model_type == 'linear':
            train, test = chainer.datasets.get_mnist(ndim=1)
            model = ova_network.LinearModel(784, num_classes)
        elif model_type == 'DNN':
            train, test = chainer.datasets.get_mnist(ndim=1)
            model = ova_network.MLP(1000, num_classes)
        elif model_type == 'CNN':
            train, test = chainer.datasets.get_mnist(ndim=3)
            model = ova_network.CNN(num_classes)
        else:
            raise ValueError
    elif data_type == 'cifar100':
        num_classes = 100

        (train_images, train_labels), (test_images, test_labels) = cifar.get_cifar100(scale=255.0)
        mean = np.array([125.3069, 122.95015, 113.866])
        std = np.array([62.99325, 62.088604, 66.70501])
        train_images -= mean[:, None, None]
        test_images -= mean[:, None, None]
        train_images /= std[:, None, None]
        test_images /= std[:, None, None]
        train = chainer.datasets.TupleDataset(train_images, train_labels)
        test = chainer.datasets.TupleDataset(test_images, test_labels)

        if model_type == 'Resnet50':
            model = ova_network.ResNet50(num_classes)
            load_npz(model_path, model, not_load_list=['fc7'])
        else:
            raise ValueError
    else:
        num_classes = 10
        train, test = chainer.datasets.get_cifar10(scale=1.0)
        if model_type == 'Resnet50':
            model = ova_network.ResNet50(num_classes)
        elif model_type == 'Resnet101':
            model = ova_network.ResNet101(num_classes)
        elif model_type == 'VGG':
            model = ova_network.VGG(num_classes)
        elif model_type == 'CNN':
            model = ova_network.CNN(num_classes)
        else:
            raise ValueError

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    model = L.Classifier(model)

    if opt == 'Adam':
        optimizer = chainer.optimizers.Adam()
    else:
        optimizer = chainer.optimizers.MomentumSGD(lr=initial_lr)
    optimizer.setup(model)
    if weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=args.batchsize, repeat=False, shuffle=False)

    train_updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)

    trainer = training.Trainer(train_updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/accuracy',
         'validation/main/loss', 'elapsed_time']))

    if opt != 'Adam':
        trainer.extend(extensions.ExponentialShift(
            'lr', lr_decay_rate), trigger=(lr_decay_epoch, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()