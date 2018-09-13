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
from chainer.datasets import TransformDataset
from chainercv import transforms
from functools import partial
import cv2 as cv
import six

import ova_network
from importlib import import_module
import sys, os
import numpy as np
import hierarchy_network as h_net
from scipy.sparse import csr_matrix

import cifar
import mnist

import doc_preprocess
import separate
import dataset
import updater
import accuracy


USE_OPENCV = True


def cv_rotate(img, angle):
    if USE_OPENCV:
        img = img.transpose(1, 2, 0) / 255.
        center = (img.shape[0] // 2, img.shape[1] // 2)
        r = cv.getRotationMatrix2D(center, angle, 1.0)
        img = cv.warpAffine(img, r, img.shape[:2])
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    else:
        raise NotImplemented
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img.transpose(1, 2, 0) / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    return img


def transform(
        inputs, mean, std, random_angle=15., pca_sigma=25.5, expand_ratio=1.2,
        crop_size=(28, 28), train=True):
    img, label = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)

    # Color augmentation
    if train and pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)

    """
    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]
    """

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))

    return img, label


def general_transform(inputs, sparse):
    instance, label = inputs

    if sparse:
        instance = np.array(csr_matrix.todense(instance))

    return instance, label


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


class Dataset(object):
    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[1])
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, slice):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in six.moves.range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length


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
    parser.add_argument('--unit', '-u', type=int, default=300)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--optimizer', '-op', type=str, default='Adam')
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
    parser.add_argument('--train_file', '-train_f', type=str, default='dataset/LSHTC1/LSHTC1_selected03.train')
    parser.add_argument('--test_file', '-test_f', type=str, default='dataset/LSHTC1/LSHTC1_selected03.test')
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
    unit = args.unit
    f_train = args.train_file
    f_test = args.test_file

    if data_type == 'toy':
        num_classes = 4

        train = chainer.datasets.TupleDataset(data_generate())
        test = chainer.datasets.TupleDataset(data_generate())

        train_transform = partial(general_transform, sparse=False)
        test_transform = partial(general_transform, sparse=False)

        model = ova_network.LinearModel(2, num_classes)
    elif data_type == 'mnist':
        num_classes = 10
        train_transform = partial(general_transform, sparse=False)
        test_transform = partial(general_transform, sparse=False)
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
        train = chainer.datasets.tuple_dataset.TupleDataset(train_images, train_labels)
        test = chainer.datasets.tuple_dataset.TupleDataset(test_images, test_labels)

        train_transform = partial(
            transform, mean=0.0, std=1.0, train=True)
        test_transform = partial(transform, mean=0.0, std=1.0, train=False)

        if model_type == 'Resnet50':
            model = ova_network.ResNet50(num_classes)
            load_npz(model_path, model, not_load_list=['fc7'])
        else:
            raise ValueError
    elif data_type == 'LSHTC1' or data_type == 'Dmoz':
        train, test, num_classes = doc_preprocess.load_data(f_train, f_test)
        train = Dataset(*train)
        test = Dataset(*test)

        train_transform = partial(general_transform, sparse=True)
        test_transform = partial(general_transform, sparse=True)
        if model_type == 'DocModel':
            model = ova_network.DocModel(n_in=328282, n_mid=unit, n_out=num_classes)
        elif model_type == 'DocModel2':
            model = ova_network.DocModel(n_in=328282, n_mid=unit, n_out=num_classes)
        elif model_type == 'linear':
            model = ova_network.LinearModel(n_in=92586, n_out=num_classes)
        else:
            raise ValueError
    else:
        train_transform = partial(
            transform, mean=0.0, std=1.0, train=True)
        test_transform = partial(transform, mean=0.0, std=1.0, train=False)
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

    print('num_classes: ' + str(num_classes))

    train = TransformDataset(train, train_transform)
    test = TransformDataset(test, test_transform)

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
    test_iter = chainer.iterators.SerialIterator(test, batch_size=1, repeat=False, shuffle=False)

    train_updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)

    trainer = training.Trainer(train_updater, (args.epoch, 'epoch'), out=args.out)

    # trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    # trainer.extend(
        # extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        # trigger=(20, 'epoch'))
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
