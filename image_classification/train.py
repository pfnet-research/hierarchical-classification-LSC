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
import network
from importlib import import_module
import sys, os
import numpy as np

import separate
import cifar


class Dataset(object):
    def __init__(self, instances, labels):
        if len(instances) != len(labels):
            raise ValueError('the lengths of instances and labels are different')
        instances, labels = instances[np.argsort(labels)], np.sort(labels)
        label_type = np.unique(labels)
        partition = [np.searchsorted(labels, k, side='left') for k in label_type]
        partition.append(np.size(instances, axis=0))

        self._instances = instances
        self._labels = labels
        self._length = len(instances)
        self._partition = partition

    def __getitem__(self, index):
        if isinstance(index, slice):
            instances = self._instances[index]
            labels = self._labels[index]

            length = len(instances)

            # バッチ内の各ラベルからランダムに要素を取り出す。
            sampled_instances = [random.choice(self._instances[self._partition[label]:self._partition[label + 1]])
                                 for label in labels]
            return [(instances[i], labels[i], sampled_instances[i]) for i in range(length)]
        else:
            instance = self._instances[index]
            label = self._labels[index]
            sampled_instance = random.choice(self._instances[self._partition[label]:self._partition[label + 1]])
            return instance, label, sampled_instance

    def __len__(self):
        return self._length


class Updater(chainer.training.StandardUpdater):
    def __init__(self, model, data, iter, optimizer, lam=0.5, mu=10.0, device=-1):
        self.model = model
        self.data = data
        self.lam = lam
        self.mu = mu
        super(Updater, self).__init__(iter, optimizer, device=device)

    def update_core(self):
        self.model.cleargrads()

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        optimizer = self._optimizers['main']

        # tuple of instance array and label array
        instances, labels, sampled_instances = self.converter(batch, self.device)
        y = F.softmax(self.model(instances, unchain=True))

        H_Y = self.entropy((F.sum(y, axis=0) / batchsize), axis=0)
        H_YX = F.sum(self.entropy(y, axis=1), axis=0) / batchsize
        loss_mut_info = - self.lam * (self.mu * H_Y - H_YX)

        # sampled instancesがリストになっているが、これがnumpy arrayになっているハズ
        sampled_y = F.softmax(self.model(sampled_instances))
        sampled_y.unchain_backward()

        loss_cc = self.loss_cross_entropy(y, sampled_y) / batchsize

        loss = loss_cc + loss_mut_info
        loss.backward()

        optimizer.update()

        chainer.reporter.report({'main/loss': loss})
        chainer.reporter.report({'main/loss_cc': loss_cc})
        chainer.reporter.report({'main/loss_mut_info': loss_mut_info})
        chainer.reporter.report({'main/H_Y': H_Y})
        chainer.reporter.report({'main/H_YX': H_YX})

    @staticmethod
    def entropy(x, axis=0):
        return - F.sum(x * F.log(x + 1e-8), axis=axis)

    @staticmethod
    def loss_class_clustering(p, q):
        return F.sum(p * F.log((p + 1e-8) / (q + 1e-8)))

    @staticmethod
    def loss_cross_entropy(p, q):
        index = F.argmax(q, axis=1).data
        u = F.select_item(p, index)
        return -F.sum(F.log(u + 1e-8))


def data_generate():
    instance, labels = np.empty((400, 2), np.float32), np.empty(400, np.int32)
    var = 1.0

    instance[0:100] = np.random.randn(100, 2) * var + np.array([1, 1])
    instance[100:200] = np.random.randn(100, 2) * var + np.array([1, -1])
    instance[200:300] = np.random.randn(100, 2) * var + np.array([-1, 1])
    instance[300:400] = np.random.randn(100, 2) * var + np.array([-1, -1])

    labels[0:100] = np.zeros(100, np.int32)
    labels[100:200] = np.zeros(100, np.int32) + 1
    labels[200:300] = np.zeros(100, np.int32) + 2
    labels[300:400] = np.zeros(100, np.int32) + 3

    return Dataset(instance, labels)


def load_data(data_type='toy', ndim=1):
    mean = np.array([125.3069, 122.95015, 113.866])
    std = np.array([62.99325, 62.088604, 66.70501])
    if data_type == 'toy':
        return data_generate(), data_generate()
    elif data_type == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.get_mnist(ndim=ndim)
        return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
    elif data_type == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar.get_cifar10()
        return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
    elif data_type == 'cifar100':
        (train_images, train_labels), (test_images, test_labels) = cifar.get_cifar100(scale=255.0)
        train_images -= mean[:, None, None]
        test_images -= mean[:, None, None]
        train_images /= std[:, None, None]
        test_images /= std[:, None, None]
        return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
    else:
        raise ValueError


def check_cluster(model, train, num_classes, num_cluster, batchsize=128, device=-1):
    i, N = 0, len(train)
    cc = None
    ss = None

    while i <= N:
        # concat_examplesは(instances, labels)を返す。
        xx = F.softmax(model(chainer.dataset.convert.concat_examples(train[i:i+batchsize], device=device)[0])).data
        if device >= 0:
            xx = cuda.to_cpu(xx)

        if cc is None:
            cc = np.argmax(xx, axis=1)
        else:
            cc = np.append(cc, np.argmax(xx, axis=1))

        if ss is None:
            ss = np.sum(xx, axis=0)
        else:
            ss = ss + np.sum(xx, axis=0)
        i += batchsize

    ss /= N
    partition = train._partition
    cluster = [tuple(np.sum(cc[partition[k]:partition[k + 1]] == c)
                     for c in range(num_cluster)) for k in range(num_classes)]
    return cluster, ss


class MyNpzDeserializer(serializers.NpzDeserializer):
    def load(self, obj, not_load_list=None):
        obj.serialize(self, not_load_list)


def load_npz(file, obj, path='', strict=True, not_load_list=None):
    with np.load(file) as f:
        d = MyNpzDeserializer(f, path=path, strict=strict)
        d.load(obj, not_load_list)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--data_type', '-d', type=str, default='cifar100')
    parser.add_argument('--model_type', '-m', type=str, default='Resnet50')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--cluster', '-c', type=int, default=2)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0005)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--mu', '-mu', type=float, default=20.0)
    args = parser.parse_args()

    model_file = 'models/ResNet.py'
    model_name = 'ResNet50'
    model_path = "/home/user/.chainer/pretrained/cifar10/model_500.npz"

    gpu = args.gpu
    data_type = args.data_type
    model_type = args.model_type
    num_cluster = args.cluster

    ndim = 1
    if data_type == 'toy':
        model = network.LinearModel(2, 2)
        num_classes = 2
    elif data_type == 'mnist':
        num_classes = 10
        if model_type == 'linear':
            model = network.LinearModel(784, num_cluster)
        elif model_type == 'DNN':
            model = network.MLP(1000, num_cluster)
        elif model_type == 'CNN':
            ndim = 3
            model = network.CNN(num_cluster)
        else:
            raise ValueError
    elif data_type == 'cifar100':
        num_classes = 100
        if model_type == 'Resnet50':
            model = network.ResNet50(args.cluster)
            load_npz(model_path, model, not_load_list=['fc7'])
        else:
            raise ValueError
    else:
        num_classes = 10
        if model_type == 'Resnet50':
            model = network.ResNet50(num_cluster)
        elif model_type == 'Resnet101':
            model = network.ResNet101(num_cluster)
        elif model_type == 'VGG':
            model = network.VGG(num_cluster)
        elif model_type == 'CNN':
            model = network.CNN(num_cluster)
        else:
            raise ValueError

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if args.weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    train, test = load_data(data_type, ndim)

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)

    updater = Updater(model, train, train_iter, optimizer, device=gpu, mu=args.mu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')

    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/loss_cc',
         'main/loss_mut_info', 'main/H_Y', 'main/H_YX', 'elapsed_time']))

    trainer.run()

    res, ss = check_cluster(model, train, num_classes, num_cluster, device=gpu)
    res_sum = tuple(0 for _ in range(num_cluster))
    for i in range(num_classes):
        res_sum = tuple(res_sum[j] + res[i][j] for j in range(num_cluster))

    with open('train.res', 'w') as f:
        print(res, res_sum, ss, file=f)

    res, ss = check_cluster(model, test, num_classes, num_cluster, device=gpu)
    res_sum = tuple(0 for _ in range(num_cluster))
    for i in range(num_classes):
        res_sum = tuple(res_sum[j] + res[i][j] for j in range(num_cluster))

    with open('test.res', 'w') as f:
        print(res, res_sum, ss, file=f)

    cluster_label = separate.det_cluster(model, train, num_classes, batchsize=128, device=gpu)
    print(cluster_label)


if __name__ == '__main__':
    main()