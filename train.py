import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from functools import partial
from chainer.backends import cuda
import random
import argparse
import chainer
from chainer import function
from chainer import serializers
from importlib import import_module
import sys, os
import numpy as np
from scipy.sparse import csr_matrix

from network import hierarchy_network as h_net
from network import network, hierarychy_classifier
import clustering
import cifar
import mnist
import doc_preprocess
import dataset

import separate
import accuracy
import classification


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


def load_data(data_type='toy', ndim=1, f_train='', f_test=''):
    if data_type == 'toy':
        train_instances, train_labels = data_generate()
        test_instances, test_labels = data_generate()
        return (train_instances, train_labels), (test_instances, test_labels), 4
    elif data_type == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.get_mnist(ndim=ndim)
        return (train_images, train_labels), (test_images, test_labels), 10
    elif data_type == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar.get_cifar10()
        return (train_images, train_labels), (test_images, test_labels), 10
    elif data_type == 'LSHTC1' or data_type == 'Dmoz':
        (train_instances, train_labels), (test_instances, test_labels), num_classes = \
            doc_preprocess.load_data(f_train, f_test)
        return (train_instances, train_labels), (test_instances, test_labels), num_classes
    elif data_type == 'cifar100':
        mean = np.array([125.3069, 122.95015, 113.866])
        std = np.array([62.99325, 62.088604, 66.70501])
        (train_images, train_labels), (test_images, test_labels) = cifar.get_cifar100(scale=255.0)
        train_images -= mean[:, None, None]
        test_images -= mean[:, None, None]
        train_images /= std[:, None, None]
        test_images /= std[:, None, None]
        return (train_images, train_labels), (test_images, test_labels), 100
    else:
        raise ValueError


def check_cluster(model, train, num_classes, num_cluster, batchsize=128, device=-1, sparse=False):
    with chainer.using_config('train', False):
        i, N = 0, len(train)
        cc = None
        ss = None

        while i <= N:
            train_batch = train[i:i + batchsize]
            if sparse:
                train_batch = np.array(csr_matrix.todense(train_batch))
            # concat_examplesは(instances, labels)を返す。
            xx = F.softmax(model(chainer.dataset.convert.concat_examples(train_batch, device=device)[0])).data
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


def random_assignment(num_clusters, num_classes):
    assignment, count_classes = [], []
    for i in range(num_clusters):
        n_c = num_classes // (num_clusters - i)
        count_classes.append(n_c)
        num_classes -= n_c
        for j in range(n_c):
            assignment.append((i, j))
    return random.sample(assignment, len(assignment)), count_classes


class MyNpzDeserializer(serializers.NpzDeserializer):
    def load(self, obj, not_load_list=None):
        obj.serialize(self, not_load_list)


def load_npz(file, obj, path='', strict=True, not_load_list=None):
    with np.load(file) as f:
        d = MyNpzDeserializer(f, path=path, strict=strict)
        d.load(obj, not_load_list)


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Clustering and Classification')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch for clustering')
    parser.add_argument('--batchsize2', '-b2', type=int, default=64,
                        help='Number of images in each mini-batch for classification')
    parser.add_argument('--data_type', '-d', type=str, default='toy',
                        help='dataset name')
    parser.add_argument('--model_type', '-m', type=str, default='linear',
                        help='model to use')
    parser.add_argument('--model_path', '-mp', type=str, default='',
                        help='pre-trained model if necessary')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='gpu number to use')
    parser.add_argument('--cluster', '-c', type=int, default=2,
                        help='the size of cluster')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0000,
                        help='weight decay for classification')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='unit size for DocModel')
    parser.add_argument('--alpha', '-a', type=float, default=0.001,
                        help='learning rate for clustering')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='the number of epochs for clustering')
    parser.add_argument('--epoch2', '-e2', type=int, default=100,
                        help='the number of epochs for classification')
    parser.add_argument('--mu', '-mu', type=float, default=150.0,
                        help='the hyper-parameter for clustering')
    parser.add_argument('--out', '-o', type=str, default='results',
                        help='output directory for result file')
    parser.add_argument('--train_file', '-train_f', type=str, default='',
                        help='training dataset file')
    parser.add_argument('--test_file', '-test_f', type=str, default='',
                        help='test dataset file')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--resume2', '-r2', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--optimizer', '-op', type=str, default='Adam',
                        help='optimizer for clustering')
    parser.add_argument('--optimizer2', '-op2', type=str, default='Adam',
                        help='optimizer for classification')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                        help='initial learning rate for classification')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for classification if MomentumSGD is used')
    parser.add_argument('--lr_decay_epoch', type=float, default=25,
                        help='decay epoch for classification if MomentumSGD is used')
    parser.add_argument('--random', action='store_true', default=False,
                        help='Use random assignment or not')
    parser.add_argument('--valid', '--v', action='store_true',
                        help='Use random assignment or not')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    gpu = args.gpu
    data_type = args.data_type
    model_type = args.model_type
    num_clusters = args.cluster
    initial_lr = args.initial_lr
    lr_decay_rate = args.lr_decay_rate
    lr_decay_epoch = args.lr_decay_epoch
    opt1 = args.optimizer
    opt2 = args.optimizer2
    model_path = args.model_path
    rand_assign = args.random
    train_file = args.train_file
    test_file = args.test_file

    unit = args.unit
    alpha = args.alpha
    sparse = False

    ndim = 1
    n_in = None
    train_transform = None
    test_transform = None
    if data_type == 'toy':
        model = network.LinearModel(2, 2)
    elif data_type == 'mnist':
        if model_type == 'linear':
            model = network.LinearModel(784, num_clusters)
        elif model_type == 'DNN':
            model = network.MLP(1000, num_clusters)
        elif model_type == 'CNN':
            ndim = 3
            model = network.CNN(num_clusters)
        else:
            raise ValueError
    elif data_type == 'cifar100':
        train_transform = partial(dataset.transform, mean=0.0, std=1.0, train=True)
        test_transform = partial(dataset.transform, mean=0.0, std=1.0, train=False)
        if model_type == 'Resnet50':
            model = network.ResNet50(num_clusters)
            n_in = 2048
            load_npz(model_path, model, not_load_list=['fc7'])
        elif model_type == 'VGG':
            model = network.VGG(num_clusters)
            n_in = 1024
            load_npz(model_path, model, not_load_list=['fc6'])
        else:
            raise ValueError
    elif data_type == 'LSHTC1':
        sparse = True
        if model_type == 'DocModel':
            model = network.DocModel(n_in=1024, n_mid=unit, n_out=num_clusters)
        elif model_type == 'DocModel2':
            model = network.DocModel2(n_in=1024, n_mid=unit, n_out=num_clusters)
        elif model_type == 'linear':
            model = network.LinearModel(n_in=92586, n_out=num_clusters)
        else:
            raise ValueError
    elif data_type == 'Dmoz':
        sparse = True
        if model_type == 'DocModel':
            model = network.DocModel(n_in=561127, n_mid=unit, n_out=num_clusters)
        elif model_type == 'linear':
            model = network.LinearModel(n_in=1024, n_out=num_clusters)
        else:
            raise ValueError
    else:
        if model_type == 'Resnet50':
            model = network.ResNet50(num_clusters)
        elif model_type == 'Resnet101':
            model = network.ResNet101(num_clusters)
        elif model_type == 'VGG':
            model = network.VGG(num_clusters)
        elif model_type == 'CNN':
            model = network.CNN(num_clusters)
        else:
            raise ValueError

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    (train_instances, train_labels), (test_instances, test_labels), num_classes \
        = load_data(data_type, ndim, train_file, test_file)

    if rand_assign:
        assignment, count_classes = random_assignment(num_clusters, num_classes)
    else:
        if opt1 == 'Adam':
            optimizer = chainer.optimizers.Adam(alpha=alpha)
        else:
            optimizer = chainer.optimizers.SGD(lr=alpha)
        optimizer.setup(model)

        train = clustering.dataset.Dataset(*(train_instances, train_labels), sparse)
        test = clustering.dataset.Dataset(*(test_instances, test_labels), sparse)

        train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize)

        train_updater = clustering.updater.Updater(model, train, train_iter, optimizer,
                                                   num_clusters=num_clusters, device=gpu, mu=args.mu)

        trainer = training.Trainer(train_updater, (args.epoch, 'epoch'), out=args.out)

        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/loss_cc',
             'main/loss_mut_info', 'main/H_Y', 'main/H_YX', 'elapsed_time']))
        trainer.extend(extensions.snapshot(), trigger=(5, 'epoch'))

        if args.resume:
            chainer.serializers.load_npz(args.resume, trainer)

        trainer.run()
        """
        end clustering
        """
        cluster_label = separate.det_cluster(model, train, num_classes, batchsize=128, device=gpu, sparse=sparse)

        assignment, count_classes = separate.assign(cluster_label, num_classes, num_clusters)

        del optimizer
        del train_iter
        del train_updater
        del trainer
        del train
        del test

        print(assignment)
    """
    start classification
    """
    model = h_net.HierarchicalNetwork(model, num_clusters, count_classes, n_in=n_in)
    if opt2 == 'Adam':
        optimizer2 = chainer.optimizers.Adam(alpha=initial_lr)
    elif opt2 == 'SGD':
        optimizer2 = chainer.optimizers.SGD(lr=initial_lr)
    else:
        optimizer2 = chainer.optimizers.MomentumSGD(lr=initial_lr)
    optimizer2.setup(model)
    if args.weight_decay > 0:
        optimizer2.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    train = classification.dataset.Dataset(train_instances, train_labels, assignment, _transform=train_transform, sparse=sparse)
    test = classification.dataset.Dataset(test_instances, test_labels, assignment, _transform=test_transform, sparse=sparse)

    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize2)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=1, repeat=False)

    train_updater = classification.updater.Updater(model, train, train_iter, optimizer2, num_clusters, device=gpu)

    trainer = training.Trainer(train_updater, (args.epoch2, 'epoch'), args.out)

    acc = accuracy.Accuracy(model, assignment, num_clusters)
    trainer.extend(extensions.Evaluator(test_iter, acc, device=gpu))

    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/loss_cluster', 'main/loss_class',
         'validation/main/accuracy', 'validation/main/cluster_accuracy',
         'validation/main/loss', 'elapsed_time']))

    if opt2 != 'Adam':
        trainer.extend(extensions.ExponentialShift(
            'lr', lr_decay_rate), trigger=(lr_decay_epoch, 'epoch'))

    if args.resume2:
        chainer.serializers.load_npz(args.resume2, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
