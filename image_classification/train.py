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
import network
from importlib import import_module
import sys, os
import numpy as np
from scipy.sparse import csr_matrix

import hierarchy_network as h_net

import cifar
import mnist
import doc_preprocess

import separate
import dataset
import updater
import accuracy
import six


class Dataset(object):
    def __init__(self, instances, labels, sparse=False):
        print(instances.shape, labels.shape)
        instances, labels = instances[np.argsort(labels)], np.sort(labels)
        label_type = np.unique(labels)
        partition = [np.searchsorted(labels, k, side='left') for k in label_type]
        partition.append(np.size(instances, axis=0))

        self._instances = instances
        self._labels = labels
        self._length = len(labels)
        self._partition = partition
        self.sparse = sparse

    def __getitem__(self, index):
        if isinstance(index, slice):
            instances = self._instances[index]

            if self.sparse:
                instances = np.array(csr_matrix.todense(instances))

            labels = self._labels[index]

            length = len(instances)

            # バッチ内の各ラベルからランダムに要素を取り出す。
            sampled_instances = [self._instances[random.choice(range(self._partition[label],
                                                                     self._partition[label + 1]))]
                                 for label in labels]
            if self.sparse:
                sampled_instances = [np.array(csr_matrix.todense(sampled_instance)) for sampled_instance in sampled_instances]

            return [(instances[i], labels[i], sampled_instances[i]) for i in range(length)]
        else:
            instance = self._instances[index]
            if self.sparse:
                instance = np.array(csr_matrix.todense(instance))

            label = self._labels[index]
            sampled_instance = self._instances[random.choice(range(self._partition[label], self._partition[label + 1]))]

            if self.sparse:
                sampled_instance = np.array(csr_matrix.todense(sampled_instance))

            return instance, label, sampled_instance

    def __len__(self):
        return self._length


class Updater(chainer.training.StandardUpdater):
    def __init__(self, model, data, iter, optimizer, num_clusters = 30, lam=0.5, mu=10.0, device=-1):
        self.model = model
        self.data = data
        self.lam = lam
        self.mu = mu

        self.cum_y = np.ones(num_clusters) / num_clusters
        if device >= 0:
            self.cum_y = cuda.to_gpu(self.cum_y, device)
        super(Updater, self).__init__(iter, optimizer, device=device)

    def update_core(self):
        self.model.cleargrads()

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        optimizer = self._optimizers['main']

        # tuple of instance array and label array
        instances, labels, sampled_instances = self.converter(batch, self.device)
        y = F.softmax(self.model(instances, unchain=True))

        print(F.sum(y, axis=0).shape)

        tmp_y = 0.1 * (F.sum(y, axis=0) / batchsize) + 0.9 * self.cum_y
        H_Y = self.entropy(tmp_y, axis=0)
        H_YX = F.sum(self.entropy(y, axis=1), axis=0) / batchsize
        chainer.reporter.report({'main/H_YX': H_YX})
        H_YX = 0
        loss_mut_info = - self.lam * (self.mu * H_Y - H_YX)

        xp = cuda.get_array_module(*instances)

        self.cum_y *= 0.9
        for yy in y.data:
            index = xp.argmax(yy)
            self.cum_y[index] += 0.1 / batchsize
        self.cum_y /= xp.linalg.norm(self.cum_y)

        # sampled instancesがリストになっているが、これがnumpy arrayになっているハズ
        with chainer.using_config('train', False):
            sampled_y = F.softmax(self.model(sampled_instances))

        loss_cc = self.loss_cross_entropy(y, sampled_y) / batchsize

        loss = loss_cc + loss_mut_info
        loss.backward()

        optimizer.update()

        chainer.reporter.report({'main/loss': loss})
        chainer.reporter.report({'main/loss_cc': loss_cc})
        chainer.reporter.report({'main/loss_mut_info': loss_mut_info})
        chainer.reporter.report({'main/H_Y': H_Y})

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
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--batchsize2', '-b2', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--data_type', '-d', type=str, default='LSHTC1')
    parser.add_argument('--model_type', '-m', type=str, default='DocModel')
    parser.add_argument('--model_path', '-mp', type=str,
                        default='./models/ResNet50_model_500.npz')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--cluster', '-c', type=int, default=100)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0000)
    parser.add_argument('--unit', '-u', type=int, default=300)
    parser.add_argument('--alpha', '-a', type=float, default=0.005)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--epoch2', '-e2', type=int, default=10)
    parser.add_argument('--mu', '-mu', type=float, default=30.0)
    parser.add_argument('--out', '-o', type=str, default='results')

    parser.add_argument('--train_file', '-train_f', type=str, default='dataset/LSHTC1/LSHTC1_selected03.train')
    parser.add_argument('--test_file', '-test_f', type=str, default='dataset/LSHTC1/LSHTC1_selected03.test')

    parser.add_argument('--train_instance', '-train_i', type=str, default='PDSparse/examples/LSHTC1/LSHTC1.train')
    parser.add_argument('--train_label', '-train_l', type=str, default='PDSparse/examples/LSHTC1/LSHTC1.train')
    parser.add_argument('--test_instance', '-test_i', type=str, default='PDSparse/examples/LSHTC1/LSHTC1.train')
    parser.add_argument('--test_label', '-test_l', type=str, default='PDSparse/examples/LSHTC1/LSHTC1.train')

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--resume2', '-r2', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--optimizer', '-op', type=str, default='Adam')
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
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
    opt = args.optimizer
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
        num_classes = 4
    elif data_type == 'mnist':
        num_classes = 10
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
        num_classes = 100
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
        num_classes = None
        if model_type == 'DocModel':
            model = network.DocModel(n_in=1024, n_mid=unit, n_out=num_clusters)
        elif model_type == 'linear':
            model = network.LinearModel(n_in=92586, n_out=num_clusters)
        else:
            raise ValueError
    elif data_type == 'Dmoz':
        sparse = True
        num_classes = None
        if model_type == 'DocModel':
            model = network.DocModel(n_in=561127, n_mid=unit, n_out=num_clusters)
        elif model_type == 'linear':
            model = network.LinearModel(n_in=1024, n_out=num_clusters)
        else:
            raise ValueError
    else:
        num_classes = 10
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
        optimizer = chainer.optimizers.Adam(alpha=alpha)
        optimizer.setup(model)

        train = Dataset(*(train_instances, train_labels), sparse)
        test = Dataset(*(test_instances, test_labels), sparse)

        train_iter = chainer.iterators.MultiprocessIterator(train, batch_size=args.batchsize)

        train_updater = Updater(model, train, train_iter, optimizer, device=gpu, mu=args.mu)

        trainer = training.Trainer(train_updater, (args.epoch, 'epoch'), out=args.out)

        trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
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
        """
        res, ss = check_cluster(model, train, num_classes, num_clusters, device=gpu)
        res_sum = tuple(0 for _ in range(num_clusters))
        for i in range(num_classes):
            res_sum = tuple(res_sum[j] + res[i][j] for j in range(num_clusters))
        print(res, res_sum, ss)
        """

        """
        res, ss = check_cluster(model, test, num_classes, num_clusters, device=gpu)
        res_sum = tuple(0 for _ in range(num_clusters))
        for i in range(num_classes):
            res_sum = tuple(res_sum[j] + res[i][j] for j in range(num_clusters))
        """
        cluster_label = separate.det_cluster(model, train, num_classes, batchsize=128, device=gpu, sparse=sparse)

        assignment, count_classes = separate.assign(cluster_label, num_classes, num_clusters)

        del optimizer
        del train_iter
        del train_updater
        del trainer
        del train
        del test

    print(count_classes)

    """
    start classification
    """
    model = h_net.HierarchicalNetwork(model, num_clusters, count_classes, n_in=n_in)
    if opt == 'Adam':
        optimizer2 = chainer.optimizers.Adam(alpha=initial_lr)
    elif opt == 'SGD':
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

    train = dataset.Dataset(train_instances, train_labels, assignment, _transform=train_transform, sparse=sparse)
    test = dataset.Dataset(test_instances, test_labels, assignment, _transform=test_transform, sparse=sparse)

    train_iter = chainer.iterators.MultiprocessIterator(train, batch_size=args.batchsize2)
    test_iter = chainer.iterators.MultiprocessIterator(test, batch_size=args.batchsize2, repeat=False)

    train_updater = updater.Updater(model, train, train_iter, optimizer2, num_clusters, device=gpu)

    trainer = training.Trainer(train_updater, (args.epoch2, 'epoch'), args.out)

    acc = accuracy.Accuracy(model, assignment, num_clusters)
    trainer.extend(extensions.Evaluator(test_iter, acc, device=gpu))

    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/loss_cluster', 'main/loss_class',
         'validation/main/accuracy', 'validation/main/cluster_accuracy',
         'validation/main/loss', 'validation/main/loss_cluster',
         'validation/main/loss_class']))

    if opt != 'Adam':
        trainer.extend(extensions.ExponentialShift(
            'lr', lr_decay_rate), trigger=(lr_decay_epoch, 'epoch'))

    if args.resume2:
        chainer.serializers.load_npz(args.resume2, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
