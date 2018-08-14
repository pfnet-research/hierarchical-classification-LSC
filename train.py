import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
import random
import mnist
import cifar
import argparse
import network


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
    def __init__(self, model, data, iter, optimizer, lam=0.2, mu=6.0, device=-1):
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
        y = self.model(instances)

        H_Y = self.entropy((F.sum(y, axis=0) / batchsize), axis=0)
        H_YX = F.sum(self.entropy(y, axis=1), axis=0) / batchsize
        loss_mut_info = - self.lam * (self.mu * H_Y - H_YX)

        # sampled instancesがリストになっているが、これがnumpy arrayになっているハズ
        sampled_y = self.model(sampled_instances)
        sampled_y.unchain_backward()

        loss_cc = self.loss_class_clustering(y, sampled_y) / batchsize

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
    if data_type == 'toy':
        return data_generate(), data_generate()
    elif data_type == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.get_mnist(ndim=ndim)
        return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
    elif data_type == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar.get_cifar10()
        return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
    else:
        raise ValueError


def check_cluster(model, train, num_classes, num_cluster, device=-1):
    xx = model(chainer.dataset.convert.concat_examples(train, device=device)[0]).data
    if device >= 0:
        xx = cuda.to_cpu(xx)
    cc = np.argmax(xx, axis=1)

    partition = train._partition
    cluster = [tuple(np.sum(cc[partition[k]:partition[k + 1]] == c)
                     for c in range(num_cluster)) for k in range(num_classes)]
    return cluster


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--data_type', '-d', type=str, default='mnist')
    parser.add_argument('--model_type', '-m', type=str, default='CNN')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--cluster', '-c', type=int, default=2)
    args = parser.parse_args()

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
        elif model_type == 'MLP':
            model = network.MLP(1000, num_cluster)
        elif model_type == 'CNN':
            ndim = 3
            model = network.CNN(num_cluster)
        else:
            raise ValueError
    else:
        num_classes = 10
        if model_type == 'Resnet50':
            model = network.ResNet50(num_cluster)
        else:
            raise ValueError

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = load_data(data_type, ndim)

    train_iter = chainer.iterators.SerialIterator(train, batch_size=256)

    updater = Updater(model, train, train_iter, optimizer, device=gpu)

    trainer = training.Trainer(updater, (10, 'epoch'), out='result/result.txt')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/loss_cc', 'main/loss_mut_info', 'main/H_Y', 'main/H_YX', 'elapsed_time']))

    trainer.run()

    res = check_cluster(model, train, num_classes, num_cluster, device=gpu)
    print(res)
    res = check_cluster(model, test, num_classes, num_cluster, device=gpu)
    print(res)

if __name__ == '__main__':
    main()