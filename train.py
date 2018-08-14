import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import random
import mnist


class LinearModel(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(LinearModel, self).__init__()
        with self.init_scope():
            self.w = L.Linear(n_in, n_out)

    def __call__(self, x):
        return F.softmax(self.w(x))


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return F.softmax(self.l3(h2))


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
    def __init__(self, model, data, iter, optimizer, lam=0.1, mu=4.0, device=-1):
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


def load_data(data_type='toy'):
    if data_type == 'toy':
        return data_generate()
    else:
        (train_images, train_labels), _ = mnist.get_mnist()
        return Dataset(train_images, train_labels)


def check_cluster(model, train, num_classes, num_cluster):
    xx = model(chainer.dataset.convert.concat_examples(train)[0])
    cc = np.argmax(xx.data, axis=1)

    partition = train._partition
    cluster = [tuple(np.sum(cc[partition[k]:partition[k + 1]] == c)
                     for c in range(num_cluster)) for k in range(num_classes)]
    return cluster


def main():
    data_type = 'mnist'
    model_type = 'MLP'
    gpu = -1

    if data_type == 'toy':
        model = LinearModel(2, 2)
        num_cluster = 2
        num_classes = 2
    else:
        num_cluster = 2
        num_classes = 10
        if model_type == 'linear':
            model = LinearModel(784, num_cluster)
        elif model_type == 'MLP':
            model = MLP(1000, num_cluster)
        else:
            raise ValueError

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = load_data(data_type)

    train_iter = chainer.iterators.SerialIterator(train, batch_size=128)

    updater = Updater(model, train, train_iter, optimizer)

    trainer = training.Trainer(updater, (5, 'epoch'), out='result/result.txt')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss_cc', 'main/loss_mut_info', 'main/H_Y', 'main/H_YX', 'elapsed_time']))

    trainer.run()

    print(check_cluster(model, train, num_classes, num_cluster))


if __name__ == '__main__':
    main()