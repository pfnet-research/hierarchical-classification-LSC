import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np


class Updater(chainer.training.StandardUpdater):
    def __init__(self, model, data, iter, optimizer, num_clusters=30, mu=10.0, device=-1):
        self.model = model
        self.data = data
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

        tmp_y = 0.1 * (F.sum(y, axis=0) / batchsize) + 0.9 * self.cum_y
        H_Y = self.entropy(tmp_y, axis=0)
        H_YX = F.sum(self.entropy(y, axis=1), axis=0) / batchsize
        chainer.reporter.report({'main/H_YX': H_YX})
        loss_mut_info = - self.mu * H_Y

        xp = cuda.get_array_module(*instances)

        self.cum_y *= 0.9
        for yy in y.data:
            index = xp.argmax(yy)
            self.cum_y[index] += 0.1 / batchsize
        self.cum_y /= xp.sum(self.cum_y)

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