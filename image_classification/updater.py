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
from chainer.iterators import serial_iterator


class Updater(chainer.training.StandardUpdater):
    def __init__(self, model, data, iter, optimizer, num_clusters, device=-1):
        self.model = model
        self.data = data
        self.num_clusters = num_clusters
        super(Updater, self).__init__(iter, optimizer, device=device)

    def update_core(self):
        self.model.cleargrads()

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        partition = np.zeros(self.num_clusters+1).astype(np.int32)
        for _, cluster, _ in batch:
            partition[cluster+1] += 1
        for i in range(1, self.num_clusters+1):
            partition[i] += partition[i-1]

        optimizer = self._optimizers['main']

        # tuple of instance array and label array
        instances, clusters, classes = self.converter(batch, self.device)
        xp = cuda.get_array_module(*instances)
        index = xp.argsort(clusters)
        instances, clusters, classes = xp.take(instances, index, axis=0), \
                                       xp.take(clusters, index), xp.take(classes, index)

        y, cluster_y, class_y = self.model.train(instances, clusters, classes, partition)
        loss = self.mle_loss(y) / batchsize
        loss_cluster = self.mle_loss(cluster_y) / batchsize
        loss_class = self.mle_loss(class_y) / batchsize

        loss.backward()

        optimizer.update()

        chainer.reporter.report({'main/loss': loss})
        chainer.reporter.report({'main/loss_cluster': loss_cluster})
        chainer.reporter.report({'main/loss_class': loss_class})

    @staticmethod
    def mle_loss(p):
        return -F.sum(F.log(p + 1e-8))
