import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda
import numpy


class HierarchicalNetwork(chainer.ChainList):
    def __init__(self, model, num_clusters, class_list):
        super(HierarchicalNetwork, self).__init__()

        self.num_clusters = num_clusters
        self.model = model
        for i in range(num_clusters):
            fc = L.Linear(None, class_list[i])
            self.add_link(fc)
        self.add_link(model)

    def train(self, x, cluster_array, class_array, partition):
        """
        :param x: instance in a minibatch
        :param cluster_array: the array of instances' cluster
        :param class_array: the array of instances' class
        (class means the number in the cluster)
        :param partition:
        :return: xp array, xp array
        """
        h = self.model.conv(x)
        cluster_output = F.softmax(self.model.cluster(h))
        class_output = None

        """
        for i in range(2):
            print(self[i].W.data, self[i].b.data)
        print("")
        """

        for cluster in range(self.num_clusters):
            if class_output is None:
                class_output = F.softmax(self[cluster](h[partition[cluster]:partition[cluster+1]]))
            else:
                output = F.softmax(self[cluster](h[partition[cluster]:partition[cluster+1]]))
                class_output = F.concat((class_output, output), axis=0)

        return F.select_item(class_output, class_array) * F.select_item(class_output, class_array)

    def inference(self, x):
        """
        :param x: instance in a minibatch
        :return: two numpy array
        """
        xp = cuda.get_array_module(*x)

        h = self.model.conv(x)
        cluster_array = xp.argmax(self.model.cluster(h).data, axis=1)

        class_array = None
        for i, index in enumerate(cluster_array):
            class_index = xp.argmax(self[index](h[i:i+1]).data, axis=1)

            if class_array is None:
                class_array = class_index
            else:
                class_array = xp.concatenate((class_array, class_index))

        return cluster_array, class_array
