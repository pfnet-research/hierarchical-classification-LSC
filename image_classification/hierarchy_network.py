import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda
import numpy


class HierarchicalNetwork(chainer.ChainList):
    def __init__(self, model, num_clusters, class_list, n_in=None):
        super(HierarchicalNetwork, self).__init__()

        self.num_clusters = num_clusters
        self.model = model
        for i in range(num_clusters):
            fc = L.Linear(n_in, class_list[i])
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
        cluster_output = F.select_item(cluster_output, cluster_array)

        class_output = None

        for cluster in range(self.num_clusters):
            if partition[cluster] == partition[cluster+1]:
                continue
            output = F.softmax(self[cluster](h[partition[cluster]:partition[cluster + 1]]))
            output = F.select_item(output, class_array[partition[cluster]:partition[cluster + 1]])
            if class_output is None:
                class_output = output
            else:
                class_output = F.concat((class_output, output), axis=0)

        return cluster_output * class_output, cluster_output, class_output

    def inference(self, x):
        """
        :param x: instance in a minibatch
        :return: two numpy array
        """
        xp = cuda.get_array_module(*x)

        h = self.model.conv(x)
        cluster_array = xp.argmax(self.model.cluster(h).data, axis=1)
        if xp != numpy:
            cluster_array_cpu = cuda.to_cpu(cluster_array)
        else:
            cluster_array_cpu = cluster_array

        class_array = None
        for i, index in enumerate(cluster_array_cpu):
            class_index = xp.argmax(self[index](h[i:i+1]).data, axis=1)

            if class_array is None:
                class_array = class_index
            else:
                class_array = xp.concatenate((class_array, class_index))

        return cluster_array, class_array
