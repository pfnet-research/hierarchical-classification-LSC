import chainer
import chainer.links as L
import chainer.functions as F


class Hierarchical_Network(chainer.Chain):
    def __init__(self, model, num_cluster, class_list):
        super(Hierarchical_Network, self).__init__()
        self.model = model
        self.fc_list = []
        for i in range(num_cluster):
            fc = L.Linear(None, class_list[i])
            self.fc_list.append(fc)

    def __call__(self, model, t):
        cluster, clas = t
        h = self.model.conv
        cluster_output = F.softmax(self.model.cluster(h))
        class_output = F.softmax(self.fc_list[cluster](h))
        return cluster_output[cluster] * class_output[clas]
