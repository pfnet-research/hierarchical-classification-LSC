from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class HierarchicalClassifier(link.Chain):
    def __init__(self, network):
        super(HierarchicalClassifier, self).__init__()
        self.network = network
