import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda


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


class CNN(chainer.Chain):
    def __init__(self, num_cluster):
        super(CNN, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(None, 20, 5)
            self.cn2 = L.Convolution2D(20, 50, 5)
            self.fc1 = L.Linear(800, 500)
            self.fc2 = L.Linear(500, num_cluster)

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
        h3 = F.relu(self.fc1(h2))
        return F.softmax(self.fc2(h3))

class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, use_conv=False):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in, n_mid, 1, stride, 0, True, w)
            self.bn1 = L.BatchNormalization(n_mid)
            self.conv2 = L.Convolution2D(n_mid, n_mid, 3, 1, 1, True, w)
            self.bn2 = L.BatchNormalization(n_mid)
            self.conv3 = L.Convolution2D(n_mid, n_out, 1, 1, 0, True, w)
            self.bn3 = L.BatchNormalization(n_out)
            if use_conv:
                self.conv4 = L.Convolution2D(
                    n_in, n_out, 1, stride, 0, True, w)
                self.bn4 = L.BatchNormalization(n_out)
        self.use_conv = use_conv

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return h + self.bn4(self.conv4(x)) if self.use_conv else h + x


class Block(chainer.ChainList):

    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
        for _ in range(n_bottlenecks - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet(chainer.Chain):

    def __init__(self, n_class=10, n_blocks=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, 1, 0, True, w)
            self.bn2 = L.BatchNormalization(64)
            self.res3 = Block(64, 64, 256, n_blocks[0], 1)
            self.res4 = Block(256, 128, 512, n_blocks[1], 2)
            self.res5 = Block(512, 256, 1024, n_blocks[2], 2)
            self.res6 = Block(1024, 512, 2048, n_blocks[3], 2)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.relu(self.bn2(self.conv1(x)))
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.fc7(h)
        return F.softmax(h)


class ResNet50(ResNet):

    def __init__(self, n_class=10):
        super(ResNet50, self).__init__(n_class, [3, 4, 6, 3])


class ResNet101(ResNet):

    def __init__(self, n_class=10):
        super(ResNet101, self).__init__(n_class, [3, 4, 23, 3])


class ResNet152(ResNet):

    def __init__(self, n_class=10):
        super(ResNet152, self).__init__(n_class, [3, 8, 36, 3])

