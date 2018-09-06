import chainer
import chainer.links as L
import chainer.functions as F


class LinearModel(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(LinearModel, self).__init__()
        with self.init_scope():
            self.w = L.Linear(n_in, n_out)

    def conv(self, x, unchain=False):
        return x

    def cluster(self, h):
        return self.w(h)

    def __call__(self, x, unchain=False):
        return self.w(x)


class DocModel(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, relu=False):
        super(DocModel, self).__init__()
        with self.init_scope():
            self.w1 = L.Linear(n_in, n_mid)
            self.w2 = L.Linear(n_mid, n_out)

    def conv(self, x, unchain=False):
        return F.relu(self.w1(x))

    def cluster(self, h):
        return self.w2(h)

    def __call__(self, x, unchain=False):
        h = F.relu(self.w1(x))
        return self.w2(h)


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def conv(self, x, unchain=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return h2

    def cluster(self, h):
        return self.l3(h)

    def __call__(self, x, unchain=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class CNN(chainer.Chain):
    def __init__(self, num_cluster):
        super(CNN, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(None, 32, 5)
            self.cn2 = L.Convolution2D(32, 64, 5)
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(500, num_cluster)

    def conv(self, x, unchain=False):
        h = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.cn2(h)), 2)
        h = F.relu(self.fc1(h))
        return h

    def cluster(self, h):
        return self.fc2(h)

    def __call__(self, x, unchain=False):
        h = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.cn2(h)), 2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


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

    def __call__(self, x, unchain=False):
        h = F.relu(self.bn2(self.conv1(x)))
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        if unchain:
            h.unchain_backward()
        h = self.res6(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.fc7(h)
        return h

    def conv(self, x, unchain=False):
        h = F.relu(self.bn2(self.conv1(x)))
        h = self.res3(h)
        h = self.res4(h)
        if unchain:
            h.unchain_backward()
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        return h

    def cluster(self, h):
        h = self.fc7(h)
        return h

    def serialize(self, serializer, not_load_list=None):
        super(chainer.Chain, self).serialize(serializer)
        if not_load_list is None:
            not_load_list = []

        d = self.__dict__
        for name in self._children:
            if name in not_load_list:
                continue
            d[name].serialize(serializer[name])


class ResNet50(ResNet):

    def __init__(self, n_class=10):
        super(ResNet50, self).__init__(n_class, [3, 4, 6, 3])


class ResNet101(ResNet):

    def __init__(self, n_class=10):
        super(ResNet101, self).__init__(n_class, [3, 4, 23, 3])


class ResNet152(ResNet):

    def __init__(self, n_class=10):
        super(ResNet152, self).__init__(n_class, [3, 8, 36, 3])


class VGG(chainer.Chain):

    def __init__(self, n_class=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, pad=1)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_3 = L.BatchNormalization(256)
            self.conv3_4 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_4 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024)
            self.fc5 = L.Linear(1024, 1024)
            self.fc6 = L.Linear(None, n_class)

    def __call__(self, x, unchain=False):
        if unchain:
            with chainer.using_config('train', False):
                h = F.relu(self.bn1_1(self.conv1_1(x)))
                h = F.relu(self.bn1_2(self.conv1_2(h)))
                h = F.max_pooling_2d(h, 2, 2)

                h = F.relu(self.bn2_1(self.conv2_1(h)))
                h = F.relu(self.bn2_2(self.conv2_2(h)))
                h = F.max_pooling_2d(h, 2, 2)

                h = F.relu(self.bn3_1(self.conv3_1(h)))
                h = F.relu(self.bn3_2(self.conv3_2(h)))
                h = F.relu(self.bn3_3(self.conv3_3(h)))
                h = F.relu(self.bn3_4(self.conv3_4(h)))
                h = F.max_pooling_2d(h, 2, 2)
        else:
            h = F.relu(self.bn1_1(self.conv1_1(x)))
            h = F.relu(self.bn1_2(self.conv1_2(h)))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)

            h = F.relu(self.bn2_1(self.conv2_1(h)))
            h = F.relu(self.bn2_2(self.conv2_2(h)))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)

            h = F.relu(self.bn3_1(self.conv3_1(h)))
            h = F.relu(self.bn3_2(self.conv3_2(h)))
            h = F.relu(self.bn3_3(self.conv3_3(h)))
            h = F.relu(self.bn3_4(self.conv3_4(h)))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)

        h = F.relu(self.fc4(h))
        if unchain:
            h.unchain_backward()
        h = F.relu(self.fc5(h))
        h = self.fc6(h)
        return h

    def conv(self, x, unchain=False):
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc5(h)), ratio=0.5)

        return h

    def cluster(self, h):
        h = self.fc6(h)
        return h

    def serialize(self, serializer, not_load_list=None):
        super(chainer.Chain, self).serialize(serializer)
        if not_load_list is None:
            not_load_list = []

        d = self.__dict__
        for name in self._children:
            if name in not_load_list:
                continue
            d[name].serialize(serializer[name])