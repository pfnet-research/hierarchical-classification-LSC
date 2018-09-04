import numpy as np
import random
from chainer.datasets import TupleDataset

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms
import cv2 as cv

from scipy.sparse import csr_matrix

USE_OPENCV = True


def cv_rotate(img, angle):
    if USE_OPENCV:
        img = img.transpose(1, 2, 0) / 255.
        center = (img.shape[0] // 2, img.shape[1] // 2)
        r = cv.getRotationMatrix2D(center, angle, 1.0)
        img = cv.warpAffine(img, r, img.shape[:2])
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    else:
        raise NotImplemented
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img.transpose(1, 2, 0) / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    return img


def transform(
        inputs, mean, std, random_angle=15., pca_sigma=25.5, expand_ratio=1.2,
        crop_size=(28, 28), train=True):
    img = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)

    # Color augmentation
    if train and pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)

    """
    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]
    """

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))

    return img


class Dataset(object):
    def __init__(self, instances, labels, assignment, _transform=None, sparse=False):
        clusters, classes = [assignment[label][0] for label in labels], \
                            [assignment[label][1] for label in labels]
        length = len(labels)
        self._datasets = (instances, clusters, classes)
        self._length = length
        self.transform = _transform
        self.sparse = sparse

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise NotImplementedError
            batches = [dataset[index] for dataset in self._datasets]
            instances = [tuple([self.transform(instance) for instance in batches[2]])]
            clusters = [tuple([cluster for cluster in batches[0]])]
            classes = [tuple([_class for _class in batches[1]])]
            return [instances, clusters, classes]
        else:
            batches = [dataset[index] for dataset in self._datasets]
            instance, cluster, _class = tuple(batches)
            if self.transform is not None:
                instance = self.transform(instance)
            if self.sparse:
                instance = np.array(csr_matrix.todense(instance))
            return instance, cluster, _class

    def __len__(self):
        return self._length
