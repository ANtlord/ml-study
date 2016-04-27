#!/usr/bin/env python
from abc import abstractmethod
from typing import Set, List
import os

from titanic import answer
from math import log10
from sklearn.cluster import KMeans
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from skimage import img_as_float
from collections import defaultdict


MAX_FLOAT_COLOR_VAL = 1


class Assimilator:
    @abstractmethod
    def assimilate(self, data: List):
        pass


class MeanAssimilator(Assimilator):
    def assimilate(self, data: List):
        return np.mean(data)


class MedianAssimilator(Assimilator):
    def assimilate(self, data: List):
        return np.median(data)


def assimilate(the_pixels: np.array, by_clusters: List[int], assimilator: Assimilator):
    clustered_colors = defaultdict(list)
    assimilated_colors = defaultdict(list)

    for i, pixel in enumerate(the_pixels):
        clustered_colors[by_clusters[i]].append(pixel)

    for key, val in clustered_colors.items():
        r = assimilator.assimilate([x[0] for x in val])
        g = assimilator.assimilate([x[1] for x in val])
        b = assimilator.assimilate([x[2] for x in val])
        assimilated_colors[key] = [r, g, b]

    return [assimilated_colors[x] for x in by_clusters]


if __name__ == '__main__':
    image = imread('parrots_4.jpg')
    image = img_as_float(image)
    min_clusters = 0
    for i in range(1, 20 + 1):
        train_data = np.vstack(tuple(image.tolist()))
        clr = KMeans(i, init='k-means++', random_state=241)
        train_res = clr.fit_predict(train_data)

        mean_assimilated_colors = assimilate(train_data, train_res, MeanAssimilator())
        median_assimilated_colors = assimilate(train_data, train_res, MedianAssimilator())

        deltas = (train_data - mean_assimilated_colors) ** 2
        mean_mse = sum([x.sum() for x in deltas]) / (train_data.shape[0] * train_data.shape[1])
        mean_psnr = 10 * log10(1/mean_mse)

        deltas = (train_data - median_assimilated_colors) ** 2
        median_mse = sum([x.sum() for x in deltas]) / (train_data.shape[0] * train_data.shape[1])
        median_psnr = 10 * log10(1/median_mse)

        imsave(os.path.join(os.getcwd(), 'mean_assimilated_colors-%s.jpg' % i),
               np.reshape(mean_assimilated_colors, image.shape))
        imsave(os.path.join(os.getcwd(), 'median_assimilated_colors-%s.jpg' % i),
               np.reshape(median_assimilated_colors, image.shape))

        if mean_psnr > 20 or median_psnr > 20:
            min_clusters = i
            break

    answer(str(min_clusters), 'clustering.txt')
