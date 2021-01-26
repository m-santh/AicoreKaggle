import torch
import torchvision

import transforms
import utils
import numpy


class Dataset:
    def __init__(self, args):
        self.args = args
        self.data_X = numpy.load('X_train.npy')
        self.data_Y = numpy.load('y_train.npy')
        self.test_data = numpy.load('X_test.npy')
        self.validation_data, self.train_data = utils.random_split(
            (data_X, data_Y), args.validation_percent
        )

    def validation(self):
        return torch.utils.data.DataLoader(
            self.validation_data,
            batch_size=self.args.batch_size,
            pin_memory=self.args.device == "cuda",
        )

    def train(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=self.args.device == "cuda",
        )
