#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cPickle as pickle
import gzip

import numpy as np


def pickle_dump(path, obj, compress=False):
    if compress:
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path, compressed=False):
    if compressed:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def calculate_rmse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    diff = y_true - y_pred
    return np.sqrt(np.dot(diff.T, diff) / len(y_true))


def calculate_mae(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)
