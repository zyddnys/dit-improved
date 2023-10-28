
import numpy as np
import torch

class AvgMeter() :
	def __init__(self) :
		self.reset()

	def reset(self) :
		self.sum = 0
		self.count = 0

	def __call__(self, val = None, reset = False) :
		if val is not None :
			self.sum += val
			self.count += 1
		if self.count > 0 :
			ret = self.sum / self.count
		else :
			ret = 0
		if reset :
			self.reset()
		return ret

def multilabel_classification_metrics( probs, labels, eps = 0.001 ) : # average=samples
	p, r, fs = 0, 0, 0
	for i in range( probs.shape[0] ) :
		tp = np.sum( [ 1 if x >= 0.5 and y >= 0.5 else 0 for x, y in zip( probs[i], labels[i] ) ] )
		tn = np.sum( [ 1 if x < 0.5 and y < 0.5 else 0 for x, y in zip( probs[i], labels[i] ) ] )
		fp = np.sum( [ 1 if x >= 0.5 and y < 0.5 else 0 for x, y in zip( probs[i], labels[i] ) ] )
		fn = np.sum( [ 1 if x < 0.5 and y >= 0.5 else 0 for x, y in zip( probs[i], labels[i] ) ] )
		p_ = tp / ( tp + fp + eps )
		r_ = tp / ( tp + fn + eps )
		fs_ = 2 * ( p_ * r_ ) / ( p_ + r_ + eps )
		p += p_
		r += r_
		fs += fs_
	p /= probs.shape[0]
	r /= probs.shape[0]
	fs /= probs.shape[0]
	return p, r, fs

import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
