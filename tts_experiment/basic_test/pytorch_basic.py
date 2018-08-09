# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn


# basic test code
tensor_a = torch.ones(1, 3)
print '**********tensor:\n', tensor_a
var_a = autograd.Variable(tensor_a)
print '********variable:\n', var_a

# make a sequence of length 5
inputs = [var_a*i for i in range(1, 6)]
inputs = torch.cat(inputs)
print '**********inputs:\n', inputs


# input dim is 3, output dim is 8
dim_input = 3
dim_output = 8
lstm = nn.LSTM(dim_input, dim_output)

# initialize the hidden state
hidden = (autograd.Variable(torch.zeros(1, 1, dim_output)),
          autograd.Variable(torch.zeros((1, 1, dim_output))))
print '**********hidden:\n', hidden

# we can do the entire sequence all at once
# the first value returned by LSTM is all of the hidden states throughout the sequence
# the second is just the most recent hidden state
output, hidden = lstm(inputs.view(len(inputs), 1, -1), hidden)
print '**********output:\n', torch.cat(output)
print '**********hidden:\n', torch.cat(hidden)









