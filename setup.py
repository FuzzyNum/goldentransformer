import random
import logging
import numpy as np
import torch
import csv
from pytorchfi import core
from finetune import ground_truth_model

class perturbed_model:
    def __init__(self,model,tokenizer,batch_size,input_shape,layer_types,use_cuda=False):
        self.ground_truth=model
        self.inj_model=core.fault_injection(self.ground_truth,batch_size,input_shape,layer_types,use_cuda)
        self.tokenizer=tokenizer
    def n_per_layer_experiment(self,n=100):