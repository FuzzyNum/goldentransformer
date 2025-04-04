import random
import logging
import numpy as np
import torch
import csv
from pytorchfi import core
from finetune import Ground_truth_model

class stuck_at_one(core.fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def error(self, module, input, output):
        shape = output.shape
        output = output.flatten()
        max_bit = random.randint(0,31)
        mask = (0x00000001 << max_bit+1) - 1
        for i in range(len(output)):
            # Get the float as an integer
            int_value = output[i].view(torch.int)
            # operate with the integers to inject faults using a mask
            # Stuck-at-1
            faulty = torch.bitwise_or(int_value, mask)
            # Get the faulty integer as a float
            output[i] = faulty.view(torch.float)
        # Reshape the flattened tensor
        output = output.view(shape)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

class stuck_at_zero(core.fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def mul_neg_one(self, module, input, output):
        shape = output.shape
        output = output.flatten()
        max_bit = random.randint(0,31)
        mask = (0x00000001 << max_bit+1) - 1
        for i in range(len(output)):
            # Get the float as an integer
            int_value = output[i].view(torch.int)
            # operate with the integers to inject faults using a mask
            # Stuck-at-1
            faulty = torch.bitwise_and(int_value, mask)
            # Get the faulty integer as a float
            output[i] = faulty.view(torch.float)
        # Reshape the flattened tensor
        output = output.view(shape)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()
class bit_flip(core.fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def mul_neg_one(self, module, input, output):
        shape = output.shape
        output = output.flatten()
        max_bit = random.randint(0,31)
        mask = (0x00000001 << max_bit+1) - 1
        for i in range(len(output)):
            # Get the float as an integer
            int_value = output[i].view(torch.int)
            # operate with the integers to inject faults using a mask
            # Stuck-at-1
            faulty = torch.bitwise_xor(int_value, mask)
            # Get the faulty integer as a float
            output[i] = faulty.view(torch.float)
        # Reshape the flattened tensor
        output = output.view(shape)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()
