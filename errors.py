import random
import logging
import numpy as np
import torch
import csv
from finetune import Ground_truth_model
import modified_fi

class stuck_at_one(modified_fi.ModifiedFaultInjection):
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

class stuck_at_zero(modified_fi.ModifiedFaultInjection):
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
            faulty = torch.bitwise_and(int_value, mask)
            # Get the faulty integer as a float
            output[i] = faulty.view(torch.float)
        # Reshape the flattened tensor
        output = output.view(shape)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()
class bit_flip(modified_fi.ModifiedFaultInjection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    def error(self, module, input, output):
        shape = output.shape
        output = output.flatten()

        for i in range(len(output)):
            float_val = output[i].item()
            int_val = np.frombuffer(np.float32(float_val).tobytes(), dtype=np.uint32)[0]

            # Flip one random bit
            bit = random.randint(0, 31)
            flipped = int_val ^ (1 << bit)

            # Convert back to float
            new_float = np.frombuffer(np.uint32(flipped).tobytes(), dtype=np.float32)[0]
            output[i] = torch.tensor(new_float, dtype=torch.float32)

        output = output.view(shape)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()


class single_bit_flip_func(modified_fi.ModifiedFaultInjection):
    def __init__(self, model, batch_size, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.bits = kwargs.get("bits", 8)
        self.LayerRanges = []

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    def _twos_comp_shifted(self, val, nbits):
        if val < 0:
            val = (1 << nbits) + val
        else:
            val = self._twos_comp(val, nbits)
        return val

    def _twos_comp(self, val, bits):
        # compute the 2's complement of int value val
        if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)  # compute negative value
        return val  # return positive value as is

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("orig value:", orig_value)

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("quantum:", quantum)
        logging.info("twos_comple:", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("bits:", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logging.info("sign extend bits", bits)

        # flip a bit
        # use MSB -> LSB indexing
        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("bits", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("out", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("new_value", new_value)

        return torch.tensor(new_value, dtype=save_type)

    def single_bit_flip_signed_across_batch(self, module, input, output):
        corrupt_conv_set = self.get_corrupt_layer()
        range_max = self.get_conv_max(self.get_curr_layer())
        logging.info("curr_conv", self.get_curr_layer())
        logging.info("range_max", range_max)

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_curr_layer(),
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                prev_value = output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]][
                    self.CORRUPT_DIM2[i]
                ][self.CORRUPT_DIM3[i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]][
                    self.CORRUPT_DIM2[i]
                ][self.CORRUPT_DIM3[i]] = new_value

        else:
            self.assert_inj_bounds()
            if self.get_curr_layer() == corrupt_conv_set:
                prev_value = output[self.CORRUPT_BATCH][self.CORRUPT_DIM1][
                    self.CORRUPT_DIM2
                ][self.CORRUPT_DIM3]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH][self.CORRUPT_DIM1][self.CORRUPT_DIM2][
                    self.CORRUPT_DIM3
                ] = new_value

        self.updateLayer()
        if self.get_curr_layer() >= self.get_total_layers():
            self.reset_curr_layer()
