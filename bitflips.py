import random
import logging
import numpy as np
import torch
import csv
from pytorchfi import core
from finetune import Ground_truth_model

def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model, layer=-1):
    if layer == -1:
        layer = random.randint(0, pfi_model.get_total_layers() - 1)

    shape = pfi_model.get_layer_shape(layer)
    try: 
       _, seq_len, hidden = shape
       h=random.randint(0,hidden-1)
    except ValueError:
        _, seq_len = shape
        h=0

    c = random.randint(0, seq_len-1)
    
    w = 0  # if 2D

    return (layer, c, h, w)


def random_weight_location(pfi_model, layer=-1):
    loc = []

    if layer == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_layers() - 1)
    else:
        corrupt_layer = layer
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi_model.get_original_model().named_parameters():
        if "features" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1

    if curr_layer != pfi_model.get_total_layers():
        raise AssertionError
    if len(loc) != 5:
        raise AssertionError

    return tuple(loc)


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)

def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    (layer, C, H, W) = random_neuron_location(pfi_model)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_neuron_fi(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )

def random_neuron_inj_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    if not randLoc:
        (layer, C, H, W) = random_neuron_location(pfi_model)
    if not randVal:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (layer, C, H, W) = random_neuron_location(pfi_model)
        if randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )

def random_inj_per_layer(pfi_model, min_val=-1, max_val=1):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))
    


    b = random_batch_element(pfi_model)
    for i in range(pfi_model.get_total_layers()):
        if layer_num!=pfi_model.get_total_layers():
            for i in range(0,20):
                (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
                batch.append(b)
                layer_num.append(layer)
                c_rand.append(C)
                h_rand.append(H)
                w_rand.append(W)
                value.append(random_value(min_val=min_val, max_val=max_val))
        else:
            (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )

def random_inj_per_layer_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))
    for i in range(0,pfi_model.get_total_layers()):
        for b in range(pfi_model.get_total_batches()):
            if layer_num!=pfi_model.get_total_layers():
                for j in range(0,20):
                    (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
                    batch.append(b)
                    layer_num.append(layer)
                    c_rand.append(C)
                    h_rand.append(H)
                    w_rand.append(W)
                    value.append(random_value(min_val=min_val, max_val=max_val))
                    print(j)
            else:
                (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
                batch.append(b)
                layer_num.append(layer)
                c_rand.append(C)
                h_rand.append(H)
                w_rand.append(W)
                value.append(random_value(min_val=min_val, max_val=max_val))


        

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )

def random_inj_one_layer_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True, layer_given=0
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))
    i = layer_given
    for b in range(pfi_model.get_total_batches()):
        if layer_num!=pfi_model.get_total_layers():
            for j in range(0,100):
                (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)

                batch.append(b)
                layer_num.append(layer)
                c_rand.append(C)
                h_rand.append(H)
                w_rand.append(W)
                value.append(random_value(min_val=min_val, max_val=max_val))
                print(j)
        else:
            (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(random_value(min_val=min_val, max_val=max_val))


        

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )

def random_weight_inj(pfi_model, corrupt_conv=-1, min_val=-1, max_val=1):
    (layer, k, c_in, kH, kW) = random_weight_location(pfi_model, corrupt_conv)
    faulty_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_weight_fi(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zeroFunc_rand_weight(pfi_model):
    (layer, k, c_in, kH, kW) = random_weight_location(pfi_model)
    return pfi_model.declare_weight_fi(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData