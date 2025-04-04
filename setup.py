import random
import logging
import numpy as np
import torch
import csv
from pytorchfi import core
from finetune import Ground_truth_model
from errors import stuck_at_one, stuck_at_zero, bit_flip
from bitflips import *
import matplotlib.pyplot as plt

class Perturbed_model:
    def __init__(self,model_,tokenizer,batch_size,input_shape,layer_types,use_cuda=False):
        self.ground_truth=model_
        self.inj_model=core.fault_injection(self.ground_truth.model,batch_size,input_shape,layer_types,use_cuda)
        self.tokenizer=tokenizer
    def setup(self):
        self.inj_model._traverse_model_set_hooks()
        print(self.inj_model.print_pytorchfi_layer_summary)
    def experiment(self,test_loader,n=100):
        accuracies=[]
        for i in range(n):
            # Inject faults into the model
            # Here, we use random neuron perturbation as an example
            # You can customize the fault injection parameters as needed
            random_inj_per_layer_batched(self.inj_model)
                
            # Evaluate the faulty model
            accuracy = self.evaluate(self.inj_model, test_loader)
            accuracies.append(accuracy)
                
                # Reset the fault injection for the next iteration
            self.inj_model.fi_reset()

        # Visualize the accuracy results
        plt.plot(range(n), accuracies, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Under Fault Injections')
        plt.show()
    def evaluate(self, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        total, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = (t.to(device) for t in batch)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

