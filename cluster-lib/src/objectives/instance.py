"""
Non-parametric Instance Discrimination Loss
https://github.com/zhirongw/lemniscate.pytorch

Code is based on Tensorflow implementation: 
https://github.com/neuroailab/LocalAggregation
"""
import math
import torch
import numpy as np

from src.utils.tensor import l2_normalize


class InstanceDiscriminationLoss(object):

    def __init__(self, indices, outputs, memory_bank, k=4096, t=0.07, m=0.5):
        self.k, self.t, self.m = k, t, m

        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)

        self.memory_bank = memory_bank
        self.device = outputs.device
        self.data_len = memory_bank.size
    
    def _softmax(self, dot_prods):
        Z = 2876934.2 / 1281167 * self.data_len
        return torch.exp(dot_prods / self.t) / Z

    def updated_new_data_memory(self):
        data_memory = self.memory_bank.at_idxs(self.indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * self.outputs
        return l2_normalize(new_data_memory, dim=1)

    def compute_data_prob(self):
        logits = self.memory_bank.get_dot_products(self.outputs, self.indices)
        return self._softmax(logits)

    def compute_noise_prob(self):
        batch_size = self.indices.size(0)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k), 
                                   device=self.device)  # U(0, data_len)
        noise_indx = noise_indx.long()
        logits = self.memory_bank.get_dot_products(self.outputs, noise_indx)
        noise_probs = self._softmax(logits)
        return noise_probs

    def get_loss(self, *args):
        batch_size = self.indices.size(0)
        data_prob = self.compute_data_prob()
        noise_prob = self.compute_noise_prob()

        assert data_prob.size(0) == batch_size
        assert noise_prob.size(0) == batch_size
        assert noise_prob.size(1) == self.k

        base_prob = 1.0 / self.data_len
        eps = 1e-7
       
        ## Pmt
        data_div = data_prob + (self.k * base_prob + eps)
        ln_data = torch.log(data_prob) - torch.log(data_div)
        
        ## Pon
        noise_div = noise_prob + (self.k * base_prob + eps)
        ln_noise = math.log(self.k * base_prob) - torch.log(noise_div)

        curr_loss = -(torch.sum(ln_data) + torch.sum(ln_noise))
        curr_loss = curr_loss / batch_size
       
        # for tracking purposes
        extra_terms = [
            -torch.sum(ln_data).item() / batch_size, 
            -torch.sum(ln_noise).item() / batch_size
        ]

        return curr_loss, extra_terms
            
    
