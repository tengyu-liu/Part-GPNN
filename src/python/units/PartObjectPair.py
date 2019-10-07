import torch
import torch.nn as nn

class PartObjectPair(nn.Module):
    def __init__(self, update_type):
        super(PartObjectPair, self).__init__()
        self.update_type = update_type
        if self.update_type == 'mult':
            self.weights = [[nn.Parameter(torch.Tensor(1, 1)) for _ in range(94)] for _ in range(94)]
        elif self.update_type == 'concat':
            self.weights = [[nn.Parameter(torch.Tensor(1, 512)) for _ in range(94)] for _ in range(94)]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_features, part_cls, obj_cls):
        if self.update_type == 'mult':
            feature *= self.sigmoid(self.weights[part_cls][obj_cls])
        elif self.update_type == 'concat':
            feature = torch.concat([input_features, self.weights[part_cls][obj_cls]])
        return feature


