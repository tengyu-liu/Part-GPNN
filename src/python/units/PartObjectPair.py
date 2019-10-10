import torch
import torch.nn as nn

class PartObjectPair(nn.Module):
    def __init__(self, update_type):
        super(PartObjectPair, self).__init__()
        self.update_type = update_type
        if self.update_type == 'mult':
            self.weights = [[nn.Parameter(torch.Tensor(1, 1)).cuda() for _ in range(95)] for _ in range(95)]
        elif self.update_type == 'concat':
            self.weights = [[nn.Parameter(torch.Tensor(1, 512)).cuda() for _ in range(95)] for _ in range(95)]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_features, part_cls, obj_cls):
        if self.update_type == 'mult':
            feature = input_features * self.sigmoid(self.weights[part_cls][obj_cls])
        elif self.update_type == 'concat':
            feature = torch.concat([input_features, self.weights[part_cls][obj_cls]])
        return feature


