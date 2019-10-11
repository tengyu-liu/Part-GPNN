import torch
import torch.nn as nn

class PartObjectPair(nn.Module):
    def __init__(self, update_type, suppress_hh=False):
        super(PartObjectPair, self).__init__()
        self.update_type = update_type
        if self.update_type == 'mult':
            self.weights = [[nn.Parameter(torch.Tensor(np.random.normal(scale=0.01, size=[1,1]))).cuda() for _ in range(95)] for _ in range(95)]
        elif self.update_type == 'concat':
            self.weights = [[nn.Parameter(torch.Tensor(np.random.normal(scale=0.01, size=[1,512]))).cuda() for _ in range(95)] for _ in range(95)]
            self.dense = torch.nn.Linear(1512, 1000)
            # self.relu = torch.nn.ReLU()
        self.suppress_hh = suppress_hh
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_features, i_cls, j_cls, i_human, j_human):
        if self.update_type == 'mult':
            feature = input_features * self.sigmoid(self.weights[i_cls][j_cls])
        elif self.update_type == 'concat':
            feature = torch.cat([input_features, self.weights[i_cls][j_cls]], 1)
            feature = self.relu(self.dense(feature))
        if self.suppress_hh:
            if i_human == -1 and i_human == j_human:
                    feature *= 0
            if i_human != -1 and j_human != -1 and i_human != j_human:
                feature *= 0
        return feature


