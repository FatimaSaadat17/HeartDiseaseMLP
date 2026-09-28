import torch.nn as nn
import torch
# initialize and define base NN model
class MLP(nn.Module):
    def __init__(self, l1_param, l2_param=10):
        # define layers as attributes of the self object instance
        super().__init__()
        l1 = nn.Linear(l1_param,l2_param)
        a1 = nn.ReLU()
        l2 = nn.Linear(l2_param, 1)
        s1 = nn.Sigmoid()
        l = [l1, a1, l2, s1]
        
        # put all defined layers in nn.ModuleList object
        self.module_list = nn.ModuleList(l)

    def forward(self, x):
        # specify how layers are used in the forward pass of NN
        for f in self.module_list:
            x = f(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)
        return (pred >= 0.5).float() # return predicted class label 0/1 for the sample

    def weight_initialization(self):
        for layer in self.module_list:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)