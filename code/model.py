# This part is establishing AlphaNet-v1 model with self defined layers
# Input: torch.Size([1000, 1, 9, 30])
# Output: torch.Size([1000])
from torch import nn
from self_defined_layers import *


# Define a custom layer from a given function
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# Model
class AlphaNet_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts_corr = Lambda(ts_corr)
        self.ts_cov = Lambda(ts_cov)
        self.ts_stddev = Lambda(ts_stddev)
        self.ts_zscore = Lambda(ts_zscore)
        self.ts_return = Lambda(ts_return)
        self.ts_decaylinear = Lambda(ts_decaylinear)
        self.ts_mean_extract = Lambda(ts_mean_extract)
        self.BN = nn.BatchNorm2d(1, affine=True, track_running_stats=True)
        self.ts_mean_pool = Lambda(ts_mean_pool)
        self.ts_max = Lambda(ts_max)
        self.ts_min = Lambda(ts_min)
        self.Flatten = nn.Flatten(1, 3)
        self.linear1 = nn.Linear(702, 30)
        self.linear2 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.weights = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, xb):
        # extract layer
        # xb = xb.type(torch.float64)
        xb_1 = self.BN(self.ts_corr(xb))  # N, 1, 36, 3  ; BN: 2 learnable parameters
        xb_2 = self.BN(self.ts_cov(xb))  # N, 1, 36, 3
        xb_3 = self.BN(self.ts_stddev(xb))  # N, 1, 9, 3
        xb_4 = self.BN(self.ts_zscore(xb))  # N, 1, 9, 3
        xb_5 = self.BN(self.ts_return(xb))  # N, 1, 9, 3
        xb_6 = self.BN(self.ts_decaylinear(xb))  # N, 1, 9, 3
        xb_7 = self.BN(self.ts_mean_extract(xb))  # N, 1, 9, 3

        # pool layer
        xb = torch.cat([xb_1, xb_2, xb_3, xb_4, xb_5, xb_6, xb_7], 2)  # N, 1, 117, 3
        xb_8 = self.BN(self.ts_mean_pool(xb))  # N, 1, 117, 1
        xb_9 = self.BN(self.ts_max(xb))  # N, 1, 117, 1
        xb_10 = self.BN(self.ts_min(xb))  # N, 1, 117, 1

        # flatten layer
        xb = torch.cat([self.Flatten(xb), self.Flatten(xb_8),
                        self.Flatten(xb_9), self.Flatten(xb_10)], 1)  # N, 702

        # fully connected layer & hidden layer
        xb = self.linear1(xb)  # N, 30 ; linear1: 2 learnable parameters
        xb = self.dropout(xb)
        xb = f.relu(xb)

        # output layer
        xb = self.linear2(xb)  # N, 1 ; linear2: 2 learnable parameters
        xb = xb * self.weights + self.bias  # N, 1 ; 2 learnable parameters
        return xb.view(-1)

