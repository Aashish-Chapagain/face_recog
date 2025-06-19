from torch import nn
import torch 



layer = nn.Linear(100, 10)


input = torch.randn(1, 100)  # Example input tensor with batch size 1 and 100 features

output = layer(input)



print(output.shape)