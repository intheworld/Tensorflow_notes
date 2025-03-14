import torch

def toy_example(x):
    y = x.sin()
    z = y.cos()
    return z

x = torch.randn(1000, device="cuda", requires_grad=True)
compiled_f = torch.compile(toy_example, backend='inductor')

output = compiled_f(x)