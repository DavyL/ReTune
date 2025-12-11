import deepinv as dinv
import torch
class LogPrior(dinv.optim.Prior):

    def __init__(self, prior):
        super(dinv.optim.Prior).__init__()
        self.prior = prior

    def fn(self, x, *args, **kwargs):
        return self.prior.fn(x, *args, **kwargs)

    def grad(self, x, *args, **kwargs):
        return self.prior.grad(x, *args, **kwargs)

    def prox(self, x, ths=1.0, gamma=1.0, *args, **kwargs):
        return self.prior.prox(x, torch.exp(ths), torch.exp(gamma), *args, **kwargs)

new_prior = LogPrior(dinv.optim.L12Prior())
p = new_prior.prox(torch.randn((10,3,20,20,5)), ths = -torch.tensor(10.0), gamma = -torch.tensor(5.0))
print(p)