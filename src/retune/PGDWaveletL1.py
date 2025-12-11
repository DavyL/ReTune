r"""
Training a reconstruction network.
====================================================================================================

This example shows how to train a simple reconstruction network for an image
inpainting inverse problem.

"""
# %%
import deepinv as dinv
import torch
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
from ptwt.conv_transform_2 import *
import cyclospin
from torchvision.io import read_image
from SeparablePriors import  ListSeparablePrior
#from deepinv.utils.demo import get_data_home
#ORIGINAL_DATA_DIR = get_data_home()


#
# %%
def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()

  
class PGD(torch.nn.Module):
    def __init__(self, data_fidelity,  prior, linear_operator, stepsize, lambd, L_layers, R_restart, device, zero_init=False, learn_stepsize = False,*args, **kwargs):
        super(PGD, self).__init__(*args, **kwargs)

        self.data_fidelity = data_fidelity
        self.prior = prior
        self.dict_adj = linear_operator.adjoint
        self.dict_fwd = linear_operator.forward

        self.lambd = torch.nn.Parameter(torch.tensor(lambd, device=device), requires_grad = True)
        self.elambda    = torch.exp(self.lambd)
        self.stepsize = torch.tensor(stepsize, device=device)
        if learn_stepsize:
            self.stepsize   = torch.nn.Parameter(torch.tensor(stepsize, device = device), requires_grad = True)
        self.estepsize  = torch.exp(self.stepsize)

        self.L_layers   = L_layers
        self.R_restart  = R_restart

        self.zero_init = zero_init

    def forward(self, x, y, u, physics, ret_crit = False):
        self.elambda    = torch.exp(self.lambd)
        self.estepsize  = torch.exp(self.stepsize)
        crit_list = []
        if self.zero_init is True:
            u = u - u
        with torch.no_grad():
            for r in range(self.R_restart):
                x, u = self.R_step(x, y, u, physics, crit_list)

        self.elambda    = torch.exp(self.lambd)
        self.estepsize  = torch.exp(self.stepsize)

        x, u = self.R_step(x, y, u, physics, crit_list)

        if ret_crit:
            return x, u, crit_list
        return x,u
    
    def R_step(self, x, y, u, physics, crit_list):
        for l in range(self.L_layers):
            #print(f'r,l {r,l}')
            x, u = self.L_step(x, y, u, physics,l)
            crit = self.criterion(y,u, physics).cpu().detach().numpy()
            crit_list.append(crit)
        return x, u
    
    def L_step(self, x, y, u, physics, it=None):
        v = u -  self.estepsize * self.dict_fwd(physics.A_adjoint(physics.A(self.dict_adj(u)) - y))
        u = self.prior.prox(v, ths=1.0,gamma= self.elambda * self.estepsize)
        x_out = self.dict_adj(u)
        return x_out, u

    
    def criterion(self, y, u, physics):
        df  = self.data_fidelity.fn(self.dict_adj(u), y, physics)
        reg = self.elambda*self.prior.fn(u)
        return df+reg
    



# %%
# Setup paths for data loading and results.
# --------------------------------------------
#
if __name__ == "__main__":
    BASE_DIR = Path(".")
    print(BASE_DIR)
    DATA_DIR = BASE_DIR / "measurements"
    CKPT_DIR = BASE_DIR / "ckpts"

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    #img_size = 128 if torch.cuda.is_available() else 32
    n_channels = 3  # 3 for color images, 1 for gray-scale images
    probability_mask = 0.5  # probability to mask pixel



    WaveletDictionary = cyclospin.WaveletDictionaryDI 
    #CycloWaveletDictionary = cyclospin.CycloWaveletDictionary 
    # Initialize wavelet transform
    J_scales = 4
    wavelet_transform = WaveletDictionary(wavelet_name="db4", levels=J_scales, device=device)
    #cyclo_wavelet_transform = CycloWaveletDictionary(wavelet_name="db1", levels=4)


    data_fid = dinv.optim.L2()
    prior = ListSeparablePrior(dinv.optim.L1Prior(), separable_weights=torch.zeros(J_scales+1, device=device))
    lin_op = wavelet_transform
    #prior = dinv.optim.prior.WaveletPrior(level=3, wv="db8", p=1, device=device)
    #prior = dinv.optim.prior.TVPrior(n_it_max=20)

    image_path = "../../ressources/393035.jpg"
    image_file = read_image(image_path)
    image = image_file.unsqueeze(0)[...,:256,:256].to(torch.float32).to(device)/255
    print(f'image has shape {image.shape}')
        # Generate inpainting operator
    #physics = dinv.physics.Denoising( device=device)
    physics = dinv.physics.Inpainting(
        tensor_size=image.shape[1:], mask=probability_mask, device=device
    )
    #physics = dinv.physics.Blur(
    #    dinv.physics.blur.gaussian_blur(sigma=(2.0, 0.5), angle=0), device=device, padding='reflect'
    #)

    truth = image
    obs = physics(image)

    stepsize_init = -3.0
    lambda_init = -3.0
    L_steps = 50
    R_restarts = 10

    x_init =obs
    u_init = lin_op.forward(x_init)
    model = PGD(data_fid, prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    for name, param in model.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    x_outputs, u_outputs, crit = model(x_init, obs, u_init, physics, ret_crit = True)

    # %%
    fig, axs = plt.subplots(1,3, figsize=[15,5])
    axs[0].imshow(reshape_to_plot(truth[0]))
    axs[1].imshow(reshape_to_plot(obs[0]))
    axs[2].imshow(reshape_to_plot(x_outputs[0]))
    #fig.axis('off')
    plt.figure()
    plt.plot(crit)
    plt.grid()
    plt.yscale('log')
    print(crit[-1])
    # %%

    lambda_list = [np.log(10)*j for j in np.linspace(-2,1,8)]

    all_crit_list = []
    all_x_list = []
    all_u_list = []
    for cur_lambda in lambda_list:
        model = PGD(data_fid, prior, lin_op, stepsize_init, cur_lambda, L_steps, R_restarts, device)
        x_outputs, u_outputs, crit = model(x_init, obs, u_init, physics, ret_crit = True)
        all_x_list.append(x_outputs)
        all_u_list.append(u_outputs)    
        all_crit_list.append(crit)

    # %%
    plt.figure()
    for lambda_idx, cur_lambda in enumerate(lambda_list):
        plt.plot(all_crit_list[lambda_idx], label=f'{np.exp(cur_lambda)}')
    plt.legend()
    #plt.yscale('log')
    #plt.savefig('figs/first_plots/crit_fn_of_lambda.pdf')

    max_j = 8
    fig, axs = plt.subplots(1+4, 2, figsize=(5,15))
    axs[0,0].imshow(reshape_to_plot(truth[0]))
    axs[0,1].imshow(reshape_to_plot(obs[0]))
    axs[0,0].axis('off')
    axs[0,1].axis('off')
    for j in range(4):
        axs[1+j,0].imshow(reshape_to_plot(all_x_list[2*j][0]))# %%
        axs[1+j,1].imshow(reshape_to_plot(all_x_list[2*j + 1][0]))# %%
        axs[1+j,0].axis('off')
        axs[1+j,1].axis('off')
    #plt.savefig('figs/first_plots/imgs_fn_of_lambda.pdf')


# %%
