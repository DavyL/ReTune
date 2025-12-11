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
from torchvision.io import read_image
import os

from PGDWaveletL1 import PGD
from cyclospin import IdentityDictionary

#
def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()


class PGD_Drunet(PGD):
    def __init__(self,sigma_denoise_init = -5.0, *args, **kwargs):
        super(PGD_Drunet,self).__init__(*args, **kwargs)
        self.dict_fwd = lambda x:x
        self.dict_adj = lambda x:x
        self.sigma_den_list = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(sigma_denoise_init), requires_grad=True) for _ in range(self.L_layers)])    

    def L_step(self, x, y, u, physics, it):
        v = u -  self.estepsize * self.dict_fwd(physics.A_adjoint(physics.A(self.dict_adj(u)) - y))
        u = self.prior.prox(v, ths=1.0, sigma_denoiser=torch.exp(self.sigma_den_list[it]),gamma= self.elambda * self.estepsize)
        x_out = self.dict_adj(u)
        return x_out, u
    
    def criterion(self, y, u, physics):
        df  = self.data_fidelity.fn(self.dict_adj(u), y, physics)
        #reg = self.elambda*self.prior.fn(u)
        return df#+reg
    
class PGD_Drunet_layerindep(PGD_Drunet):
    def __init__(self,*args, **kwargs):
        super(PGD_Drunet_layerindep,self).__init__(*args, **kwargs)

    def L_step(self, x, y, u, physics, it):
        v = u -  self.estepsize * self.dict_fwd(physics.A_adjoint(physics.A(self.dict_adj(u)) - y))
        u = self.prior.prox(v, ths=1.0, sigma_denoiser=torch.exp(self.sigma_den_list[0]),gamma= self.elambda * self.estepsize)
        x_out = self.dict_adj(u)
        return x_out, u
    
    def criterion(self, y, u, physics):
        df  = self.data_fidelity.fn(self.dict_adj(u), y, physics)
        #reg = self.elambda*self.prior.fn(u)
        return df#+reg
    
class GSPnP(dinv.optim.prior.RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)
def remove_params_from_model(model):
    def get_parent_module_and_attr(model, name):
        """
        Get the parent module and the final attribute name for a given nested parameter.
        """
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:  # Go up to the second last part
            parent = getattr(parent, p)
        return parent, parts[-1]  # Return the parent module and the attribute name

    with torch.no_grad():  # Ensure no gradient tracking
        for name, param in list(model.named_parameters()):  # Convert to list to safely modify
            #print(f'name {name}, param {param}, param.data {param.data}')            
            # Get parent module and attribute name
            parent_module, attr_name = get_parent_module_and_attr(model, name)            
            # Extract tensor value to preserve data
            tensor_val = param.data            
            # Remove parameter from parent module
            delattr(parent_module, attr_name)            
            # Register as a buffer instead (so it won't be a parameter anymore)
            parent_module.register_buffer(attr_name, tensor_val)

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

    C_channels = 3  # 3 for color images, 1 for gray-scale images
    probability_mask = 0.5  # probability to mask pixel



    data_fid = dinv.optim.L2()
    lin_op = IdentityDictionary()
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

    stepsize_init = np.log(1e-2)
    lambda_init = np.log(1e0)
    L_steps = 10
    R_restarts = 10
    x_init = obs
    u_init = obs

    # %%
    BASE_DIR = Path(".")
    print(BASE_DIR)
    DATA_DIR = BASE_DIR / "measurements"
    CKPT_DIR = BASE_DIR / "ckpts"

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    #torch.manual_seed(seed)

#    print(f'device is {device}')
#    operation = 'deblurring_with_large_blur'    
#    operation = 'deblurring_with_large_noise'    
#    train_dataset_name = "div2k"
#
#    measurement_dir = DATA_DIR / train_dataset_name / operation
#    psf_size = 25
#    filter_rgb = torch.zeros((1, 3, psf_size, psf_size), device=device)
#    filter_rgb[:, 0, :, psf_size // 2 : psf_size // 2 + 1] = 1.0 / psf_size
#    filter_rgb[:, 1, psf_size // 2 : psf_size // 2 + 1, :] = 1.0 / psf_size
#    filter_rgb[:, 2, ...] = (
#        torch.diag(torch.ones(psf_size, device=device)) / psf_size
#    )
#    physics = dinv.physics.Blur(filter_rgb, device = device, noise_model=dinv.physics.GaussianNoise(0.1), padding='reflect')
#    physics.load_state_dict(torch.load(measurement_dir/'physics0.pt', weights_only=False))


    # %%
    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))
    with torch.no_grad():    
        model_PnP = PGD_Drunet(-5.0, data_fid, Drunet_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device=device)
        x_outputs_PnP, u_outputs_PnP, crit_PnP = model_PnP(x_init, obs, u_init, physics, ret_crit = True)

    fig, axs = plt.subplots(1,3, figsize=[15,5])
    axs[0].imshow(reshape_to_plot(truth[0]))
    axs[1].imshow(reshape_to_plot(obs[0]))
    axs[2].imshow(reshape_to_plot(x_outputs_PnP[0]))

    

    # %%
    #sig_den_list = [-5.0,-2.0,-1.5,-1.4,-1.3,-1.2,-1.0,0.0,5.0]
    sig_den_list = [-2.]
    fig, axs = plt.subplots(1,2+len(sig_den_list), figsize=[15,5])
    axs[0].imshow(reshape_to_plot(truth[0]))
    axs[1].imshow(reshape_to_plot(obs[0]))
    obs = physics(image)
    x_init = obs
    u_init = obs
    L_steps = 10
    R_restarts = 10
    stepsize_init = np.log(1.95)
    #stepsize_init = np.log(0.25)
    lambda_init = np.log(1.0/10) 
    sigma_denoise_init = -1.3
    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))

    for idx, sigma_denoise_init in enumerate(sig_den_list):
        model_Drunet =  PGD_Drunet_layerindep(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, sigma_denoise_init = sigma_denoise_init, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart =R_restarts, zero_init=False, device = device)
        model = model_Drunet
        with torch.no_grad():    
            obs = physics(image)
            x_init = obs
            u_init = obs
            #model_PnP = PGD_Drunet(data_fid, Drunet_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)
            x_outputs_PnP, u_outputs_PnP, crit_PnP = model(x_init, obs, u_init, physics, ret_crit = True)
        axs[2+idx].imshow(reshape_to_plot(x_outputs_PnP[0]))
        axs[2+idx].set_title(sigma_denoise_init)
        axs[2+idx].set_axis_off()
    axs[0].set_axis_off() 
    axs[1].set_axis_off() 
    #fig.savefig('figs/test_sigma_init_pnp_deblurring.pdf')




# %%

    GSDrunet_prior = GSPnP(denoiser=dinv.models.GSDRUNet(pretrained="download", device=device))
    with torch.no_grad():    
        model_GSDrunet = PGD_Drunet(-5.0, data_fid, GSDrunet_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)
        x_outputs_GSDrunet, u_outputs_GSDrunet, crit_GSDrunet = model_GSDrunet(x_init, obs, u_init, physics, ret_crit = True)

# %%
    fig, axs = plt.subplots(1,3, figsize=[15,5])
    axs[0].imshow(reshape_to_plot(truth[0]))
    axs[1].imshow(reshape_to_plot(obs[0]))
    axs[2].imshow(reshape_to_plot(x_outputs_GSDrunet[0]))

# %%
