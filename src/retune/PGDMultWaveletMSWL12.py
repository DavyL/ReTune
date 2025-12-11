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
from torchvision.io import read_image
import os


from PGDWaveletL1 import PGD
import cyclospin
from SeparablePriors import ListSeparablePrior
from PGDWaveletMSWL12 import WeightedPGD
#
def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()




# %%
# Setup paths for data loading and results.
# --------------------------------------------
#
if __name__ == "__main__":
    savefigs_dir = 'figs/testMultL12'
    if not os.path.exists(savefigs_dir):
        os.makedirs(savefigs_dir)
    savefigs_dir += '/'
    BASE_DIR = Path(".")
    print(BASE_DIR)
    DATA_DIR = BASE_DIR / "measurements"
    CKPT_DIR = BASE_DIR / "ckpts"

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    img_size = 128 if torch.cuda.is_available() else 32
    C_channels = 3  # 3 for color images, 1 for gray-scale images
    probability_mask = 0.5  # probability to mask pixel



    # Initialize wavelet transform
    B_bands = 3
    J_scales = 4
    wav_names = ["db4", "db8", "coif6", "sym5", "haar"]
    wavelet_transform = cyclospin.MultipleWaveletsDictionary(wavelet_names=wav_names, levels=J_scales, device=device)
    M = len(wav_names)
    F_len = M*(J_scales+1)#Length of a wavelet dictionary decomposition over M wavelets
    #cyclo_wavelet_transform = CycloWaveletDictionary(wavelet_name="db1", levels=4)


    data_fid = dinv.optim.L2()
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

    stepsize_init = np.log(1e-2)
    lambda_init = np.log(1e-2)
    L_steps = 50
    R_restarts = 10

    x_init = obs
    u_init = lin_op.forward(x_init)
    
    init_sep_weights = torch.randn(F_len, device=device) ##Separable weights for \sum_j \lambda_j ||c_j||_2
    init_c_b_weights = torch.randn((C_channels, B_bands), device = device) ##Non Separable weights for  \sqrt{\sum_{c,b} w_{c,b} x_{c,b}^2}
    init_c_weights = init_c_b_weights[:,0]
    init_b_weights = init_c_b_weights[0,:]

    # %%
    L1_prior = ListSeparablePrior(dinv.optim.L1Prior(), separable_weights=init_sep_weights)
 
    with torch.no_grad():
        model_L1 = PGD(data_fid, L1_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    for name, param in model_L1.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    x_outputs_L1, u_outputs_L1, crit_L1 = model_L1(x_init, obs, u_init, physics, ret_crit = True)
    # %% Cross channel + cross bands \sqrt{\sum_{c,b} x_{c,b}^2}
    L12_cross_c_b_prior = ListSeparablePrior(dinv.optim.L12Prior(l2_axis=(1,4)), separable_weights=init_sep_weights)
    with torch.no_grad():
        model_L12_cross_c_b = PGD(data_fid, L12_cross_c_b_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    for name, param in model_L12_cross_c_b.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    with torch.no_grad():
        x_outputs_cross_c_b, u_outputs_cross_c_b, crit_cross_c_b = model_L12_cross_c_b(x_init, obs, u_init, physics, ret_crit = True)

    # %% Cross channel \sqrt{\sum_{c} x_{c}^2}
    L12_cross_c_prior = ListSeparablePrior(dinv.optim.L12Prior(l2_axis=(1,)), separable_weights=init_sep_weights)
    with torch.no_grad():
        model_L12_cross_c = PGD(data_fid, L12_cross_c_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    for name, param in model_L12_cross_c.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    with torch.no_grad():
        x_outputs_cross_c, u_outputs_cross_c, crit_cross_c = model_L12_cross_c(x_init, obs, u_init, physics, ret_crit = True)

    # %% Cross bands \sqrt{\sum_{c} x_{c}^2}
    L12_cross_b_prior = ListSeparablePrior(dinv.optim.L12Prior(l2_axis=(4,)), separable_weights=init_sep_weights)
    with torch.no_grad():
        model_L12_cross_b = PGD(data_fid, L12_cross_b_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    for name, param in model_L12_cross_b.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    with torch.no_grad():
        x_outputs_cross_b, u_outputs_cross_b, crit_cross_b = model_L12_cross_b(x_init, obs, u_init, physics, ret_crit = True)
        
    # %% Weights Cross channel + cross bands \sqrt{\sum_{c,b} w_{c,b} x_{c,b}^2}
    L12_weighted_cross_c_b_prior = ListSeparablePrior(dinv.optim.L12Prior(l2_axis=(1,4)), separable_weights=init_sep_weights)
    model_L12_weighted_cross_c_b_prior = WeightedPGD(weights_init=init_c_b_weights, weight_axis = (1,4), is_list=True, data_fidelity = data_fid, prior = L12_weighted_cross_c_b_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart = R_restarts, device = device)
    model_L12_weighted_cross_c_b_prior.weights = torch.nn.Parameter(model_L12_weighted_cross_c_b_prior.weights, requires_grad=True)

    for name, param in model_L12_weighted_cross_c_b_prior.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    with torch.no_grad():
        x_outputs_weighted_cross_c_b, u_outputs_weighted_cross_c_b, crit_weightedcross_c_b = model_L12_weighted_cross_c_b_prior(x_init, obs, u_init, physics, ret_crit = True)

    # %%  cross bands \sqrt{\sum_{b} w_{b} x_{b}^2}
    L12_weighted_cross_b_prior = ListSeparablePrior(dinv.optim.L12Prior(l2_axis=(4,)), separable_weights=init_sep_weights)
    model_L12_weighted_cross_b_prior = WeightedPGD(weights_init=init_b_weights, weight_axis = (4,), is_list=True, data_fidelity = data_fid, prior = L12_weighted_cross_b_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart = R_restarts, device = device)
    model_L12_weighted_cross_b_prior.weights = torch.nn.Parameter(model_L12_weighted_cross_b_prior.weights, requires_grad=True)

    for name, param in model_L12_weighted_cross_b_prior.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    with torch.no_grad():
        x_outputs_weighted_cross_b, u_outputs_weighted_cross_b, crit_weightedcross_b = model_L12_weighted_cross_b_prior(x_init, obs, u_init, physics, ret_crit = True)

    # %%  cross channels \sqrt{\sum_{b} w_{c} x_{c}^2}
    L12_weighted_cross_c_prior = ListSeparablePrior(dinv.optim.L12Prior(l2_axis=(1,)), separable_weights=init_sep_weights)
    model_L12_weighted_cross_c_prior = WeightedPGD(weights_init=init_c_weights, weight_axis = (1,), is_list=True, data_fidelity = data_fid, prior = L12_weighted_cross_c_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart = R_restarts, device = device)
    model_L12_weighted_cross_c_prior.weights = torch.nn.Parameter(model_L12_weighted_cross_c_prior.weights, requires_grad=True)

    for name, param in model_L12_weighted_cross_c_prior.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    with torch.no_grad():
        x_outputs_weighted_cross_c, u_outputs_weighted_cross_c, crit_weightedcross_c = model_L12_weighted_cross_c_prior(x_init, obs, u_init, physics, ret_crit = True)




    # %%
    fig, axs = plt.subplots(2,5, figsize=[15,5])
    axs[0,0].imshow(reshape_to_plot(truth[0]))
    axs[0,0].set_title(r'gt')

    axs[0,1].imshow(reshape_to_plot(obs[0]))
    axs[0,1].set_title(r'$y$')

    axs[0,2].imshow(reshape_to_plot(x_outputs_L1[0]))
    axs[0,2].set_title(r'$WL^1$')

    axs[0,3].imshow(reshape_to_plot(x_outputs_cross_c_b[0]))
    axs[0,3].set_title(r'$WL^{1,2}(B,C)$')

    axs[0,4].imshow(reshape_to_plot(x_outputs_cross_c[0]))
    axs[0,4].set_title(r'$WL^{1,2}(C)$')

    axs[1,0].imshow(reshape_to_plot(x_outputs_cross_b[0]))
    axs[1,0].set_title(r'$WL^{1,2}(B)$')

    axs[1,1].imshow(reshape_to_plot(x_outputs_weighted_cross_c_b[0]))
    axs[1,1].set_title(r'$WL^{1,2}_{W(C,B)}(B,C)$')

    axs[1,2].imshow(reshape_to_plot(x_outputs_weighted_cross_c[0]))
    axs[1,2].set_title(r'$WL^{1,2}_{W(C)}(C)$')

    axs[1,3].imshow(reshape_to_plot(x_outputs_weighted_cross_b[0]))
    axs[1,3].set_title(r'$WL^{1,2}_{W(B)}(B)$')

    for ax1 in axs:
        for ax in ax1:
            ax.axis('off') 
    #fig.savefig(savefigs_dir + 'gt_obs_reconstructions.pdf', bbox_inches ='tight', transparent=True)
    
    #fig.axis('off')
    fig, axs = plt.subplots(1,7, figsize=[15,5])
    axs[0].plot(crit_L1)
    axs[0].grid()
    axs[0].set_yscale('log')
    axs[0].set_title(r'$WL^1$')

    axs[1].plot(crit_cross_c_b)
    axs[1].grid()
    axs[1].set_yscale('log')
    axs[1].set_title(r'$WL^{1,2}(B,C)$')

    axs[2].plot(crit_cross_c)
    axs[2].grid()
    axs[2].set_yscale('log')
    axs[2].set_title(r'$WL^{1,2}(C)$')

    axs[3].plot(crit_cross_b)
    axs[3].grid()
    axs[3].set_yscale('log')
    axs[3].set_title(r'$WL^{1,2}(B)$')

    axs[4].plot(crit_weightedcross_c_b)
    axs[4].grid()
    axs[4].set_yscale('log')
    axs[4].set_title(r'$WL^{1,2}_{W(C,B)}(B,C)$')

    axs[5].plot(crit_weightedcross_c)
    axs[5].grid()
    axs[5].set_yscale('log')
    axs[5].set_title(r'$WL^{1,2}_{W(C)}(C)$')

    axs[6].plot(crit_weightedcross_b)
    axs[6].grid()
    axs[6].set_yscale('log')
    axs[6].set_title(r'$WL^{1,2}_{W(B)}(B)$')




# %%
