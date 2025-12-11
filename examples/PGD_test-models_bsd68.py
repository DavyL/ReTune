r"""
Training a reconstruction network.
====================================================================================================

This example shows how to train a simple reconstruction network for an image
inpainting inverse problem.

"""
# %%
import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset
import matplotlib.pylab as plt
import numpy as np
from ptwt.conv_transform_2 import *
import cyclospin
from cyclospin import WaveletDictionary, IdentityDictionary
from deepinv.optim import data_fidelity
#from deepinv.utils.demo import get_data_home
#ORIGINAL_DATA_DIR = get_data_home()
import os
from PGDWaveletL1 import PGD
from PGDWaveletMSWL12 import L12, SeparablePrior, ListSeparablePrior, WeightedPGD
from PGDDenoisingPrior import PGD_Drunet, remove_params_from_model, PGD_Drunet_layerindep

def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()


# %%

if __name__ == "__main__":

    ## Function to test a model once all variables have been set by running cells below
    def test_model_and_make_figs(model):
        loss_list_no_reg    = []
        psnr_list_no_reg    = []
        loss_list       = []
        psnr_list       = []
        crit_list       = [] ## For test data
        loss_list_single_step  = []
        psnr_list_single_step  = []
        loss_list_r_step       = []
        psnr_list_r_step       = []
        crit_list_r_step = []

        with torch.no_grad():
            #On train data
            for batch_idx, (truths, observations) in enumerate(train_dataloader):
                b_size = observations.shape[0]
                truths = truths.to(device)
                observations = torch.tensor(observations, device = device)
                loss_no_reg = loss_fn(observations, truths).cpu().detach().numpy()
                psnr_no_reg = psnr_fn(observations, truths).cpu().detach().numpy()
                psnr_list_no_reg.extend(psnr_no_reg.tolist())
                loss_list_no_reg.extend(loss_no_reg.tolist())    
                # Forward pass
                x_init = observations
                u_init = model.dict_fwd(x_init)
                x_outputs, u_outputs = model(x_init, observations, u_init, physics)
                loss = loss_fn(x_outputs, truths).cpu().detach().numpy()
                psnr = psnr_fn(x_outputs, truths).cpu().detach().numpy()
                psnr_list.extend(psnr.tolist())
                loss_list.extend(loss.tolist())

                print(f'loss list {loss_list}, psnr list {psnr_list}')
                single_step_x_outputs, single_step_u_outputs = model.L_step(x_init, observations, u_init, physics, it=0) ## Modifier L step en K step/enlever it=0
                single_step_loss = loss_fn(single_step_x_outputs, truths).cpu().detach().numpy()
                single_step_psnr = psnr_fn(single_step_x_outputs, truths).cpu().detach().numpy()
                psnr_list_single_step.extend(single_step_psnr.tolist())
                loss_list_single_step.extend(single_step_loss.tolist())

                r_step_x_outputs, r_step_u_outputs = model.R_step(x_init, observations, u_init, physics, crit_list = []) ## Modifier L step en K step/enlever it=0
                r_step_loss = loss_fn(r_step_x_outputs, truths).cpu().detach().numpy()
                r_step_psnr = psnr_fn(r_step_x_outputs, truths).cpu().detach().numpy()
                psnr_list_r_step.extend(r_step_psnr.tolist())
                loss_list_r_step.extend(r_step_loss.tolist())

            #test_truth = test_truth.to(device)
            #test_observations = torch.tensor(test_observation, device = device)
                
            # On test data
            b_test_size = test_observation.shape[0]
            x_init = test_observation
            u_init = model.dict_fwd(x_init)
            test_x_outputs,             test_u_outputs, crit_list = model(x_init, test_observation, u_init, physics, ret_crit=True)
            test_single_step_x_outputs, test_single_step_u_outputs = model.L_step(x_init, test_observation, u_init, physics, it=0)
            test_r_step_x_outputs,      test_r_step_u_outputs = model.R_step(x_init, test_observation, u_init, physics, crit_list_r_step)
            #loss_test = loss_fn(test_x_outputs, test_truth)/b_test_size
            #psnr_test = psnr_fn(test_x_outputs, test_truth)/b_test_size

            torch.cuda.empty_cache()
            #On test data

        ## Make figures

        for i in range(3):
            fig, axs = plt.subplots(1,5, figsize=(20,15))

            axs[0].set_title(r'gt')
            axs[1].set_title(r'$y$' +  f'\t{round(psnr_fn(test_observation, test_truth)[i].item(),2)}db')
            axs[2].set_title(r'$\varphi_1(y)$' +  f'\t{round(psnr_fn(test_single_step_x_outputs, test_truth)[i].item(),2)}db')
            axs[3].set_title(r'$\Phi_{K}^{T = 1}(y)$' +  f'\t{round(psnr_fn(test_r_step_x_outputs, test_truth)[i].item(),2)}db')
            axs[4].set_title(r'$\Phi_{K}^{T}(y)$' +  f'\t{round(psnr_fn(test_x_outputs, test_truth)[i].item(),2)}db')

            axs[0].imshow(reshape_to_plot(test_truth[i]))
            axs[1].imshow(reshape_to_plot(test_observation[i]))
            axs[2].imshow(reshape_to_plot(test_single_step_x_outputs[i]))
            axs[3].imshow(reshape_to_plot(test_r_step_x_outputs[i]))
            axs[4].imshow(reshape_to_plot(test_x_outputs[i]))

            for ax in axs:
                ax.axis('off') 
            fig.savefig(f'{testmodels_figs_dir}im{i}_gt_obs_reconstructions-{model_name}.pdf', bbox_inches ='tight', transparent=True)
        
        ##Plot criterion
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(crit_list, label=r'$\Phi_{K}^T$')
        ax.plot(crit_list_r_step, label=r'$\Phi_{K}^{T=1}$')
        ax.set_yscale('log')
        ax.grid()
        ax.set_title(r'$F(x_k^t) + G_\theta(x_k^t)$')
        ax.set_xlabel(r'$tT + k$')
        ax.legend()
        fig.savefig(f'{testmodels_figs_dir}criterions-{model_name}.pdf', bbox_inches ='tight', transparent=True)
        
        Nbins = 64
        fig, axs = plt.subplots(4,1, figsize=(5,20), sharex = True)
        axs[0].hist(psnr_list_no_reg, bins=Nbins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        axs[0].set_xlabel('PSNR Value', fontsize=12)
        axs[0].set_ylabel('Frequency', fontsize=12)
        axs[0].set_title('PSNR of observations', fontsize=14)
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)

        axs[1].hist(psnr_list_single_step, bins=Nbins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        axs[1].set_xlabel('PSNR Value', fontsize=12)
        axs[1].set_ylabel('Frequency', fontsize=12)
        axs[1].set_title('PSNR single layer', fontsize=14)
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)

        axs[2].hist(psnr_list_r_step, bins=Nbins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        axs[2].set_xlabel('PSNR Value', fontsize=12)
        axs[2].set_ylabel('Frequency', fontsize=12)
        axs[2].set_title('PSNR no restart', fontsize=14)
        axs[2].grid(axis='y', linestyle='--', alpha=0.7)

        axs[3].hist(psnr_list, bins=Nbins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        axs[3].set_xlabel('PSNR Value', fontsize=12)
        axs[3].set_ylabel('Frequency', fontsize=12)
        axs[3].set_title('PSNR restart', fontsize=14)
        axs[3].grid(axis='y', linestyle='--', alpha=0.7)


        for ax in axs:
            ax.grid() 
        fig.savefig(f'{testmodels_figs_dir}criterions_and_parameters_through_epochs-{model_name}.pdf', bbox_inches ='tight', transparent=True)



    project = 'restart_denoising'

    BATCH_size = 32
    operation = "denoising"
    train_dataset_name = "CBSD68"
    test_dataset_name = "set3c"
    C_channels = 3 #number of channels

    #%%
    BASE_DIR = Path(".")
    print(BASE_DIR)
    DATA_DIR = BASE_DIR / "measurements"
    CKPT_DIR = BASE_DIR / "ckpts"

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f'device is {device}')

    measurement_dir = DATA_DIR / train_dataset_name / operation
    img_size = 256 if torch.cuda.is_available() else 32
    n_channels = 3  # 3 for color images, 1 for gray-scale images
    probability_mask = 0.5  # probability to mask pixel
    #physics = dinv.physics.Inpainting(
    #    tensor_size=(n_channels, img_size, img_size), mask=probability_mask, device=device
    #)
    physics = dinv.physics.Denoising()
    physics.load_state_dict(torch.load(measurement_dir/'physics0.pt', weights_only=False))
    num_workers = 4 if torch.cuda.is_available() else 0

    #deepinv_datasets_path = 'measurements/div2k/inpainting/dinv_dataset0.h5'
    deepinv_datasets_path = 'measurements/CBSD68/denoising/dinv_dataset0.h5'

    train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
    test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)
    
    train_batch_size = BATCH_size if torch.cuda.is_available() else 1
    test_batch_size  = 4 if torch.cuda.is_available() else 1

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
    )
    iter_loader = iter(test_dataloader)    #Load a fixed image for test
    test_truth, test_observation = next(iter_loader)
    test_truth = test_truth.to(device) 
    test_observation = test_observation.to(device)

    # choose training losses
    loss_fn = dinv.loss.SupLoss(metric=dinv.metric.MSE())
    psnr_fn = dinv.metric.PSNR()


    C = 3 #number of channels

    data_fid = dinv.optim.L2()


    fig, axs = plt.subplots(1,2)
    axs[0].imshow(reshape_to_plot(test_truth[0]))
    axs[1].imshow(reshape_to_plot(test_observation[0]))
    axs[1].set_title(f'{round(psnr_fn(test_observation, test_truth)[0].sum().item(),2)}db')
 
    # %%
    ######## ONDELETTES ##########
    # Initialize wavelet transform
    B_bands = 3 #orientations
    J_scales = 4
    wav_name = "db4"
    lin_op = cyclospin.WaveletDictionary(wavelet_name=wav_name, levels=J_scales, device=device)
    stepsize_init = 0.5
    lambda_init = np.log(1.0/10)
    sigma_denoise_init = -1.0
    init_sep_weights = torch.zeros(J_scales+1, device=device) ##Separable weights for \sum_j \lambda_j ||c_j||_2
    init_c_b_weights = torch.zeros((C_channels, B_bands), device = device) ##Non Separable weights for  \sqrt{\sum_{c,b} w_{c,b} x_{c,b}^2}
    init_c_weights = init_c_b_weights[:,0]
    init_b_weights = init_c_b_weights[0,:]

    L_steps = 10
    R_restarts = 25
    epochs =   100
    # %% L1 model with learned lambda and steps
    group = 'learn_wavelet_reg' ##Fix name

    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L1'
    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L1_prior = ListSeparablePrior(dinv.optim.L1Prior(), separable_weights=init_sep_weights)
    model_L1 = PGD(data_fid, L1_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    model_L1.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L1
    test_model_and_make_figs(model)
    # %% L1 model with learned lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'WL1'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    WL1_prior = ListSeparablePrior(dinv.optim.L1Prior(), separable_weights=init_sep_weights)
    model_WL1 = PGD(data_fid, WL1_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)
    WL1_prior.separable_weights = torch.nn.Parameter(WL1_prior.separable_weights, requires_grad=True)

    model_WL1.load_state_dict(torch.load(model_path, weights_only=False))

    model = model_WL1
    test_model_and_make_figs(model)

    # %% L12 model (l2 on channels and bands) with learned lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12_cross_c_b'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L12_cross_c_b'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L12_cross_c_b_prior = ListSeparablePrior(L12(l2_axis=(1,4)), separable_weights=init_sep_weights)
    model_L12_cross_c_b = PGD(data_fid, L12_cross_c_b_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)
    L12_cross_c_b_prior.separable_weights = torch.nn.Parameter(L12_cross_c_b_prior.separable_weights, requires_grad=True)
 
    model_L12_cross_c_b.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L12_cross_c_b
    test_model_and_make_figs(model)
    # %% L12 model (l2 on channels) with learned lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12_cross_c'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L12_cross_c'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L12_cross_c_prior = ListSeparablePrior(L12(l2_axis=(1,)), separable_weights=init_sep_weights)
    model_L12_cross_c = PGD(data_fid, L12_cross_c_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)
    L12_cross_c_prior.separable_weights = torch.nn.Parameter(L12_cross_c_prior.separable_weights, requires_grad=True)
 
    model_L12_cross_c.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L12_cross_c
    test_model_and_make_figs(model)
    # %% L12 model (l2 on bands) with learned lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12_cross_b'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L12_cross_b'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L12_cross_b_prior = ListSeparablePrior(L12(l2_axis=(4,)), separable_weights=init_sep_weights)
    model_L12_cross_b = PGD(data_fid, L12_cross_b_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)
    L12_cross_b_prior.separable_weights = torch.nn.Parameter(L12_cross_b_prior.separable_weights, requires_grad=True)
 
    model_L12_cross_b.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L12_cross_b
    test_model_and_make_figs(model)
    # %% Weighted L12 model (l2 on bands and channels) with learned w_{c,b}, lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12_cross_c_b'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L12_weighted_cross_c_b'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L12_weighted_cross_c_b_prior = ListSeparablePrior(L12(l2_axis=(1,4)), separable_weights=init_sep_weights)
    model_L12_weighted_cross_c_b_prior = WeightedPGD(weights_init=init_c_b_weights, weight_axis = (1,4), is_list=True, data_fidelity = data_fid, prior = L12_weighted_cross_c_b_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart = R_restarts, device = device)
    model_L12_weighted_cross_c_b_prior.weights = torch.nn.Parameter(model_L12_weighted_cross_c_b_prior.weights, requires_grad=True)
    L12_weighted_cross_c_b_prior.separable_weights = torch.nn.Parameter(L12_weighted_cross_c_b_prior.separable_weights, requires_grad=True)

    model_L12_weighted_cross_c_b_prior.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L12_weighted_cross_c_b_prior
    test_model_and_make_figs(model)
    # %% Weighted L12 model (l2 on bands) with learned w_{b}, lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12_cross_b'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L12_weighted_cross_b'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L12_weighted_cross_b_prior = ListSeparablePrior(L12(l2_axis=(4,)), separable_weights=init_sep_weights)
    model_L12_weighted_cross_b_prior = WeightedPGD(weights_init=init_b_weights, weight_axis = (4,), is_list=True, data_fidelity = data_fid, prior = L12_weighted_cross_c_b_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart = R_restarts, device = device)
    model_L12_weighted_cross_b_prior.weights = torch.nn.Parameter(model_L12_weighted_cross_b_prior.weights, requires_grad=True)
    L12_weighted_cross_b_prior.separable_weights = torch.nn.Parameter(L12_weighted_cross_b_prior.separable_weights, requires_grad=True)

    model_L12_weighted_cross_b_prior.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L12_weighted_cross_b_prior
    test_model_and_make_figs(model)
    # %% Weighted L12 model (l2 on channels) with learned w_{c}, lambda_j, lambda and steps
    group = 'learn_wavelet_reg-WL12_cross_c'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'L12_weighted_cross_c'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    L12_weighted_cross_c_prior = ListSeparablePrior(L12(l2_axis=(1,)), separable_weights=init_sep_weights)
    model_L12_weighted_cross_c_prior = WeightedPGD(weights_init=init_c_weights, weight_axis = (1,), is_list=True, data_fidelity = data_fid, prior = L12_weighted_cross_c_b_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart = R_restarts, device = device)
    model_L12_weighted_cross_c_prior.weights    = torch.nn.Parameter(model_L12_weighted_cross_c_prior.weights, requires_grad=True)
    L12_weighted_cross_c_prior.separable_weights = torch.nn.Parameter(L12_weighted_cross_c_prior.separable_weights, requires_grad=True)

    model_L12_weighted_cross_c_prior.load_state_dict(torch.load(model_path, weights_only=True))

    model = model_L12_weighted_cross_c_prior
    test_model_and_make_figs(model)
    

    # %%
    ############# DRUNET #########
    L_steps = 10
    R_restarts = 25
    epochs =   100
    BATCH_size = 8
    stepsize_init = 0.5
    lambda_init = np.log(1.0/10)
    sigma_denoise_init = -1.0
    #    name = 'learn_drunet'    
    lin_op = IdentityDictionary()


    # %% Drunet where all parameters have been learned
    group = 'learn_drunet_full'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)

    model_name = 'Drunet'
    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained=None, device=device))
    model_Drunet = PGD_Drunet(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, stepsize = 1.0, lambd = 1.0, L_layers = L_steps, R_restart =R_restarts, device = device, sigma_denoise_init=1.0)
    
    model_Drunet.load_state_dict(torch.load(model_path, weights_only=False))

    model = model_Drunet
    test_model_and_make_figs(model)
    # %% Drunet pnp prior where only sigma and stepsize of PGD are learned 
    group = 'learn_drunet_layer-indep_noise_step'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'Drunet_layer-indep_noise_step'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))
    model_Drunet = PGD_Drunet_layerindep(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart =R_restarts, device = device)
    model = model_Drunet

    remove_params_from_model(model)
    model.sigma_den_list = torch.nn.ParameterList(model.sigma_den_list)
    model.lambd = torch.nn.Parameter(model.lambd)
    model.stepsize = torch.nn.Parameter(model.stepsize)    
    model_Drunet.load_state_dict(torch.load(model_path, weights_only=False))

    model = model_Drunet
    test_model_and_make_figs(model)

    # %% Drunet pnp prior where one sigma by layer and stepsize of PGD are learned 
    group = 'learn_drunet_noises_step'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'Drunet_noises_step'

    savemodels_dir = 'models/' + filenames + '/' ## path to load models
    model_path = savemodels_dir + model_name + '.pt'
    testmodels_figs_dir = savemodels_dir + 'test_figs' ## Savefig path of tests on trained models
    if not os.path.exists(testmodels_figs_dir):
        os.makedirs(testmodels_figs_dir)
    testmodels_figs_dir += '/'

    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))
    model_Drunet = PGD_Drunet(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart =R_restarts, device = device, sigma_denoise_init=sigma_denoise_init)
    model = model_Drunet

    remove_params_from_model(model)
    model.sigma_den_list = torch.nn.ParameterList(model.sigma_den_list)
    model.lambd = torch.nn.Parameter(model.lambd)
    model.stepsize = torch.nn.Parameter(model.stepsize) 
    model_Drunet.load_state_dict(torch.load(model_path, weights_only=False))

    model = model_Drunet
    test_model_and_make_figs(model)




# %%
