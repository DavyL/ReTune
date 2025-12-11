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
from deepinv.optim import data_fidelity
from torchvision.io import read_image
import os
import cyclospin

import wandb

from PGDWaveletL1 import PGD
from cyclospin import IdentityDictionary
from PGDWaveletL1 import SeparablePrior, ListSeparablePrior
from PGDDenoisingPrior import PGD_Drunet,PGD_Drunet_layerindep, GSPnP, remove_params_from_model
from PGDWaveletMSWL12 import WeightedPGD, L12
#
def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()


# %%

if __name__ == "__main__":
    
    def test_model(model, epoch):
        print(f'EPOCH {epoch}. Processing test images')
        with torch.no_grad():
            # Forward pass
            test_batch_len = test_observation.shape[0]
            x_init = test_observation
            u_init = model.dict_fwd(x_init)
            x_outputs, u_outputs = model(x_init, test_observation, u_init, physics)
            loss = loss_fn(x_outputs, test_truth).sum().cpu().detach().numpy()/test_batch_len
            psnr = psnr_fn(x_outputs, test_truth).sum().cpu().detach().numpy()/test_batch_len
            print(f'Test loss is {loss.item()}, psnr is {psnr}')
            x_test_wandb_list = []
            for idx in range(test_batch_len):
                image = wandb.Image(reshape_to_plot(x_outputs[idx]), caption=f"test {idx}")
                x_test_wandb_list.append(image)
            wandb.log({"restored_test_images": x_test_wandb_list})
            wandb.log({"test_epoch": epoch, "test_loss": loss , "test_psnr": psnr})
    def train_model(model):
        for epoch in range(epochs):
            print(f'epoch {epoch}/{epochs}')
            epoch_loss = 0.0
            for batch_idx, (truths, observations) in enumerate(train_dataloader):
                b_size = observations.shape[0]
                truths = truths.to(device)
                observations = torch.tensor(observations, device = device)
                
                # Forward pass
                x_init = observations
                u_init = model.dict_fwd(x_init)
                x_outputs, u_outputs = model(x_init, observations, u_init, physics)
                loss = loss_fn(x_outputs, truths).sum()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                epoch_loss += loss.item()
                psnr = psnr_fn(x_outputs, truths).sum()
                psnr_list.append(psnr.cpu().detach().numpy())
                loss_list.append(loss.cpu().detach().numpy())
                lambda_list.append(model.elambda.cpu().detach().numpy())
                stepsize_list.append(model.estepsize.cpu().detach().numpy())
          
                print(f'loss is {loss.item()/b_size}, psnr is {psnr.item()/b_size}, lambda is {lambda_list[-1]}, step is {stepsize_list[-1]}')
                wandb.log({"epoch": epoch, "train_loss" : loss.item()/b_size, "psnr": psnr.item()/b_size})
                
                for name, param in model.named_parameters():
                    wandb.log({f"params/{name}": param.clone().cpu().detach().numpy()})  # Logging mean value

                torch.cuda.empty_cache()
            wandb.log({"epoch_loss": epoch_loss / len(train_dataloader)})
    ##Test
            test_model(model, epoch)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")
        wandb.finish()
        torch.save(model.state_dict(), savemodels_dir + model_name + '.pt')

    def make_figs(model):
        with torch.no_grad():
            #test_truth = test_truth.to(device)
            observations = torch.tensor(test_observation, device = device)
            crit_list = []
            # Forward pass
            x_init = observations
            u_init = model.dict_fwd(x_init)
            x_outputs, u_outputs = model(x_init, observations, u_init, physics)
            single_step_x_outputs, single_step_u_outputs = model.L_step(x_init, observations, u_init, physics, it=0)
            r_step_x_outputs, r_step_u_outputs = model.R_step(x_init, observations, u_init, physics, crit_list)
            loss = loss_fn(x_outputs, test_truth).sum()
            print(f'loss is {loss.item()}')


        fig, axs = plt.subplots(1,5, figsize=(20,5))
        axs[0].imshow(reshape_to_plot(test_truth[0]))
        axs[1].imshow(reshape_to_plot(observations[0]))
        axs[2].imshow(reshape_to_plot(single_step_x_outputs[0]))
        axs[3].imshow(reshape_to_plot(r_step_x_outputs[0]))
        axs[4].imshow(reshape_to_plot(x_outputs[0]))
        axs[0].set_title(r'gt')
        axs[1].set_title(r'$y$' +                   f'\t{round(psnr_fn(test_observation,        test_truth)[0].sum().item(),2)}db')
        axs[2].set_title(r'$\varphi_{k=0}(y)$' +    f'\t{round(psnr_fn(single_step_x_outputs,   test_truth)[0].sum().item(),2)}db')
        axs[3].set_title(r'$\Phi_{K}^{T=1}(y)$' +   f'\t{round(psnr_fn(r_step_x_outputs,        test_truth)[0].sum().item(),2)}db')
        axs[4].set_title(r'$\Phi_{K}^{T}(y)$' +     f'\t{round(psnr_fn(x_outputs,               test_truth)[0].sum().item(),2)}db')

        for ax in axs:
            ax.axis('off') 
        fig.savefig(f'{savefigs_dir}gt_obs_reconstructions-{model_name}.pdf', bbox_inches ='tight', transparent=True)
        
        #sigma_den_array = np.array(sigma_den_list)  # Shape: (epochs, num_params)
        fig, axs = plt.subplots(1,3, figsize=(20,5))
        axs[0].plot(loss_list)
        axs[0].set_yscale('log')
        axs[1].plot(lambda_list)
        axs[2].plot(stepsize_list)
        #for i in range(sigma_den_array.shape[1]):  # Iterate over the number of parameters
        #    axs[3].plot(sigma_den_array[:, i], label=f'sigma{i}')
        axs[0].set_title(r'$\mathcal{L}$')
        axs[1].set_title(r'$\lambda^{[\ell]}$')
        axs[2].set_title(r'$\tau^{[\ell]}$')
        #axs[3].set_title(r'$\sigma_{den}$')
        #axs[3].legend()  # Add legend to differentiate sigma_den parameters
        for ax in axs:
            ax.grid() 
        fig.savefig(f'{savefigs_dir}criterions_and_parameters_through_epochs-{model_name}.pdf', bbox_inches ='tight', transparent=True)
    def init_training(model):
        config_dict = {
            'stepsize_init' : stepsize_init,
            'lambda_init' : lambda_init,  
            'learning_rate' : learning_rate,
            'L_steps' : L_steps,
            'R_restarts' : R_restarts,
            'epochs' : epochs,
            'operation' : operation,
            'train_dataset_name' : train_dataset_name,
            'test_dataset_name' : test_dataset_name,
            'measurement_dir' : measurement_dir,
            'deepinv_datasets_path' : deepinv_datasets_path,
            'img_size' : img_size,
            'n_channels' : n_channels,
            'probability_mask' : probability_mask,
        }
        wandb.init(name = filenames, project=project, group = group, config = config_dict)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))
        
        ##Saving test images
        x_test_wandb_list = []
        y_test_wandb_list = []
        for idx in range(test_observation.shape[0]):
            image_x = wandb.Image(reshape_to_plot(test_truth[idx]), caption=f"test gt {idx}")
            image_y = wandb.Image(reshape_to_plot(test_observation[idx]), caption=f"test obs {idx}")
            x_test_wandb_list.append(image_x)
            y_test_wandb_list.append(image_y)
        wandb.log({"test_obs_images": y_test_wandb_list, "test_gt_images": x_test_wandb_list})
        
    # %%
    project = 'restart_denoising'
    L_steps = 10
    R_restarts = 25
    epochs =   100
    BATCH_size = 8

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

    # %%
    lin_op = IdentityDictionary()

    stepsize_init = 0.5
    lambda_init = np.log(1.0/10)
    sigma_denoise_init = -1.0

    data_fid = dinv.optim.L2()
    
    #learning_rate = 5e-5
    learning_rate = 5e-2

    # choose training losses
    loss_fn = dinv.loss.SupLoss(metric=dinv.metric.MSE())
    psnr_fn = dinv.metric.PSNR()
    
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(reshape_to_plot(test_truth[0]))
    axs[1].imshow(reshape_to_plot(test_observation[0]))
    axs[1].set_title(f'{round(psnr_fn(test_observation, test_truth).sum().item(),2)}db')

    # %%

    group = 'learn_drunet_full'

    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)

    savefigs_dir = 'figs/' + filenames
    if not os.path.exists(savefigs_dir):
        os.makedirs(savefigs_dir)
    savefigs_dir += '/'
    savemodels_dir = 'models/' + filenames
    if not os.path.exists(savemodels_dir):
        os.makedirs(savemodels_dir)
    savemodels_dir += '/'
    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))
    model_Drunet = PGD_Drunet(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart =R_restarts, device = device, sigma_denoise_init=sigma_denoise_init)
    
    model = model_Drunet
    model_name = 'Drunet'

    for name, param in model.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    loss_list       = []
    psnr_list       = []
    lambda_list     = []
    stepsize_list   = []
    init_training(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    train_model(model)
    make_figs(model)
    # %%

    group = 'learn_drunet_layer-indep_noise_step'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'Drunet_layer-indep_noise_step'

    savefigs_dir = 'figs/' + filenames
    if not os.path.exists(savefigs_dir):
        os.makedirs(savefigs_dir)
    savefigs_dir += '/'
    savemodels_dir = 'models/' + filenames
    if not os.path.exists(savemodels_dir):
        os.makedirs(savemodels_dir)
    savemodels_dir += '/'
    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))
    model_Drunet = PGD_Drunet_layerindep(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart =R_restarts, device = device)
    model = model_Drunet

    remove_params_from_model(model)
    model.sigma_den_list = torch.nn.ParameterList(model.sigma_den_list)
    model.lambd = torch.nn.Parameter(model.lambd)
    model.stepsize = torch.nn.Parameter(model.stepsize)    

    for name, param in model.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    loss_list       = []
    psnr_list       = []
    lambda_list     = []
    stepsize_list   = []

    init_training(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    train_model(model)
    make_figs(model)
    # %%

    group = 'learn_drunet_noises_step'
    filenames = group + "_" + operation + "_" + train_dataset_name + "_B" + str(BATCH_size) + "_L" + str(L_steps) + "_R" + str(R_restarts) + "_E" + str(epochs)
    model_name = 'Drunet_noises_step'

    savefigs_dir = 'figs/' + filenames
    if not os.path.exists(savefigs_dir):
        os.makedirs(savefigs_dir)
    savefigs_dir += '/'
    savemodels_dir = 'models/' + filenames
    if not os.path.exists(savemodels_dir):
        os.makedirs(savemodels_dir)
    savemodels_dir += '/'
    Drunet_prior = dinv.optim.prior.PnP(denoiser=dinv.models.DRUNet(pretrained="download", device=device))
    model_Drunet = PGD_Drunet(data_fidelity = data_fid, prior = Drunet_prior, linear_operator = lin_op, stepsize = stepsize_init, lambd = lambda_init, L_layers = L_steps, R_restart =R_restarts, device = device, sigma_denoise_init=sigma_denoise_init)
    model = model_Drunet

    remove_params_from_model(model)
    model.sigma_den_list = torch.nn.ParameterList(model.sigma_den_list)
    model.lambd = torch.nn.Parameter(model.lambd)
    model.stepsize = torch.nn.Parameter(model.stepsize)    

    for name, param in model.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    loss_list       = []
    psnr_list       = []
    lambda_list     = []
    stepsize_list   = []

    init_training(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    train_model(model)
    make_figs(model)
