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
from deepinv.optim import data_fidelity
#from deepinv.utils.demo import get_data_home
#ORIGINAL_DATA_DIR = get_data_home()
import os
from PGDWaveletL1 import PGD, J_scales
from SeparablePriors import SeparablePrior, ListSeparablePrior

def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()



# %%

if __name__ == "__main__":

    WAVELET_NAME = "db6"
    J_scales = 4
    savefigs_dir = 'figs/learn_pgd_lambda_and_steps'
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


    operation = "denoising"
    train_dataset_name = "CBSD68"
    test_dataset_name = "set3c"

    my_dataset_name = "demo_training_denoising"
    measurement_dir = DATA_DIR / train_dataset_name / operation
    img_size = 128 if torch.cuda.is_available() else 32
    n_channels = 3  # 3 for color images, 1 for gray-scale images
    probability_mask = 0.5  # probability to mask pixel

    #physics = dinv.physics.Denoising( device=device)
    physics = dinv.physics.Inpainting(
        tensor_size=(n_channels, img_size, img_size), mask=probability_mask, device=device
    )
    #physics = dinv.physics.Blur(
    #    dinv.physics.blur.gaussian_blur(sigma=(.5, 0.5), angle=0), device=device, padding='reflect'
    #)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(img_size)]
    )
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(img_size)]
    )

    train_dataset = load_dataset(train_dataset_name, train_transform, measurement_dir)
    test_dataset = load_dataset(test_dataset_name,   test_transform, measurement_dir)


    # %%
    num_workers = 4 if torch.cuda.is_available() else 0
    n_images_max = (
        128 if torch.cuda.is_available() else 50
    )  # maximal number of images used for training
    deepinv_datasets_path = dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        device=device,
        save_dir=measurement_dir,
        train_datapoints=n_images_max,
        num_workers=num_workers,
        dataset_filename=str(my_dataset_name),
    )

    train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
    test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)


    # %%


    train_batch_size = 32 if torch.cuda.is_available() else 1
    test_batch_size  = 32 if torch.cuda.is_available() else 1

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
    )
    image = train_dataset[0][0]
    C, H, W = image.shape

    plt.figure()
    plt.imshow(reshape_to_plot(image))


    WaveletDictionary = cyclospin.WaveletDictionary 
    #CycloWaveletDictionary = cyclospin.CycloWaveletDictionary 
    # Initialize wavelet transform
    wavelet_transform = WaveletDictionary(wavelet_name=WAVELET_NAME, levels=J_scales, device = device)
    #cyclo_wavelet_transform = CycloWaveletDictionary(wavelet_name="db1", levels=4)


    data_fid = dinv.optim.L2()
    prior = ListSeparablePrior(dinv.optim.L1Prior(), separable_weights=torch.zeros(J_scales+1, device=device))
    lin_op = wavelet_transform
    #prior = dinv.optim.prior.WaveletPrior(level=3, wv="db8", p=1, device=device)
    #prior = dinv.optim.prior.TVPrior(n_it_max=20)

    stepsize_init = -3.0
    lambda_init = -3.0
    L_steps = 50
    R_restarts = 10

    model = PGD(data_fid, prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device)

    for name, param in model.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')


    # %%
    epochs = 10  # choose training epochs
    learning_rate = 5e-3

    # choose training losses
    loss_fn = dinv.loss.SupLoss(metric=dinv.metric.MSE())

    # choose optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

    loss_list       = []
    lambda_list     = []
    stepsize_list   = []
    for epoch in range(epochs):
        print(f'epoch {epoch}/{epochs}')
        epoch_loss = 0.0
        for batch_idx, (truths, observations) in enumerate(train_dataloader):
            truths = truths.to(device)
            observations = torch.tensor(observations, device = device)
            
            # Forward pass
            x_init = observations
            u_init = model.dict_fwd(x_init)
            x_outputs, u_outputs = model(x_init, observations, u_init, physics)
            loss = loss_fn(x_outputs, truths).sum()
            print(f'loss is {loss.item()}')
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
            loss_list.append(loss.cpu().detach().numpy())
            lambda_list.append(model.elambda.cpu().detach().numpy())
            stepsize_list.append(model.estepsize.cpu().detach().numpy())
            torch.cuda.empty_cache()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")
        #for name, param in model.named_parameters():
        #    print(f'name {name}, param {param}, param.data {param.data}')

    # %%
    iter_loader = iter(test_dataloader)    
    truths, observations = next(iter_loader)
    with torch.no_grad():
        truths = truths.to(device)
        observations = torch.tensor(observations, device = device)
        
        # Forward pass
        x_init = observations
        u_init = model.dict_fwd(x_init)
        x_outputs, u_outputs = model(x_init, observations, u_init, physics)
        single_step_x_outputs, single_step_u_outputs = model.L_step(x_init, observations, u_init, physics)
        loss = loss_fn(x_outputs, truths).sum()
        print(f'loss is {loss.item()}')
    # %%
    fig, axs = plt.subplots(1,4, figsize=(20,5))
    axs[0].imshow(reshape_to_plot(truths[0]))
    axs[1].imshow(reshape_to_plot(observations[0]))
    axs[2].imshow(reshape_to_plot(single_step_x_outputs[0]))
    axs[3].imshow(reshape_to_plot(x_outputs[0]))
    axs[0].set_title(r'gt')
    axs[1].set_title(r'$y$')
    axs[2].set_title(r'$Phi_{K}^{T = 1}(y)$')
    axs[3].set_title(r'$\Phi_{K}^{T}(y)$')
    for ax in axs:
        ax.axis('off') 
    fig.savefig(savefigs_dir + 'gt_obs_reconstructions.pdf', bbox_inches ='tight', transparent=True)
    
    fig, axs = plt.subplots(1,3, figsize=(20,5))
    axs[0].plot(loss_list)
    axs[1].plot(lambda_list)
    axs[2].plot(stepsize_list)
    axs[0].set_title(r'$\mathcal{L}$')
    axs[1].set_title(r'$\lambda^{[\ell]}$')
    axs[2].set_title(r'$\tau^{[\ell]}$')
    for ax in axs:
        ax.grid() 
    fig.savefig(savefigs_dir + 'criterions_and_parameters_through_epochs.pdf', bbox_inches ='tight', transparent=True)


# %%

    #verbose = True  # print training information
    #wandb_vis = False  # plot curves and images in Weight&Bias
    #trainer = dinv.Trainer( model, device=device, save_path=str(CKPT_DIR / operation), verbose=verbose, wandb_vis=wandb_vis, physics=physics, epochs=epochs, scheduler=scheduler, losses=losses, optimizer=optimizer, show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery. 
    #                       train_dataloader=train_dataloader, eval_dataloader=test_dataloader,)
    #model = trainer.train()
    #trainer.test(test_dataloader)
