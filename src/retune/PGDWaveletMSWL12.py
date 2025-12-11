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
import cyclospin
from SeparablePriors import ListSeparablePrior

#
def reshape_to_plot(img):
    C, H, W = img.shape
    return torch.einsum('bxy->xyb',img).cpu().detach().numpy()

# %%
class L12(dinv.optim.L12Prior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def prox(self, x, *args, gamma, **kwargs):
        r"""
        Calculates the proximity operator of the :math:`\ell_{1,2}` function at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = (1 - \frac{\gamma}{max{\Vert x \Vert_2,\gamma}}) x


        where :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param int l2_axis: axis in which the l2 norm is computed.
        :return torch.Tensor: proximity operator at :math:`x`.
        """

        z = torch.norm(x, p=2, dim=self.l2_axis, keepdim=True) # Compute the norm
        z2 = torch.max(z, gamma*torch.ones_like(z)) # Compute its max w.r.t. gamma at each point
        z3 = torch.ones_like(z) #Construct a mask of ones 
        mask_z = z > 0 #Find locations where z (hence x) is not already zero
        z3[mask_z] = z3[mask_z] - gamma/z2[mask_z]  #If z < gamma -> z2 = gamma -> z3 -gamma/gamma =0  (threshold below gamma)
                                                    #Oth. z3 = 1- gamma/z2
        z4 = torch.multiply(x, z3) # All elems of x with norm < gamma are set 0; the others are z4 = x(1-gamma/|x|)   
        # Creating a mask to avoid diving by zero
        # if an element of z is zero, then it is zero in x, therefore torch.multiply(z, x) is zero as well
        return z4

class WeightedPGD(PGD):
    r"""
     Pour résoudre 
     .. math:: 
        \widehat{x}_W(y) = \operatorname{argmin}_x ||Ax - y||_2^2 + R(Wx)
     on résoud \tilde{x}(y) = argmin_x ||Aw^{-1}x - y||_2^2 + R(x)
     et on pose \widehat{x}_W(y) = W^{-1}\tilde{x}(y)
    """
    def __init__(self, weights_init, weight_axis, is_list =False, *args, **kwargs):
        super(WeightedPGD, self).__init__(*args, **kwargs)    
        self.weights = weights_init
        self.weight_axis = weight_axis
        self.apply_inv_weight   = self.apply_inv_weight_tensor
        self.apply_weight = self.apply_weight_tensor

        ## Si dict_fwd renvoie une liste, on doit utiliser une fonction differente pour appliquer les poids
        # On définit donc des fonctions différentes pour appliquer les poids dans le range de dict_fwd
        # dict_adj est toujours un tenseur
        if is_list is True:
            self.apply_inv_weight_range   = self.apply_inv_weight_list
            self.apply_weight_range = self.apply_weight_list
        else:
            self.apply_inv_weight_range   = self.apply_inv_weight_tensor
            self.apply_weight_range = self.apply_weight_tensor

    def L_step(self, x, y, u, physics, it):
        v = u -  self.estepsize * self.apply_inv_weight_range(self.dict_fwd(physics.A_adjoint(physics.A(self.dict_adj(self.apply_inv_weight_range(u))) - y)))
        u = self.prior.prox(v, ths=1.0,gamma= self.elambda * self.estepsize)
        x_out = self.dict_adj(self.apply_inv_weight_range(u))
        return x_out, u

    #def criterion(self, y, u, physics):
    #    df  = self.data_fidelity.fn(self.dict_adj(self.apply_weight_range(u)), y, physics)
    #    reg = self.elambda*self.prior.fn(self.apply_weight_range(u))
    #    return df+reg
    def criterion(self, y, u, physics):
        x=self.apply_inv_weight_range(u)
        df  = self.data_fidelity.fn(self.dict_adj(self.apply_inv_weight_range(x)), y, physics)
        reg = self.elambda*self.prior.fn(x)
        return df+reg
    
    def apply_weight_tensor(self, x: torch.Tensor):
        w_shape = [1] * x.ndim  # Create a shape of all ones
        for i, axis in enumerate(self.weight_axis):
            w_shape[axis] = self.weights.shape[i]  # Insert w's dimensions into the corresponding axes
    
        w_reshaped = torch.exp(self.weights).view(w_shape)
        return x * w_reshaped     
    def apply_inv_weight_tensor(self, x: torch.Tensor):
        w_shape = [1] * x.ndim  # Create a shape of all ones
        for i, axis in enumerate(self.weight_axis):
            w_shape[axis] = self.weights.shape[i]  # Insert w's dimensions into the corresponding axes
    
        w_reshaped = torch.exp(-self.weights).view(w_shape)
        return x * w_reshaped     
    
    def apply_weight_list(self, x: dinv.utils.TensorList):
        for i, tensor in enumerate(x):
           x[i] = self.apply_weight_tensor(tensor)
        return x
  
    def apply_inv_weight_list(self, x: dinv.utils.TensorList):
        for i, tensor in enumerate(x):
           x[i] = self.apply_inv_weight_tensor(tensor)
        return x 
           
       
#class WeightedL12(L12):
#    def __init__(self, weights_init, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.weights = weights_init
#
#    def fn(self, x, *args, **kwargs):
#        weighted_x = self.apply_weight(x, self.weights, self.l2_axis)
#
#        x_l2 = torch.norm(weighted_x, p=2, dim=self.l2_axis)
#        return torch.norm(x_l2.reshape(x.shape[0], -1), p=1, dim=-1)
#    
#    def prox(self, x, *args, gamma, **kwargs):
#
#        weighted_x = self.apply_weight(x, self.weights, self.l2_axis)
#        mask_z = weighted_x > 0
#        z = torch.norm(weighted_x, p=2, dim=self.l2_axis, keepdim=True) # Compute the norm
#        z2 = torch.max(z, gamma*torch.ones_like(z)) # Compute its max w.r.t. gamma at each point
#        z3 = torch.ones_like(z) #Construct a mask of ones 
#        mask_z = z > 0 #Find locations where z (hence x) is not already zero
#        z3[mask_z] = z3[mask_z] - gamma/z2[mask_z]  #If z < gamma -> z2 = gamma -> z3 -gamma/gamma =0  (threshold below gamma)
#                                                    #Oth. z3 = 1- gamma/z2
#        z4 = torch.multiply(x, z3) # All elems of x with norm < gamma are set 0; the others are z4 = x(1-gamma/|x|)   
#        # Creating a mask to avoid diving by zero
#        # if an element of z is zero, then it is zero in x, therefore torch.multiply(z, x) is zero as well
#        return z4
#    
#    def apply_weight(self, x: torch.Tensor, w: torch.Tensor, weight_axis: tuple):
#        w_shape = [1] * x.ndim  # Create a shape of all ones
#        for i, axis in enumerate(weight_axis):
#            w_shape[axis] = w.shape[i]  # Insert w's dimensions into the corresponding axes
#    
#        w_reshaped = torch.exp(w).view(w_shape)
#        return x * w_reshaped






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

    img_size = 128 if torch.cuda.is_available() else 32
    C_channels = 3  # 3 for color images, 1 for gray-scale images
    probability_mask = 0.5  # probability to mask pixel


    # Initialize wavelet transform
    B_bands = 3
    J_scales = 4
    wav_name = "db4"
    wavelet_transform = cyclospin.WaveletDictionaryDI(wavelet_name=wav_name, levels=J_scales, device=device)
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

    stepsize_init = np.log(1.99)
    #stepsize_init = 0.5
    lambda_init = np.log(0.1)
    L_steps = 1000
    R_restarts = 1

    x_init = obs
    u_init = lin_op.forward(x_init)

    init_sep_weights = torch.zeros_like(torch.randn(J_scales+1, device=device)) ##Separable weights for \sum_j \lambda_j ||c_j||_2
    init_c_b_weights = torch.zeros_like(torch.randn((C_channels, B_bands), device = device)) ##Non Separable weights for  \sqrt{\sum_{c,b} w_{c,b} x_{c,b}^2}
    init_c_weights = init_c_b_weights[:,0]
    init_b_weights = init_c_b_weights[0,:]

    # %%
    L1_prior = ListSeparablePrior(dinv.optim.L1Prior(), separable_weights=init_sep_weights)
 
    with torch.no_grad():
        model_L1 = PGD(data_fid, L1_prior, lin_op, stepsize_init, lambda_init, L_steps, R_restarts, device, zero_init=True)

    for name, param in model_L1.named_parameters():
        print(f'name {name}, param {param}, param.data {param.data}')
    x_outputs_L1, u_outputs_L1, crit_L1 = model_L1(x_init, obs, u_init, physics, ret_crit = True)

# %%
    fig, axs = plt.subplots(1,1, figsize=[15,5], sharey=True)
    axs.plot(crit_L1)
    #axs.plot(crit_cross_b)
    #axs.plot(crit_cross_b)
    axs.grid()
    axs.set_yscale('log')
    axs.set_title(r'$WL^1$')




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
    #%%
    #fig.axis('off')
    fig, axs = plt.subplots(1,7, figsize=[15,5], sharey=True)
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



    #fig.savefig(savefigs_dir + 'criterions.pdf', bbox_inches ='tight', transparent=True)






# %%
