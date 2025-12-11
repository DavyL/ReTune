# %%

#from ptwt.conv_transform_2 import *
import ptwt
import pywt
import torch
import numpy as np
import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset
import matplotlib.pylab as plt
import numpy as np
from torchvision.io import read_image
import pytorch_wavelets as dtptwt
from itertools import chain
#from deepinv.utils.demo import get_data_home
#ORIGINAL_DATA_DIR = get_data_home()
def reshape_to_plot(img):
    C, H, W = img.shape
    im_rs = torch.einsum('bxy->xyb',img).cpu().detach().numpy()
    return (im_rs - im_rs.min())/(im_rs.max() - im_rs.min())

def re_arrange_bands_fwd(c_from_dwt):
    return c_from_dwt.permute(0,1,3,4,2)
def re_arrange_bands_adj(c):
    return c.permute(0,1,4,2,3)
def re_arrange_scales_fwd(c_from_dwt):
    # First, extract the individual tensors
    X_J, X_list = c_from_dwt[0], c_from_dwt[1]  # Assuming tensor_list = [X_J, (X_1, X_2, ..., X_J)]

    # Permute each tensor in the list BxCxDxH_JxW_J -> BxCxH_JxW_JxD
    X_list = [re_arrange_bands_fwd(X) for X in X_list]  # Permute each X_j
    X_J_ext = X_J.unsqueeze(-1).expand(-1, -1, -1, -1, 3) # Extends one dimension to the approximation so they have same shape as details
    return X_list + [X_J_ext]
def re_arrange_scales_adj(c):
    # First, extract the individual tensors
    X_list, X_J = c[:-1], c[-1][...,0]  # Retrieves detail coeffs and removes last dimension of approx coeficients

    X_list = [re_arrange_bands_adj(X) for X in X_list]  # Permute each X_j
    return (X_J, X_list)


class WaveletDictionaryKB(torch.nn.Module):
    r"""
    Computes DWT using the "Kingsbury" implementation https://github.com/fbcotter/pytorch_wavelets of a tensor of shape [B,C,H,W]
    Returns a list of length J+1 of tensors {d_1,..., d_J, c_J} with d_j, resp. c_j, the detail, resp. scale, coefficients
    Each d_j has shape [B,C,H_j, W_j, D] where D corresponds to the  number of orientations (D=3 for separable wavelets)
    The c_J has shape [B,C,H_J,W_J,D] where coordinates are replicated in the last dimension (this avoids having different last dimension shapes of wavelet coefficients)
    """
    def __init__(self, wavelet_name="haar", levels=4, mode='periodization', device='cpu'):
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        self.dwt = dtptwt.DWTForward(J=levels, mode=mode, wave=wavelet_name).to(device)  # Accepts all wave types available to PyWavelets
        self.iwt = dtptwt.DWTInverse(mode=mode, wave=wavelet_name).to(device)
  
    def forward(self, x):
        c_dwt = self.dwt(x)
        c = re_arrange_scales_fwd(c_dwt)
        return dinv.utils.TensorList(c)

    def adjoint(self, c):
        c_dwt = re_arrange_scales_adj(c)
        x = self.iwt(c_dwt)
        return x

def reshape_from_DI_to_TensorList(Y, D=3):
    """
    Reshape the wavelet coefficients Y (a list-of-lists) into a packed list Y_rs.
    
    Parameters:
      Y: list
         The original wavelet coefficients such that:
           - Y[0] is a list of B tensors of shape [C, w, h].
           - For i >= 1, Y[i] is a list of length detail_channels
             containing tensors of shape [B, C, w, h].
      detail_channels: int
         Number of detail subbands (typically 3).
         (Must equal the length of Y[i] for i>=1.)
    
    Returns:
      Y_rs: list
         A list of length J + 1 (with J = len(Y)-1) where:
           - Y_rs[0] is a tensor of shape [B, C, w, h, detail_channels]. 
             It is obtained by stacking the B tensors from Y[0]
             (each with shape [C, w, h]) into shape [B, C, w, h]
             then “extending” the last dimension.
             (Here we replicate the approximation coefficients along a new dim of size D.)
           - For i >= 1, Y_rs[i] is a tensor of shape [B, C, w, h, detail_channels]
             such that the d'th slice in the last dimension is taken from Y[i][d].
             
    Notes:
      For i>=1, we assume that the coefficient tensors in Y[i] have a batch dimension
      that is identical for every element (so we take the first entry along that dimension).
    """
    Y_rs = []
    Y0 = torch.stack(list(Y[0]), dim=0)
    Y0 = Y0.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    Y_rs.append(Y0)
    
    for sublist in Y[1:]:
        if len(sublist) != D:
            raise ValueError("Unexpected number of detail channels in a sublist.")
        detail_tensor = torch.stack(sublist, dim=-1)
        Y_rs.append(detail_tensor)
    
    return Y_rs

def reshape_from_TensorList_to_DI(Y_rs):
    """
    Inverse of reshape_from_DI_to_TensorList.

    Parameters:
      Y_rs: list
         A list of length J+1, where each element is a tensor of shape [B, C, w, h, D].
         - Y_rs[0] contains approximation coefficients, repeated along the last dim.
         - Y_rs[1:] are detail coefficients, where each d-th slice in the last dim
           corresponds to the d-th detail subband.

    Returns:
      Y: list of lists
         - Y[0] is a list of B tensors of shape [C, w, h] (approximation coeffs).
         - For i >= 1, Y[i] is a list of D tensors, each of shape [B, C, w, h],
           corresponding to the detail subbands.
    """
    Y = []

    # Extract approximation coefficients: [B, C, w, h, D]
    # We take only one channel (e.g., first one) from the last dimension since it's repeated
    Y0 = Y_rs[0][..., 0]  # shape: [B, C, w, h]
    Y0_list = Y0#torch.unbind(Y0, dim=0)  # List of B tensors [C, w, h]
    Y.append(Y0_list)

    # Extract detail coefficients
    for detail_tensor in Y_rs[1:]:
        # detail_tensor: [B, C, w, h, D]
        D = detail_tensor.shape[-1]
        detail_list = [detail_tensor[..., d] for d in range(D)]  # list of D tensors [B, C, w, h]
        Y.append(detail_list)

    return Y 
class WaveletDictionaryDI(torch.nn.Module):
    r"""
    Computes DWT of a tensor of shape [B,C,H,W]
    Returns a list of length J+1 of tensors (FIX) { c_J, d_1,..., d_J,} with d_j, resp. c_j, the detail, resp. scale, coefficients
    Each d_j has shape [B,C,H_j, W_j, D] where D corresponds to the  number of orientations (D=3 for separable wavelets)
    The c_J has shape [B,C,H_J,W_J,D] where coordinates are replicated in the last dimension (this avoids having different last dimension shapes of wavelet coefficients)
    """
    def __init__(self, wavelet_name="haar", levels=4, mode='periodization', device='cpu'):
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        self.dwt = dinv.models.WaveletDenoiser(level=levels, wv=wavelet_name).dwt
        self.iwt = dinv.models.WaveletDenoiser(level=levels, wv=wavelet_name).iwt
    def forward(self, x):
        c_dwt = self.dwt(x)
        c = reshape_from_DI_to_TensorList(c_dwt)
        return dinv.utils.TensorList(c)

    def adjoint(self, c):
        c_dwt = reshape_from_TensorList_to_DI(c)
        x = self.iwt(c_dwt)
        return x
    
class MultipleWaveletsDictionary(torch.nn.Module):
    r"""
    Computes DWT of a tensor of shape [B,C,H,W] over a family of M wavelets \Psi = (\psi_1,...,\psi_M)
    Returns a list of length M*(J+1) of tensors ({d_1^1,..., d_J^1, c_J^1}, ...,d_1^M,..., d_J^M, c_J^M}) 
    with d_j^m, resp. c_j^m, the detail, resp. scale, coefficients of wavelet \psi^m computed by a WaveletDictionary()
    The adjoint is obtained by averaging the reconstructions over all wavelets
   """
    def __init__(self, wavelet_names=["haar", "db4"], levels=4, mode='periodization', device='cpu'):
        super().__init__()
        self.M = len(wavelet_names)
        self.wavelet_family = [pywt.Wavelet(wavelet_name) for wavelet_name in wavelet_names]
        self.levels = levels
        self.dwts = [dtptwt.DWTForward(J=levels, mode=mode, wave=wavelet_name).to(device) for wavelet_name in wavelet_names]  # Accepts all wave types available to PyWavelets
        self.iwts = [dtptwt.DWTInverse(mode=mode, wave=wavelet_name).to(device) for wavelet_name in wavelet_names]
  
    def forward(self, x):
        c_nested = [re_arrange_scales_fwd(dwt(x)) for dwt in self.dwts]
        # Flatten the list of lists into a single list.
        c = list(chain.from_iterable(c_nested))
        return dinv.utils.TensorList(c)        
        #c = []
        #for dwt in self.dwts:
        #    c_dwt = dwt(x)
        #    c = c + re_arrange_scales_fwd(c_dwt) #List concatenation of wavelet transforms
        #return dinv.utils.TensorList(c)

    def adjoint(self, c):
        x_list = [
            iwt(re_arrange_scales_adj(c[m*(self.levels + 1):(m+1)*(self.levels + 1)]))
            for m, iwt in enumerate(self.iwts)
        ]
        # Stack all reconstructed images into a single tensor and compute their mean.
        x_stack = torch.stack(x_list, dim=0)
        return x_stack.mean(dim=0)       
        #x_list = []
        #m=0
        #for iwt in self.iwts:
        #    start_slice, stop_slice = m*(self.levels + 1),(m+1)*(self.levels + 1)
        #    c_dwt = re_arrange_scales_adj(c[start_slice:stop_slice]) #Extract coefs of each dwt
        #    x_list.append(iwt(c_dwt))
        #    m+=1
#
        #x_sum = torch.zeros_like(x_list[0])
        #for x_single in x_list:
        #    x_sum += x_single #Sum each reconstructed image
        #return x_sum/len(x_list) #Return average reconstruction


    
class IdentityDictionary(torch.nn.Module):
    def __init__(self, ):
        super(IdentityDictionary, self).__init__() 
    def forward(x):
        return x
    def adjoint(x):
        return x
    
class SynthesisDictionary(torch.nn.Module):
    def __init__(self, F_features, device, filter_size = (1,3,5,5), sigma=0.2):
        super(SynthesisDictionary, self).__init__() 
        self.F_features = F_features
        self.filters = [torch.nn.Parameter(sigma*torch.randn(filter_size, device=device)) for f in range(self.F_features)]

    #x of shape [B,C,H,W,F]
    #Dx of shape [B,C,H,W]
    def forward(self, x):
        out = np.zeros_like(x.shape[:-1])
        for f in range(self.F_features):
            out += dinv.physics.functional.conv2d_fft(x, self.filters[f])
        return out
    
    #x of shape [B,C,H,W]
    #D*x of shape [B,C,H,W, F]
    def adjoint(self, x):
        out = np.zeros_like([*(x.shape), self.F_features])
        for f in range(self.F_features):
            out[...,f] = dinv.physics.functional.conv_transpose2d_fft(x, self.filters[f])
        return out
    

if __name__ == "__main__":
#    BASE_DIR = Path(".")
#    print(BASE_DIR)
#    DATA_DIR = BASE_DIR / "measurements"
#    CKPT_DIR = BASE_DIR / "ckpts"

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    image_path = "../../ressources/393035.jpg"
    image_file = read_image(image_path)
    image = image_file.unsqueeze(0)[...,:256,:256].to(torch.float32).to(device)/255
    print(f'image has shape {image.shape}')

    # %%
    ##Real transform
    xfm = dtptwt.DWTForward(J=5, mode='zero', wave='sym4').to(device)  # Accepts all wave types available to PyWavelets
    ifm = dtptwt.DWTInverse(mode='zero', wave='sym4').to(device)
    X = torch.randn(10,5,256,256)
    Yl, Yh = xfm(image)
    print(f'low pass dwt {Yl.shape}')
    print(Yh[0].shape, Yh[1].shape, Yh[2].shape)
    Y = ifm((Yl, Yh))
    print(Y.shape)
    print(torch.sum(torch.square(Y-image)))
    ##Dual-tree transform

    # %%
    xfm = dtptwt.DTCWTForward(J=7, biort='near_sym_b', qshift='qshift_b')
    ifm = dtptwt.DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
    X = torch.randn(10,5,64,64)
    Yl, Yh = xfm(X)
    print(f'low pass dtcwt {Yl.shape}')
    #print(Yh[0].shape,Yh[1].shape,Yh[2].shape)
    Y = ifm((Yl, Yh))
    print(Y.shape)
    print(torch.sum(torch.square(Y-X)))
    #for j in range(0,6):
    #    print(f'\nj:{j}\n')
    #    for k_x in range(0, 2**(j+1)):
    #        print(f'k : {k_x}\t  {get_offset_adj(k_x, j)}\t 2^j {2**j}\t {get_offset_fwd(get_offset_fwd(k_x, j),j)}')

    # %%
    ##TEST IMPLEM DEEPINV
    wavelet_type = 'db2'
    J = 4
    B = 17
    C = 5
    D = 3
    L = dinv.models.WaveletDenoiser(level=J, wv=wavelet_type).dwt
    L_adjoint = dinv.models.WaveletDenoiser(level=J, wv=wavelet_type).iwt
    X = torch.randn(B,C,128,128)

    Y_dec = L(X)
    Y_rs = reshape_from_DI_to_TensorList(Y_dec)
    Y_rs_rs = reshape_from_TensorList_to_DI(Y_rs)
    Y_rec = L_adjoint(Y_dec)
    Y_rs_rs_rec = L_adjoint(Y_rs_rs)
    print(torch.sum(torch.square(Y_rec - X)))
    print(torch.sum(torch.square(Y_rec - Y_rs_rs_rec)))
    print(torch.sum(torch.square(Y_rs_rs_rec - X)))
    
    # %%

    # Initialize wavelet transforms
    nlevels = 4
    wav_name = "db8"
    #cyclo_transform = CycloWaveletDictionary(wavelet_name=wav_name, levels=nlevels, mode="zero")
    wavelet_transform = WaveletDictionaryDI(wavelet_name=wav_name, levels=nlevels, device = device)
    wav_coeffs = wavelet_transform.forward(image)
    #print("Wavelet decomposition shape:", wav_coeffs[0].shape, wav_coeffs[-1].shape)

    reconstructed_image = wavelet_transform.adjoint(wav_coeffs)
    error = torch.norm(image - reconstructed_image) 
    
    wav_names = ["haar", "db2", "coif4", "db8"]
    #cyclo_transform = CycloWaveletDictionary(wavelet_name=wav_name, levels=nlevels, mode="zero")
    multiple_wavelet_transform = MultipleWaveletsDictionary(wavelet_names=wav_names, levels=nlevels, device = device)
    multiple_wav_coeffs = multiple_wavelet_transform.forward(image)
    #print("Wavelet decomposition shape:", wav_coeffs[0].shape, wav_coeffs[-1].shape)

    multiple_reconstructed_image = multiple_wavelet_transform.adjoint(multiple_wav_coeffs)
    multiple_error = torch.norm(image - multiple_reconstructed_image) 
    
    print("Reconstruction error:", error.item())
    print("Reconstruction error (multiple wavelets):", multiple_error.item())
    fig, axs = plt.subplots(1,3,figsize=(12,6))#print(reconstructed_face)
    axs[0].imshow(reshape_to_plot(image[0]))
    axs[0].set_title(r'gt')
    axs[0].axis('off')
    axs[1].imshow(reshape_to_plot(reconstructed_image[0]))
    axs[1].set_title(r'$D^*c$')
    axs[1].axis('off')

    axs[2].imshow(reshape_to_plot(multiple_reconstructed_image[0]))
    axs[2].set_title(r'$D^*c$ (multiple)')
    axs[2].axis('off')

    #fig.savefig(f'figs/first_plots/standard_and_adjoint-undec_{wav_name}_{nlevels}scales.pdf')

    fig,axs = plt.subplots(3, nlevels+1, figsize=(30,20))
    for band in range(3):
        for j in range(nlevels):
            axs[band,j].imshow(reshape_to_plot(wav_coeffs[j][0,:,:,:,band]))
            axs[band, j].axis('off')
        axs[band,nlevels].imshow(reshape_to_plot(wav_coeffs[-1][0,:,:,:,0]))
        axs[band,nlevels].axis('off')
    #fig.savefig(f'figs/first_plots/undec_coefs_{wav_name}_{nlevels}scales.pdf')



# %%

 
