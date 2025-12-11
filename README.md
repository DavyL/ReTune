# LIR3A
Linear and Regularized Regression for Regularity and Anisotropy. A Python library for multiscale regularity and anisotropy estimation using regularized linear regression.

This toolbox allows to compute anisotropic self-similarity parameters pointwise and perform texture segmentation from these. Computations are performed on a gpu, if available, using pytorch. This toolbox can be interfaced with the DeepInverse library. 

# Installation 
In the repository root
'python -m pip install .'

The following library is required to compute undecimated complex dual-tree wavelet coefficients : (https://github.com/DavyL/undecimated_dtcwt)

External libraries required : 'torch deepinv scikit-learn wandb h5py'