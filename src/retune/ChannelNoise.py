import deepinv as dinv
import torch 

class ChannelNoiseModel(dinv.physics.noise.NoiseModel):
    def __init__(self, sigmas=[0.1,0.1,0.1],*args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.update_parameters(sigmas)
        self.gaussian_noises = [dinv.physics.GaussianNoise(sigmas[c]) for c in range(3)]
        
    def forward(self, input ,seed=None):
        for c in range(3):
            input[:,c,...] = self.gaussian_noises[c](input[:,c,...])
        return input 
    
    def update_parameters(self, sigmas=None, **kwargs):
        if sigmas is not None:
            #self.sigmas = dinv.physics.noise.to_nn_parameter(sigmas)
            self.sigmas = torch.nn.ParameterList(sigmas)

    def fix_params(self):
        self.gaussian_noises = [dinv.physics.GaussianNoise(self.sigmas[c]) for c in range(3)]

    
if __name__ == "__main__":
    noise_model = ChannelNoiseModel(sigmas=[0.1,0.25,0.5])
    noise_physics = dinv.physics.Denoising(noise_model)
    #noise_physics.set_noise_model(noise_model)
    x = torch.ones(1, 3, 96, 128)/2.0
    #x[:, 0, :32, :] = 1
    #x[:, 1, 32:64, :] = 1
    #x[:, 2, 64:, :] = 1
    y = noise_physics(x)

    print([f'var channel {c} : {torch.std(y[:,c,...])}' for c in range(3)])
    xlin = noise_physics.A_dagger(y)  # compute the linear pseudo-inverse
    
    dinv.utils.plot([x, y, xlin], titles=["image", "meas.", "linear rec."])