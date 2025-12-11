
import deepinv as dinv
import torch

class SeparablePrior(dinv.optim.Prior):
    """
    Prior with separable structure along a specific axis of the input tensor.

    The function f is assumed to decompose as
        f(x) = sum_i f(x_i)
    where x_i is a slice of x taken along the separable axis.
    The separable weights (given in log-domain) are exponentiated to ensure
    positivity and scale the contributions of each slice. The proximity operator
    is computed slice-by-slice and the results are concatenated along the separable axis.
    
    Expected input:
      - x: a tensor of shape [A, B, ..., I, ...] where the I-axis (indexed by separable_axis)
           contains the separable components.
    """
    def __init__(self, prior, separable_axis, separable_weights, *args, **kwargs):
        """
        Args:
          prior: an object with methods fn(x, ...) and prox(x, gamma, ...) that define the base function.
          separable_axis: integer index of the axis over which the prior is separable.
          separable_weights: a tensor (or parameter) of weights (in log-domain) for each slice along the separable axis.
        """
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.separable_axis = separable_axis
        self.separable_weights = separable_weights

    def fn(self, x, *args, **kwargs):
        """
        Compute the function value f(x) as the weighted sum over slices.

        For each coordinate along the separable_axis, a slice is taken and the base prior function
        is applied. Each contribution is weighted by exp(separable_weights[coord]).

        Args:
          x: Input tensor.
          *args, **kwargs: Additional arguments passed to prior.fn.
        
        Returns:
          A tensor containing the aggregated function value.
        """
        # Exponentiate the (log-)weights.
        eseparable_weights = torch.exp(self.separable_weights)
        # Initialize f_total with zeros. Assumes the first dimension is the batch dimension.
        f_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        # Loop over the separable axis
        for coord in range(x.shape[self.separable_axis]):
            # Select the slice along the separable axis.
            sliced_x = x.select(dim=self.separable_axis, index=coord)
            # Add the weighted contribution of the prior applied to the slice.
            f_total = f_total + eseparable_weights[coord] * self.prior.fn(sliced_x, *args, **kwargs)
        return f_total

    def prox(self, x, gamma, *args, **kwargs):
        """
        Compute the proximity operator associated with f.

        The prox is computed slice-by-slice along the separable_axis. For each slice:
          prox_{gamma * exp(w) * f}(x_slice)
        is computed, and then the resulting slices are concatenated back along the separable_axis.

        Args:
          x: Input tensor.
          gamma: A scaling parameter.
          *args, **kwargs: Additional arguments passed to prior.prox.
        
        Returns:
          A tensor of the same shape as x after applying the proximity operator.
        """
        eseparable_weights = torch.exp(self.separable_weights)
        prox_slices = []
        # torch.unbind splits the tensor along the separable_axis without squeezing that axis.
        for coord, sliced_x in enumerate(torch.unbind(x, dim=self.separable_axis)):
            # Compute the prox for the current slice.
            prox_slice = self.prior.prox(sliced_x, gamma=gamma * eseparable_weights[coord], *args, **kwargs)
            # unsqueeze to restore the separable axis for later concatenation.
            prox_slices.append(prox_slice.unsqueeze(self.separable_axis))
        # Concatenate all processed slices along the separable axis.
        return torch.cat(prox_slices, dim=self.separable_axis)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass: simply returns the function value f(x).
        """
        return self.fn(x, *args, **kwargs)


class ListSeparablePrior(dinv.optim.Prior):
    """
    Prior with separable structure defined over a dinv.utils.TensorList of tensors.

    In this variant the input is assumed to be a list:
        x = [x_1, x_2, ..., x_J]
    and the function f is decomposable as:
        f(x) = f(x_1) + f(x_2) + ... + f(x_J)
    Each component is weighted by exp(separable_weights[j]), and the proximity operator
    is computed individually for each x_j.
    
    The prox returns a list with the same structure as the input.
    """
    def __init__(self, prior, separable_weights, *args, **kwargs):
        """
        Args:
          prior: an object with methods fn(x, ...) and prox(x, gamma, ...) defining the base function.
          separable_weights: a tensor (or parameter) of weights (in log-domain), one per tensor in the input list.
          
        Note:
          There is no separable_axis here because the separation is given by the list structure.
        """
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.separable_weights = separable_weights

    def fn(self, x, *args, **kwargs):
        """
        Compute the function value f(x) = sum_j exp(separable_weights[j]) * f(x_j).

        Args:
          x: A list of tensors [x_1, x_2, ..., x_J].
          *args, **kwargs: Additional arguments passed to prior.fn.
        
        Returns:
          A tensor representing the aggregated function value.
          (For example, if each prior.fn(x_j) returns a batch of values, they are summed.)
        """
        eseparable_weights = torch.exp(self.separable_weights)
        #f_list = [
        #    eseparable_weights[j] * self.prior.fn(x_j, *args, **kwargs)
        #    for j, x_j in enumerate(x)
        #]
        s = eseparable_weights[0]*self.prior.fn(x[0])/x[0].shape[-1]
        for j in range(1,len(x)):
            s+= eseparable_weights[j] * self.prior.fn(x[j], *args, **kwargs)

        return s        
        #return torch.stack(f_list, dim=0).sum(dim=0)        
        #eseparable_weights = torch.exp(self.separable_weights)
        #f_total = 0
        #for j, x_j in enumerate(x):
        #    f_total = f_total + eseparable_weights[j] * self.prior.fn(x_j, *args, **kwargs)
        #return f_total

    def prox(self, x, gamma, *args, **kwargs):
        """
        Compute the proximity operator for the separable function defined over a list.

        For each tensor x_j in the list, the prox is computed as:
          prox_{gamma * exp(separable_weights[j]) * f}(x_j)
        and the results are returned in a list.

        Args:
          x: A list of tensors [x_1, x_2, ..., x_J].
          gamma: A scaling parameter.
          *args, **kwargs: Additional arguments passed to prior.prox.
        
        Returns:
          A list of tensors, where each tensor is the result of the proximity operator applied to the corresponding input.
        """
        eseparable_weights = torch.exp(self.separable_weights)
        prox_list = [
            self.prior.prox(x_j, gamma=gamma * eseparable_weights[j], *args, **kwargs)
            for j, x_j in enumerate(x)
        ]
        return dinv.utils.TensorList(prox_list)
        #eseparable_weights = torch.exp(self.separable_weights)
        #prox_list = []
        #for j, x_j in enumerate(x):
        #    prox_j = self.prior.prox(x_j, gamma=gamma * eseparable_weights[j], *args, **kwargs)
        #    prox_list.append(prox_j)
        #return dinv.utils.TensorList(prox_list)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass: computes the aggregated function value f(x).
        """
        return self.fn(x, *args, **kwargs)
