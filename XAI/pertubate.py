from abc import ABC, abstractmethod

import torch
from torch import Tensor

# class Pertubation(ABC):
#     @abstractmethod
#     def __init__(self, device, epsilon=1.0e-7):
#         self.mask = None
#         self.epsilon = epsilon
#         self.device = device
    
#     @abstractmethod
#     def apply(self, X: Tensor, mask: Tensor):
#         if X is None or mask is None:
#             raise NameError("Missing argument")

# class FadeMovingAverage(Pertubation):
#     def __init__(self, device, epsilon=1.0e-7):
#         super().__init__(device, epsilon)
    
#     def apply(self, x: Tensor, mask: Tensor):
#         # x: [time_steps, num_nodes, channels]
#         # mask: [time_steps, num_nodes]
        
#         super().apply(x, mask)
#         T = x.shape[0]
        
#         # [time_steps, num_nodes * channels]
#         avg = torch.mean(x, 0).reshape(1, -1).to(self.device)
#         # [1, num_nodes, channels]
#         avg = avg.reshape(1, *x.size()[1:])
#         # [time_steps, num_nodes, channels]
#         avg = avg.repeat(T, 1, 1, 1)
        
#         based = torch.einsum('tnc, tn -> tnc', x, mask)
#         pert = torch.einsum('tnc, tn -> tnc', avg, 1 - mask)
        
#         # [time_steps, num_nodes, channels]
#         x_pert = based + pert
        
#         return x_pert
    


class Pertubation(ABC):
    @abstractmethod
    def __init__(self, device, eps=1.0e-7):
        self.mask_tensor = None
        self.device = device
        self.eps = eps
    
    @abstractmethod
    def apply(self, X, mask_tensor):
        if X is None or mask_tensor is None:
            raise NameError("The mask_tensor should be fitted before or while calling the apply() method")


class FadeMovingAverage(Pertubation):
    def __init__(self, device, eps=1.0e-7,alpha_init=0.9):
        super().__init__(device, eps)
        self.alpha = torch.tensor(alpha_init, requires_grad=True, device=device)
    
    def apply(self, X, mask_tensor):
        super().apply(X, mask_tensor)
        
        T = X.shape[0]
        moving_average = torch.mean(X, 0).to(self.device)
        moving_average_tilted = moving_average.repeat(T, 1, 1, 1)
        X_pert = mask_tensor * X + (1 - mask_tensor) * moving_average_tilted
        
        return X_pert
    
    
    

class Pertubation1(ABC):
    @abstractmethod
    def __init__(self, device, epsilon=1.0e-7):
        self.mask = None
        self.epsilon = epsilon
        self.device = device
    
    @abstractmethod
    def apply(self, X: Tensor, mask: Tensor):
        if X is None or mask is None:
            raise NameError("Missing argument")

class FadeMovingAverage1(Pertubation1):
    def __init__(self, device, epsilon=1.0e-7):
        super().__init__(device, epsilon)
    
    def apply(self, x: Tensor, mask: Tensor):
        # x: [time_steps, num_nodes, channels]
        # mask: [time_steps, num_nodes]
        
        super().apply(x, mask)
        T = x.shape[0]
        
        # [time_steps, num_nodes * channels]
        avg = torch.mean(x, 0).reshape(1, -1).to(self.device)
        # [1, num_nodes, channels]
        avg = avg.reshape(1, *x.size()[1:])
        # [time_steps, num_nodes, channels]
        avg = avg.repeat(T, 1, 1)
        
        based = torch.einsum('tnc, tn -> tnc', x, mask)
        pert = torch.einsum('tnc, tn -> tnc', avg, 1 - mask)
        
        # [time_steps, num_nodes, channels]
        x_pert = based + pert
        
        return x_pert
        