from spikingjelly.activation_based import neuron,surrogate
import torch

class CustomNeuron(neuron.ParametricLIFNode):
    def __init__(self, scale_reset: bool = False, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super().__init__( tau, decay_input,v_threshold, v_reset, surrogate_function,
                          detach_reset, step_mode, backend, store_v_seq)
        self.register_memory('v_loss', None)
    
    def single_step_forward(self, x):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        pre_reset_total_v = torch.max(self.v)
        self.neuronal_reset(spike)
        post_reset_total_v =  torch.max(self.v)
        self.v_loss = post_reset_total_v - pre_reset_total_v
        return spike