import torch
import torch.distributed as dist

from . import FairseqOptimizer
from fairseq.optim.fp16_optimizer import _FP16OptimizerMixin, _MemoryEfficientFP16OptimizerMixin, DynamicLossScaler


class ConcatOptimizer(FairseqOptimizer):

    def __init__(self, args, optimizer_list):
        self.optimizer_list = optimizer_list
        self.scaler = None
        self.is_fpl6 = None
        self.check_optimizer()
        if self.is_fpl6:
            for optimizer in optimizer_list:
                if self.scaler is None:
                    self.scaler = optimizer.scaler
                    self.scaler.scale_window *= len(optimizer_list)
                    # print(self.scaler.scale_window)
                else:
                    optimizer.scaler = self.scaler
                assert self.scaler is optimizer.scaler

    def check_optimizer(self):
        for optimizer in self.optimizer_list:
            if isinstance(optimizer, _FP16OptimizerMixin) or isinstance(optimizer, _MemoryEfficientFP16OptimizerMixin):
                self.is_fpl6 = True
            else:
                self.is_fpl6 = False
            if not ((isinstance(optimizer, _FP16OptimizerMixin) or isinstance(optimizer, _MemoryEfficientFP16OptimizerMixin))) and (self.is_fpl6 is True):
                raise ValueError('mixture of fp16optimizer and pf32optimizer is not supported')
            if (isinstance(optimizer, _FP16OptimizerMixin) or isinstance(optimizer, _MemoryEfficientFP16OptimizerMixin)) and (self.is_fpl6 is False):
                raise ValueError('mixture of fp16optimizer and pf32optimizer is not supported')

    @property
    def params(self):
        raise NotImplementedError

    def __getstate__(self):
        return [optimizer.__getstate__ for optimizer in self.optimizer_list]

    def get_lr(self):
        return [optimizer.get_lr() for optimizer in self.optimizer_list]
    
    def set_lr(self, lr):
        assert len(lr) == len(self.optimizer_list)
        for l, o in zip(lr, self.optimizer_list):
            o.set_lr(l)

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizer_list]
    
    def load_state_dict(self, state_dict, optimizer_overrides=None):
        for state, optimizer in zip(state_dict, self.optimizer_list):
            optimizer.load_state_dict(state, optimizer_overrides)
            if 'loss_scale' in state_dict:
                assert self.scaler.loss_scale == state_dict['loss_scale']

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        if self.is_fpl6:
            loss_scale = self.scaler.loss_scale
            loss = loss * loss_scale
            for optimizer in self.optimizer_list:
                if isinstance(optimizer, _FP16OptimizerMixin):
                    optimizer._needs_sync = True
                elif isinstance(optimizer, _MemoryEfficientFP16OptimizerMixin):
                    optmizer._grads_are_scaled = True
        loss.backward()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for optimizer in self.optimizer_list:
            optimizer.multiply_grads(c)

    def step(self, closure=None):
        """Performs a single optimization step."""
        for optimizer in self.optimizer_list:
            optimizer.step(closure)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        # for optimizer in self.optimizer_list:
        return max([optimizer.clip_grad_norm(max_norm) for optimizer in self.optimizer_list])

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for optimizer in self.optimizer_list:
            optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        return all([optimizer.supports_memory_efficient_fp16 for optimizer in self.optimizer_list])
