from copy import deepcopy
from typing import Optional, Union, Dict, Any

import lightning as L
import torch
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.callbacks.callback import Callback
import torch.distributed as dist


class EMACallBack(Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """
    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True, debug=False):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False
        self.debug = debug

    @staticmethod
    def get_state_dict(pl_module: L.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.
        
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()
        
    @overrides
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.debug:
            print('EMA Callback: on_train_start')
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            if self.debug:
                print('EMA Callback: on_train_start: rank 0')
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

            self.ema_state_dict = {k: tensor.float() for k, tensor in self.ema_state_dict.items()}
        self._ema_state_dict_ready = True

    @rank_zero_only
    @torch.no_grad()
    @torch.compile
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs) -> None:
        if self.debug:
            print('EMA Callback: on_train_batch_end')
        if False:
            # Update EMA weights
            
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)
        else:
            if trainer.fit_loop.epoch_loop._should_accumulate():
                if self.debug:
                    print('EMA Callback: on_train_batch_end: torch._foreach_lerp_')
                tensors_to_lerp = []
                # for key, value in self.get_state_dict(pl_module).items():
                for key, value in pl_module.named_parameters():
                    ema_value = self.ema_state_dict[key]
                    tensors_to_lerp.append((ema_value, value.data.float()))
                tgt_lerp, src_lerp = zip(*tensors_to_lerp)
                torch._foreach_lerp_(tgt_lerp, src_lerp, 1. - self.decay)


    
    @overrides
    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.
        if self.debug:
            print('EMA Callback: on_validation_start')
        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))

        dtype = next(pl_module.parameters()).dtype
        ema_state_dict = {}
        for key in self.original_state_dict:
            v = self.ema_state_dict[key].to(dtype) if pl_module.global_rank == 0 else torch.empty_like(self.original_state_dict[key])
            dist.broadcast(v, 0)
            ema_state_dict[key] = v

        assert ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - ema_state_dict.keys()}"
        pl_module.load_state_dict(ema_state_dict, strict=True)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}


    @overrides
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.
        if self.debug:
            print('EMA Callback: on_validation_end')
        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=True)

    @overrides(check_signature=False)
    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        if self.debug:
            print('EMA Callback: on_save_checkpoint')
        checkpoint['ema_state_dict'] = self.ema_state_dict
        checkpoint['_ema_state_dict_ready'] = self._ema_state_dict_ready

    @overrides
    def on_load_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        if self.debug:
            print('EMA Callback: on_load_checkpoint')
        if "ema_state_dict" in checkpoint and "_ema_state_dict_ready" in checkpoint:
            if self.debug:
                print('EMA Callback: success on_load_checkpoint')
            self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"]
            if pl_module.global_rank == 0:
                self.ema_state_dict = checkpoint["ema_state_dict"]
                if self.ema_device:
                    self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}
                else:
                    device = next(pl_module.parameters()).device
                    self.ema_state_dict = {k: tensor.to(device=device) for k, tensor in self.ema_state_dict.items()}

                if self.ema_device == "cpu" and self.ema_pin_memory:
                    self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}
