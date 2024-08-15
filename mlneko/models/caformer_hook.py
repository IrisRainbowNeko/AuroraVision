from timm.models.metaformer import MetaFormerBlock, MetaFormerStage
from torch import Tensor
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

def stage_forward_hook(self: MetaFormerStage, x: Tensor):
        x = self.downsample(x)
        B, C, H, W = x.shape
        x_loss_list = []

        if not self.use_nchw:
            x = x.reshape(B, C, -1).transpose(1, 2)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                x = checkpoint(block, x)
                loss = F.relu(x.abs() - 100)
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                loss = loss.mean()/len(self.blocks)
                x_loss_list.append(loss)
        else:
            for block in self.blocks:
                x = block(x)
                loss = F.relu(x.abs() - 100)
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                loss = loss.mean()/len(self.blocks)
                x_loss_list.append(loss)

        if not self.use_nchw:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x, sum(x_loss_list)

MetaFormerStage.forward = stage_forward_hook