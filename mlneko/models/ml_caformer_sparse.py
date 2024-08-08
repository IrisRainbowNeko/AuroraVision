import torch
from einops import repeat, rearrange
from timm.models import create_model
from torch import nn
from rainbowneko.models.layers import GroupLinear
import math
from torch.utils.checkpoint import checkpoint

from .ms_decoder import MSDecoder, MSDecoderLayer
from .position_encoding import build_position_encoding
from .layers import RMSNorm

class MLFormerSpares(nn.Module):
    def __init__(self, encoder, decoder: MSDecoder, num_queries=50, d_model=512, num_classes=1000, scale_skip=0, T=4.,
                 grad_checkpointing=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.final_norm = RMSNorm(d_model)
        self.cls_head = GroupLinear(d_model, math.ceil(num_classes/num_queries), group=num_queries)

        self.feats_trans = nn.ModuleList([nn.Sequential(
            RMSNorm(info['num_chs']),
            nn.Linear(info['num_chs'], d_model),
        ) for info in self.encoder.feature_info[scale_skip:]])
        self.pos_encoder = build_position_encoding('sine', d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.scale_skip = scale_skip
        self.T = T
        self.grad_checkpointing = grad_checkpointing

    def encode(self, x):
        x = self.encoder.stem(x)
        feat_list = []
        for i, stage in enumerate(self.encoder.stages):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(stage, x)
            else:
                x = stage(x)

            if i >= self.scale_skip:
                feat_list.append(x)
        return feat_list

    def decode(self, feat_list):
        q = repeat(self.query_embed.weight, 'q c -> b q c', b=feat_list[0].shape[0])
        pos_emb = [self.pos_encoder(rearrange(x, 'b c h w -> b h w c')).flatten(1,2) for x in feat_list]
        feat_list = [trans(rearrange(x, 'b c h w -> b (h w) c')) for x, trans in zip(feat_list, self.feats_trans)]

        out = self.decoder(q, feat_list, pos=pos_emb)
        return out

    def forward(self, x):
        feat_list = self.encode(x)
        pred = self.decode(feat_list)  # [B, num_queries, C]

        pred = self.final_norm(pred)
        pred = rearrange(pred, 'b (n q) cg -> n q b cg', q=self.num_queries)
        pred = (pred * torch.softmax(pred * self.T, dim=0)).sum(dim=0) # [num_queries, B, ceil(num_classes/num_queries)]
        pred = self.cls_head(pred)  # [num_queries, B, ceil(num_classes/num_queries)]
        pred = pred.transpose(0,1).flatten(1,2)[:, :self.num_classes] # [B, num_classes]
        return pred


def build_mlformer(model_name, d_model=512, pretrained_encoder=True, dec_head_dim=32, dec_layers=6, encoder_ckpt=None,
                   num_queries=50, num_classes=1000, scale_skip=1, T=4.):
    encoder = create_model(model_name, pretrained=pretrained_encoder)
    del encoder.head
    dec_layer = lambda : MSDecoderLayer(d_model, head_dim=dec_head_dim)
    decoder = MSDecoder(dec_layer, dec_layers, norm=RMSNorm(d_model))

    if encoder_ckpt is not None:
        encoder.load_state_dict(torch.load(encoder_ckpt))

    model = MLFormerSpares(encoder, decoder, num_queries=num_queries, num_classes=num_classes,
                     d_model=d_model, scale_skip=scale_skip, T=T)

    return model

def mlformer_H(num_classes, scale_skip=2, T=4.):
    return build_mlformer('caformer_b36.sail_in22k_ft_in1k_384', d_model=640, num_queries=60, dec_layers=5,
                          scale_skip=scale_skip, num_classes=num_classes, T=T)

def mlformer_L(num_classes, scale_skip=2, T=4.):
    return build_mlformer('caformer_b36.sail_in22k_ft_in1k_384', d_model=640, num_queries=60, dec_layers=3,
                          scale_skip=scale_skip, num_classes=num_classes, T=T)

def mlformer_M(num_classes, scale_skip=2, T=4.):
    return build_mlformer('caformer_m36.sail_in22k_ft_in1k_384', d_model=512, num_queries=60, dec_layers=5,
                          scale_skip=scale_skip, num_classes=num_classes, T=T)