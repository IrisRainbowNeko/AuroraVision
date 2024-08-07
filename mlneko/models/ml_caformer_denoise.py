import torch
from einops import repeat, rearrange
from timm.models import create_model
from torch import nn
from rainbowneko.models.layers import GroupLinear
import math

from .ms_decoder import MSDecoder, MSDecoderLayer
from .position_encoding import build_position_encoding
from .layers import RMSNorm

class MLFormerDenoise(nn.Module):
    def __init__(self, encoder, decoder: MSDecoder, num_queries=50, num_denoise_group=2, d_model=512, num_classes=1000,
                 scale_skip=2, T=4.):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.final_norm = RMSNorm(d_model)
        self.cls_encoder = GroupLinear(math.ceil(num_classes/num_queries), d_model, group=num_queries)
        self.cls_head = GroupLinear(d_model, math.ceil(num_classes/num_queries), group=num_queries)

        self.num_denoise_group = num_denoise_group

        self.feats_trans = nn.ModuleList([nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, d_model),
        ) for dim in self.encoder.scale_dims[scale_skip:]])
        self.pos_encoder = build_position_encoding('sine', d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.scale_skip = scale_skip
        self.T = T

    def encode(self, x):
        feat_list = []
        for i in range(self.encoder.num_stage):
            x = self.encoder.downsample_layers[i](x)
            x = self.encoder.stages[i](x)
            if i >= self.scale_skip:
                feat_list.append(x)
        return feat_list

    def decode(self, feat_list, noise_label):
        q = repeat(self.query_embed.weight, 'q c -> b q c', b=feat_list[0].shape[0])

        # noise_label: [B*N_dn_group, num_classes]
        q_dn = rearrange(noise_label, 'N (q p) -> q N p', q=self.num_queries)
        q_dn = self.cls_encoder(q_dn).transpose(0,1) # [B*N_dn_group, num_queries, C]

        pos_emb = [rearrange(self.pos_encoder(x), 'b h w c -> b (h w) c') for x in feat_list]
        feat_list = [trans(rearrange(x, 'b h w c -> b (h w) c')) for x, trans in zip(feat_list, self.feats_trans)]

        pos_emb = [repeat(x, 'b ... -> (n b) ...', n=self.num_denoise_group) for x in pos_emb]
        feat_list = [repeat(x, 'b ... -> (n b) ...', n=self.num_denoise_group) for x in feat_list]

        out = self.decoder(torch.cat([q, q_dn], dim=0), feat_list, pos=pos_emb)
        return out

    def forward(self, x, noise_label):
        feat_list = self.encode(x)
        pred = self.decode(feat_list, noise_label)  # [B*N_dn_group, num_queries, C]

        pred = self.final_norm(pred)
        pred = self.cls_head(pred.transpose(0,1))  # [num_queries, B*N_dn_group, ceil(num_classes/num_queries)]
        pred = pred.transpose(0,1).flatten(1,2)[:, :self.num_classes] # [B*N_dn_group, num_classes]

        B = pred.shape[0]//self.num_denoise_group
        return pred[:B], pred[B:]


def build_mlformer(model_name, d_model=512, pretrained_encoder=True, dec_head_dim=32, dec_layers=6, encoder_ckpt=None,
                   num_queries=50, num_classes=1000, scale_skip=1, T=4.):
    encoder = create_model(model_name, pretrained=pretrained_encoder)
    dec_layer = lambda : MSDecoderLayer(d_model, head_dim=dec_head_dim)
    decoder = MSDecoder(dec_layer, dec_layers, norm=RMSNorm(d_model))

    if encoder_ckpt is not None:
        encoder.load_state_dict(torch.load(encoder_ckpt))

    model = MLFormerDenoise(encoder, decoder, num_queries=num_queries, num_classes=num_classes,
                     d_model=d_model, scale_skip=scale_skip, T=T)

    return model

def mlformer_H(num_classes, scale_skip=2, T=4.):
    return build_mlformer('caformer_b36', d_model=768, num_queries=60,
                          scale_skip=scale_skip, num_classes=num_classes, T=T)

def mlformer_L(num_classes, scale_skip=2, T=4.):
    return build_mlformer('caformer_b36', d_model=640, num_queries=60, dec_layers=3,
                          scale_skip=scale_skip, num_classes=num_classes, T=T)

def mlformer_M(num_classes, scale_skip=2, T=4.):
    return build_mlformer('caformer_m36', d_model=512, num_queries=60, dec_layers=5,
                          scale_skip=scale_skip, num_classes=num_classes, T=T)