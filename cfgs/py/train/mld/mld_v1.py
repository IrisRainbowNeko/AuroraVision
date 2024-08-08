from functools import partial

import torch
from torch import nn
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from torchmetrics.regression import KLDivergence, MeanAbsoluteError
from rainbowneko.train.data.trans import MixUP

from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.train.data import RatioBucket
from rainbowneko.train.data.source import IndexSource, ImageLabelSource
from rainbowneko.train.loss import LossContainer, LossGroup
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.ckpt_manager import CkptManagerPKL
from rainbowneko.parser import make_base, CfgWDModelParser
from rainbowneko.train.loggers import CLILogger, TBLogger

from mlneko.models.ml_caformer_sparse import mlformer_L
from mlneko.loss import AsymmetricKLLoss, EntropyLoss
from mlneko.data import LmdbDanbooruSource

from cfgs.py.train import train_base, tuning_base
from timm.models.metaformer import caformer_b36

num_classes = 6554

class KLDivergenceSigmoid(KLDivergence):
    def forward(self, p, q):
        return super().forward(p.sigmoid(), q.sigmoid())

def make_cfg():
    dict(
        _base_=make_base(train_base, tuning_base)+[],

        exp_dir=f'exps/mld_v1.1',
        mixed_precision='fp16',
        allow_tf32=True,

        model_part=CfgWDModelParser([
            dict(
                lr=3e-4,
                layers=[''],  # train all layers
            )
        ], weight_decay=1e-2),

        ckpt_manager=CkptManagerPKL(_partial_=True, saved_model=(
            {'model':'model', 'trainable':False},
        )),

        train=dict(
            train_epochs=2,
            workers=4,
            max_grad_norm=None,
            save_step=10000,
            gradient_accumulation_steps=4,

            loss=LossContainer(_partial_=True, loss=AsymmetricKLLoss(
                weight_file='/dataset/dzy/danbooru_2023/tags_danbooru_weight.npy',
                focal_no_grad=True
            )),
            # loss=LossGroup(partial_=True, loss_list=[
            #     LossContainer(loss=AsymmetricLoss()),
            #     LossContainer(loss=EntropyLoss(), weight=0.1),
            # ]),

            optimizer=torch.optim.AdamW(_partial_=True, weight_decay=0),

            scale_lr=False,
            scheduler=dict(
                name='cosine',
                num_warmup_steps=5000,
            ),
            metrics=MetricGroup(metric_dict=dict(
                kld=MetricContainer(KLDivergenceSigmoid()),
                l1=MetricContainer(MeanAbsoluteError()),
            )),
        ),

        model=dict(
            name='mld-L_danbooru',
            #wrapper=SingleWrapper(_partial_=True, model=mlformer_L(num_classes=num_classes), key_map={'pred':'0', 'pred_all':'1'})
            wrapper=SingleWrapper(_partial_=True, model=mlformer_L(num_classes=num_classes, T=1.))
        ),

        data_train=dict(
            dataset1=ImageLabelDataset(_partial_=True, batch_size=16, loss_weight=1.0,
                source=dict(
                    data_source1=LmdbDanbooruSource(
                        img_root="/dataset/dzy/danbooru_2023_lmdb",
                        label_file="/dataset/dzy/danbooru_2023/caption_train.csv",
                        num_classes=num_classes,
                        image_transforms=torchvision.transforms.Compose([
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomVerticalFlip(),
                            torchvision.transforms.RandomRotation(25, interpolation=InterpolationMode.BILINEAR),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229, 0.224, 0.225]),
                        ])
                    ),
                ),
                batch_transform=MixUP(num_classes=num_classes, alpha=0.4),
                bucket=RatioBucket.from_files(
                    target_area=448*448,
                    num_bucket=16,
                    pre_build_bucket='/dataset/dzy/danbooru_2023/bucket-448,448-train.pkl',
                ),
            )
        ),

        logger=[
            partial(CLILogger, out_path='train.log', log_step=50),
            partial(TBLogger, out_path='tb/', log_step=10)
        ],

        evaluator=partial(Evaluator,
            interval=5000,
            metric=MetricGroup(metric_dict=dict(
                acc=MetricContainer(MultilabelAccuracy(num_labels=num_classes)),
                f1=MetricContainer(MultilabelF1Score(num_labels=num_classes)),
            )),
            dataset=dict(
                dataset1=partial(ImageLabelDataset, batch_size=16, loss_weight=1.0,
                    source=dict(
                        data_source1=LmdbDanbooruSource(
                            img_root="/dataset/dzy/danbooru_2023_lmdb",
                            label_file="/dataset/dzy/danbooru_2023/caption_test.csv",
                            num_classes=num_classes,
                            image_transforms=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229, 0.224, 0.225]),
                            ])
                        ),
                    ),
                    bucket=RatioBucket.from_files(
                        target_area=448*448,
                        num_bucket=16,
                        pre_build_bucket='/dataset/dzy/danbooru_2023/bucket-448,448-test.pkl',
                    ),
                )
            )
        ),

    )
