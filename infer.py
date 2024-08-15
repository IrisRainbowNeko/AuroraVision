import torch
from rainbowneko.ckpt_manager import CkptManagerPKL
from mlneko.models.ml_caformer_sparse import mlformer_L
import torchvision
from PIL import Image

from torchanalyzer import ModelIOAnalyzer, FlowViser

class Infer:
    def __init__(self, ckpt_path, tags_path, num_classes=6554, device='cuda'):
        self.model = mlformer_L(num_classes=num_classes, T=1., num_queries=200, ex_tokens=4)
        manager = CkptManagerPKL(saved_model=(
            {'model': '', 'trainable': False},
        ))
        manager.load_ckpt_to_model(self.model, ckpt_path)
        self.model = self.model.to(device)
        self.device = device

        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229, 0.224, 0.225]),
        ])

        with open(tags_path, encoding='utf8') as f:
            self.tags = [x.split(' -> ')[0] for x in f.read().split('\n')]

    def load_image(self, path):
        img = Image.open(path)
        img = self.trans(img)
        return img.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer_one(self, path, thr=0.5):
        img = self.load_image(path)

        # analyzer = ModelIOAnalyzer(self.model)
        # with torch.cuda.amp.autocast():
        #     info = analyzer.analyze(img)
        # FlowViser().show(self.model, info)

        with torch.cuda.amp.autocast():
            pred, feat_loss = self.model(img)

        pred = pred.cpu().view(-1)
        print(feat_loss)
        pred_tags = [(self.tags[i], x.item()) for i,x in enumerate(pred.view(-1)) if x > thr]
        return pred_tags

if __name__ == '__main__':
    infer = Infer('exps/mld_v2.1/ckpts/mld-L_danbooru-150000.ckpt', '/dataset/dzy/danbooru_2023/tags_danbooru_map_v2.csv', num_classes=[8092, 2751])
    #infer = Infer('exps/mld_v2.1-cb/ckpts/mld-L_danbooru-30000.ckpt', '/dataset/dzy/danbooru_2023/tags_danbooru_map_v2.csv', num_classes=[8092, 2751])
    #infer = Infer('exps/mld_v1.1-3/ckpts/mld-L_danbooru-220000.ckpt', '/dataset/dzy/danbooru_2023/tags_danbooru_map.csv')
    #pred_tags = infer.infer_one('imgs/00001-0-standing.png')
    pred_tags = infer.infer_one('imgs/2.jpg')
    pred_tags = sorted(pred_tags, key=lambda x: x[1], reverse = True)
    for tag, prob in pred_tags:
        print(tag, prob)