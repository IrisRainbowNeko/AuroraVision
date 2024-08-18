import argparse
import os.path

import torch
import torchvision
from PIL import Image
from tqdm.auto import tqdm
from rainbowneko.ckpt_manager import CkptManagerPKL

from mlneko.models.ml_caformer_sparse import mlformer_L

# from torchanalyzer import ModelIOAnalyzer, FlowViser

types_support = ('bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'tiff', 'webp')

class Infer:
    def __init__(self, ckpt_path, tags_path, num_classes=6554, device='cuda'):
        self.model = mlformer_L(num_classes=num_classes, T=1., num_queries=200, ex_tokens=4, grad_checkpointing=False)
        manager = CkptManagerPKL(saved_model=({'model': '', 'trainable': False},))
        manager.load_ckpt_to_model(self.model, ckpt_path)
        self.model = self.model.to(device)
        self.device = device

        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.229, 0.224, 0.225]),
        ])

        with open(tags_path, encoding='utf8') as f:
            self.tags = [x.split(' -> ')[0] for x in f.read().split('\n')]

    def load_image(self, path, target_size=448):
        img = Image.open(path)
        w,h = img.size
        #ratio = math.sqrt(target_size/(w*h))
        ratio = target_size/min(w,h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)

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
        pred_tags = [(self.tags[i], x.item()) for i,x in enumerate(pred.view(-1)) if x > thr]
        pred_tags = sorted(pred_tags, key=lambda x: x[1], reverse=True)
        return pred_tags

    def infer(self, path, thr=0.5):
        if os.path.isdir(path):
            pred_dict = {}
            file_list = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(types_support):
                        file_list.append(os.path.join(root, file))

            for path in tqdm(file_list):
                pred_tags = self.infer_one(path, thr)
                pred_dict[path] = pred_tags
            return pred_dict
        else:
            pred_tags = self.infer_one(path, thr)
            return pred_tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="anime tagger")
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--tags_path", type=str, default='tags_danbooru_v2.csv')
    parser.add_argument('--num_classes', nargs='+', type=int, default=[8092, 2751])
    parser.add_argument('--img', type=str, default='')
    args = parser.parse_args()

    infer = Infer(args.ckpt_path, args.tags_path, num_classes=args.num_classes)
    pred_tags = infer.infer(args.img)
    for tag, prob in pred_tags:
        print(tag, prob)