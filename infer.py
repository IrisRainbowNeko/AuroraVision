import torch
from rainbowneko.ckpt_manager import CkptManagerPKL
from mlneko.models.ml_caformer_sparse import mlformer_L
import torchvision
from PIL import Image

class Infer:
    def __init__(self, ckpt_path, tags_path, num_classes=6554, device='cuda'):
        self.model = mlformer_L(num_classes=num_classes, T=1., num_queries=200)
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
            self.tags = f.read().split('\n')

    def load_image(self, path):
        img = Image.open(path)
        img = self.trans(img)
        return img.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer_one(self, path, thr=0.5):
        img = self.load_image(path)
        with torch.cuda.amp.autocast():
            pred = self.model(img)

        pred = pred.cpu().view(-1)
        pred_tags = [(self.tags[i], x) for i,x in enumerate(pred.view(-1)) if x > thr]
        return pred_tags

if __name__ == '__main__':
    infer = Infer('')
    pred_tags = infer.infer_one
    print(sorted(pred_tags, key=lambda x: x[1], reverse = True))