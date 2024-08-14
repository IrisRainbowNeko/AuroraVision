import torch
from rainbowneko.train.data import ImageLabelDataset
from rainbowneko.train.data.source import ImageLabelSource
from typing import Dict, Any, Tuple
from PIL import Image
import lmdb
import struct
import io
import csv

class LmdbDanbooruSource(ImageLabelSource):
    def __init__(self, img_root, label_file, num_classes=6554, image_transforms=None, bg_color=(255, 255, 255), repeat=1, **kwargs):
        super(LmdbDanbooruSource, self).__init__(img_root, label_file, image_transforms=image_transforms, bg_color=bg_color,
                                               repeat=repeat)

        self.env = lmdb.open(img_root, readonly=True, max_readers=2048)
        self.num_classes = num_classes

    def load_image(self, img_id: int) -> Dict[str, Any]:
        with self.env.begin(write=False) as txn:
            # 读取指定的键
            specific_key = struct.pack('q', img_id)
            value = txn.get(specific_key)
            buffer = io.BytesIO(value)
        txn.abort()

        # 使用 Image.open() 从 BytesIO 对象中读取图像
        image = Image.open(buffer)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas.convert("RGB")
        return {'image': image}

    def _load_img_ids(self, label_dict):
        return list(label_dict.keys()) * self.repeat

    def _load_label_data(self, label_file: str):
        label_dict = {}
        self.size_dict = {}
        zero_count=0
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue

                img_id, tags, w, h = row
                if len(tags)==0:
                    zero_count+=1
                else:
                    label_dict[int(img_id)] = [int(x) for x in tags.split(' ')]
                    self.size_dict[int(img_id)] = (int(w), int(h))
        print('zero', zero_count)
        return label_dict

    def get_image_size(self, img_id: int) -> Tuple[int, int]:
        return self.size_dict[img_id]

    def load_label(self, img_id: str) -> Dict[str, Any]:
        label = self.label_dict.get(img_id, None)
        label_t = torch.zeros(self.num_classes, dtype=torch.float32)
        label_t[label] = 1.
        return {'label': label_t}