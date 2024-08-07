import csv
import lmdb
import struct
from tqdm import tqdm
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# 4090-2
tag_map_path = '/dataset/dzy/danbooru_2023/tags_danbooru_map.csv'
caption_path = '/dataset/dzy/danbooru_2023/caption_2023.csv'

out_path = '/dataset/dzy/danbooru_2023/caption_prune.csv'

lmdb_path = '/dataset/dzy/danbooru_2023_lmdb'
key_path = '/dataset/dzy/danbooru_2023/all_keys.txt'

print('loading tag map...')
tag_map = {}
with open(tag_map_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.read().split('\n')):
        map = line.split(' -> ')
        tag_map[map[0]] = i

print('loading captions...')
caption_dict = {}
with open(caption_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(tqdm(reader)):
        if i == 0:
            continue

        tags = row[1].split(',')
        tag_ids = [str(tag_map[tag]) for tag in tags if tag in tag_map]
        caption_dict[int(row[0])] = ' '.join(tag_ids)

# -----------
def read_and_process_image(txn, img_id, caption_dict, pbar, total_count):
    try:
        # 从LMDB读取数据
        value = txn.get(struct.pack('q', img_id))
        buffer = io.BytesIO(value)
        image = Image.open(buffer)
        w, h = image.size
        # 将结果存储在共享字典中
        pbar.update(1)
        return img_id, (caption_dict[img_id], w, h)
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return None


def proc_lmdb_info(lmdb_path, key_path, caption_dict):
    env = lmdb.open(lmdb_path, readonly=True)
    txn = env.begin(write=False)

    with open(key_path, encoding='utf8') as f:
        all_keys = f.read().split('\n')
    total_keys = len(all_keys)

    pbar = tqdm(total=total_keys, desc="Processing images")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for key in all_keys:
            if len(key)>0:
                future = executor.submit(
                    read_and_process_image,
                    txn, int(key), caption_dict, pbar, total_keys
                )
                futures.append(future)

        # 使用tqdm显示进度
        with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'tags', 'width', 'height'])

            for future in as_completed(futures):
                result = future.result()
                if result:
                    img_id, (caption, w, h) = result
                    writer.writerow((img_id, caption, w, h))

    txn.abort()
    env.close()

print('loading lmdb...')
proc_lmdb_info(lmdb_path, key_path,caption_dict)

