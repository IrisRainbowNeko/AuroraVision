import csv
import numpy as np

count_path = r'E:\dataset\tagger\tags.csv'
tag_map_path = r'E:\dataset\tagger\tags_danbooru_map.csv'

out_path = r'E:\dataset\tagger\tags_danbooru_weight.npy'

tag2id = {}

with open(tag_map_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.read().split('\n')):
        map = line.split(' -> ')
        tag2id[map[0]] = i

counts = np.zeros(len(tag2id))
with open(count_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i == 0:
            continue

        tag, count = row
        if tag in tag2id:
            counts[tag2id[tag]] = count

w1 = np.log(counts)
counts_w = w1/counts
counts_w = counts_w/np.max(counts_w)
weight = counts_w/np.sqrt(counts_w[0]*counts_w[-1])

np.save(out_path, weight)

# with open(out_path, 'w', encoding='utf-8') as f:
#     for w in weight:
#         f.write(f'{w}\n')