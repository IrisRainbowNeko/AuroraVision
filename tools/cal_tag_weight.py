import csv
import numpy as np
import torch

count_path = r'E:\dataset\tagger\tags_count_danbooru_2023.csv'
tag_map_path = r'E:\dataset\tagger\tags_danbooru_map_v2.csv'

out_path_pos = r'E:\dataset\tagger\tags_danbooru_weight_v2.2-cb_pos.npy'
out_path_neg = r'E:\dataset\tagger\tags_danbooru_weight_v2.2-cb_neg.npy'
out_path_txt = r'E:\dataset\tagger\tags_danbooru_weight_v2.2-cb.txt'

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

# w1 = np.log(counts)
# counts_w = w1/counts
# counts_w = counts_w/np.max(counts_w)
# weight = counts_w/np.sqrt(counts_w[0]*counts_w[-1])

num_images = 7350144
def log_inv():

    p = counts/num_images
    weight = np.log(1/p)
    weight_mid = np.sqrt(weight[0]*weight[-1])/3
    weight = weight/weight_mid

    p_n = (num_images-counts)/num_images
    weight_n = np.log(1/p_n)
    weight_n = weight_n/weight_mid
    return weight, weight_n


def blanace_loss(b=0.999):
    weight = (1 - b) / (1 - b ** (counts / 10))
    weight_mid = np.sqrt(weight[0] * weight[-1])
    weight = weight / weight_mid
    #weight = weight ** 0.7

    counts_n = num_images - counts
    weight_n = (1 - b) / (1 - b ** (counts_n / 2500))
    weight_n = weight_n / weight_mid
    #weight_n = weight_n ** 0.7
    return weight, weight_n

def inv_sqrt():

    p = counts/num_images
    weight = np.sqrt(1/p)
    weight_mid = np.sqrt(weight[0]*weight[-1])
    weight = weight/weight_mid

    p_n = (num_images-counts)/num_images
    weight_n = np.sqrt(1/p_n)
    weight_n = weight_n/weight_mid

    #weight_n = np.sqrt(weight_n)+0.05
    return weight, weight_n

weight, weight_n = blanace_loss()

np.save(out_path_pos, weight)
np.save(out_path_neg, weight_n)

with open(out_path_txt, 'w', encoding='utf-8') as f:
    for w, wn in zip(weight,weight_n):
        f.write(f'{w:.6f}, {wn:.6f}\n')