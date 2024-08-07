import csv
import json
from tqdm import tqdm


post_path = 'posts.json'
tag_path = 'tags.csv'


# 打开文件
with open(post_path, 'r', encoding='utf-8') as file:
    with open(tag_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'tags', 'rating', 'ext']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for line in tqdm(file):
            metas = json.loads(line.rstrip('\n'))
            writer.writerow((metas['id'], metas['tag_string'], metas['rating'], metas['file_ext']))
