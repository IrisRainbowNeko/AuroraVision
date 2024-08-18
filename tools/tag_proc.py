import os
import csv


table_file = ''
tag_map_file = ''
save_file = ''

tag_mapper = {}
with open(tag_map_file, 'r', encoding='utf-8') as f:
    for line in f.read().split('\n'):
        raw, tag = line.split(' -> ')
        tag_mapper[raw] = tag

datas = []
with open(table_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue

        img_id = row[0]
        tags = [tag_mapper[x] for x in row[7].split(' ')]
        datas.append((img_id, ','.join(tags)))

with open(save_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(datas)