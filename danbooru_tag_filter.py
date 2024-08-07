import csv
import json

tag_path = 'tags.csv'
tag_info_path = 'tags.json'
out_path = 'tags_c0.csv'
min_count = 1000

tags_count = []
with open(tag_path, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            continue

        tags_count.append(row)

tags_cls = {}
with open(tag_info_path, 'r', encoding='utf-8') as file:
    for line in file:
        info = json.loads(line.rstrip('\n'))
        tags_cls[info['name']] = info['category']

print(tags_cls['1girl'], type(tags_cls['1girl']))
with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['tag', 'count']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)

    for tag, count in tags_count:
        if int(count) >= min_count and tags_cls[tag]==0:
            writer.writerow((tag, count))