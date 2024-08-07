from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv

with open(r'F:\realbooru_full\tags.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    tags = [row[0].replace('_', ' ') for row in reader][1:]

# 初始化词干提取器
ps = PorterStemmer()

# 词干化函数
def stem_tag(tag):
    tokens = word_tokenize(tag)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# 构建词干化后的标签字典
stemmed_tags = set()
for tag in tags:
    stemmed = stem_tag(tag)
    stemmed_tags.add(stemmed)
    # if stemmed not in stemmed_tags:
    #     for tag2 in stemmed_tags:
    #         print(stemmed, tag2, fuzz.ratio(tag, tag2))
    #
    #     stemmed_tags.add(stemmed)

# 获取过滤后的标签
filtered_tags = list(stemmed_tags)

# print("原始标签：", tags)
# print("过滤后的标签：", filtered_tags)

with open(r'F:\realbooru_full\tags_token.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_tags))