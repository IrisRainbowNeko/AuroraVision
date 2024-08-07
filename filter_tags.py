import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import PorterStemmer
import csv
import spacy

ps = PorterStemmer()
nlp = spacy.load('en_core_web_sm')

# 假设已经下载了nltk的数据包，如果没有，请取消注释下面的行
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# doc = nlp('glowing female')
# lemmas = ' '.join([token.lemma_ for token in doc])
# print(lemmas)
# 0/0

def is_content_related(tag):
    """
    判断一个由多个词组成的标签是否与图像内容相关
    :param tag: 一个由多个词组成的标签
    :return: 如果标签与内容相关返回True，否则返回False
    """
    words = word_tokenize(tag)
    tagged_words = pos_tag(words)

    # 检查每个词的词性
    for word, tag in tagged_words:
        # 如果标签中包含名词或动词，则认为与内容相关
        if (tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ') or tag.startswith('RB')
                or tag.startswith('IN')):
            return True
    return False


def stem_tag(tag):
    doc = nlp(tag)
    stemmed_tokens = [token.lemma_ for token in doc]
    return ' '.join(stemmed_tokens)

# 示例标签列表，包含由多个词组成的标签
meta_tags='watermark,cosplay,amateur,uncensored,highres,slut,beautiful,lowres,monochrome,sfw,delicious,bitch,black and white'.split(',')
# exclude_tags = ('what,18,<3,cc,v,:o,300 pamela anderson hq picture [ set 1 ],:>=,?,...,!,:),;),!?,!!,0_0,3:,??,:i,;3,._.,:d,;d,'
#                 ':3,^_^,>_<,^^^,:<,:p,:q,@_@,+_+,=_=,o_o,:t,|_|,+++,:>,w,>:),:/,;o,:o,d:,=3,:|,\m/,xd,;p,>:(,:>=,;q,>o,\||/,^o^,'
#                 'u_u,x_x,:x,x,\o/,c:,>o<,<o>_<o>').split(',')

exclude_tags = ('what,18,<3,cc,v,300 pamela anderson hq picture [ set 1 ],?,...,!,!?,!!,??,^^^,+++').split(',')

tag_map={
    '34dd': '34d',
    '34ddd': '34d',
    '1boy1girl': 'male/female',
    'grey legwear': 'gray legwear',
    'ass jiggle': 'jiggle ass',
    'breast jiggle': 'jiggle breast',
    'nose pierce': 'pierce nose',
    'self - shoot': 'self shoot',
    'pantie around one leg': 'pantie around leg',
    'upside - down': 'upside down',
    'multi - colored hair': 'multicolor hair',
    'pull hair': 'hair pull',
    'spread leg': 'leg spread',
    'kiss penis': 'penis kiss',
    'spread ass': 'ass spread',
    'breast squeeze': 'squeeze breast',
    'ass slap': 'slap ass',
    'ass clap': 'clap ass',
    'solo female': 'female solo',
    'blond hair': 'blonde hair',
    'large areola': 'big areolae',
    'tattoo on arm': 'arm tattoo',
    'outside': 'outdoors',
    'gigantic breast': 'huge breast',
    'ass shake': 'shake ass',
    'tattoo on leg': 'leg tattoo',
    'ride dildo': 'ride on dildo',
    'penis lick': 'lick penis',
    'tattoo on back': 'back tattoo',
    'dildo riding': 'ride on dildo',
    'short short': 'short',
    'twerke': 'twerk',
    'large breast': 'big breast',
    'hold breast': 'breast hold',
    'crossdresset': 'crossdresse',
    'thighband': 'thigh band',
    'eye close': 'close eye',
    'dd cup': 'd cup',
    'ddd cup': 'd cup',
    'ass grab': 'grab ass',
    'completely nude': 'nude',
    'fully nude': 'nude',
    'large nipple': 'big nipple',
    'massive breast': 'huge breast',
    'crop jacket': 'croptop jacket',
    'large lip': 'big lip',
    'mature woman': 'mature female',
    'leotard aside': 'leotard',
    'completely nude female': 'nude female',
    'do yoga': 'yoga',
    'large boob': 'big boob',
    'large areolae': 'big areolae',
    'naked apron': 'nude apron',
    'breast grab': 'grab breast',
    'masturbation': 'masturbate',
    'areola': 'areolae',
    'vaginal sex': 'vaginal',
    'gigantic penis': 'huge penis',
    'hairy armpit': 'armpit hair',
    'shemale with male': 'male on shemale',
    'large ass': 'big ass',
    'thong aside': 'thong',
    'massive ass': 'huge ass',
    'one elbow glove': 'elbow glove',
    'large tit': 'big tit',
    'bent over': 'bend over',
    'bisexual ( female )': 'bisexual',
    'public nudity': 'nudity',
    'nudity': 'nude',
    'mom': 'mother',
    'bubble butt': 'bubble ass',
    'latina': 'latin',
    'belly pierce': 'navel pierce',
    'bath': 'bathe',
    'muscle': 'muscular',
    'male / female': 'male/female',
    'black & white': 'black and white',
    'dye hair': 'color hair',
    'close - up': 'closeup',
    'couch': 'sofa',
    'oral': 'oral sex',
    'naked': 'nude',
    'naked male': 'nude male',
    'naked female': 'nude female',

    'flip hair': 'hair flip',
    'tie hair': 'hair tie',
    'leg hold': 'hold leg',
    'holstere': 'holster',
}


def filter_content_tags(tags, raw_tags):
    """
    使用NLP方法筛选出描述图像内容相关的标签
    :param tags: 包含所有标签的列表
    :return: 筛选后的标签列表
    """
    filtered_tags = []
    filtered_raw_tags = []
    for tag, raw_tag in zip(tags, raw_tags):
        tag = stem_tag(tag)
        if tag in tag_map:
            tag = tag_map[tag]
        if is_content_related(tag) and tag not in filtered_tags:
            filtered_tags.append(tag)
            filtered_raw_tags.append(raw_tag)
        else:
            filtered_tags.append(tag)
            filtered_raw_tags.append(raw_tag)
            print(tag, raw_tag)
    return filtered_tags, filtered_raw_tags

def filter_realbooru(path=r'E:\dataset\tagger\tags_realbooru.csv'):
    metadatas = []
    tags = []
    raw_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if i>=2500:
                break

            if row[1]=='metadata':
                metadatas.append(row[0])
            elif row[1]=='general':
                tag = row[0]
                if tag in exclude_tags:
                    continue

                raw_tags.append(tag)
                tags.append(tag.replace('_', ' '))
        #tags = [row[0].replace('_', ' ') for row in reader][1:2500]
    return metadatas, tags + meta_tags, raw_tags

def filter_3dbooru(path=r'E:\dataset\tagger\tags_3dbooru.csv'):
    metadatas = []
    tags = []
    raw_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if i>=2500:
                break

            if row[2]=='3':
                metadatas.append(row[0])
            elif row[2]=='0':
                tag = row[1]
                if tag in exclude_tags:
                    continue

                raw_tags.append(tag)
                tags.append(tag.replace('_', ' '))
        #tags = [row[0].replace('_', ' ') for row in reader][1:2500]
    return metadatas, tags, raw_tags

def filter_danbooru(path=r'E:\dataset\tagger\tags_danbooru.csv'):
    metadatas = []
    tags = []
    raw_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue

            tag = row[0]
            if tag in exclude_tags:
                continue

            raw_tags.append(tag)
            tags.append(tag.replace('_', ' '))
        #tags = [row[0].replace('_', ' ') for row in reader][1:2500]
    return metadatas, tags, raw_tags

metadatas, tags, raw_tags = filter_danbooru()
# 筛选标签
content_tags, raw_tags = filter_content_tags(tags, raw_tags)
print('len', len(content_tags), len(raw_tags))
print(','.join(content_tags))
print(','.join(raw_tags))

with open(r'E:\dataset\tagger\tags_danbooru_prune.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(list(dict.fromkeys(content_tags))))

tags_mapper = [f'{raw_tag} -> {tag}' for tag, raw_tag in zip(content_tags, raw_tags)]

# 输出筛选后的标签
with open(r'E:\dataset\tagger\tags_danbooru_map.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(tags_mapper))
