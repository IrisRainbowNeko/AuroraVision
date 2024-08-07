from collections import Counter
import multiprocessing
import os
import csv


def process_text(text):
    # 将文本转换为小写
    text = text.lower()
    words = text.split(' ')
    return Counter(words)


def count_words_parallel(file_path):
    # 获取CPU核心数
    num_cores = multiprocessing.cpu_count()

    # 创建进程池
    pool = multiprocessing.Pool(processes=num_cores)

    # 用于存储所有Counter对象的列表
    counters = []

    # 打开文件并按块读取
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue

            counters.append(pool.apply_async(process_text, (row[1],)))

    # 关闭进程池
    pool.close()
    pool.join()

    # 合并所有Counter对象
    final_counter = Counter()
    for counter in counters:
        final_counter.update(counter.get())

    return final_counter


def write_results_to_csv(counter, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Count'])  # 写入表头

        for word, count in counter.most_common():  # 按频率降序排列
            writer.writerow([word, count])

# 使用示例
if __name__ == '__main__':
    file_path = 'caption.csv'  # 替换为你的大文本文件路径
    output_file_path = 'tags.csv'

    result = count_words_parallel(file_path)

    write_results_to_csv(result, output_file_path)
