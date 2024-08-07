import os
import lmdb
from PIL import Image, UnidentifiedImageError
from multiprocessing import Pool, cpu_count, Manager, Queue
import io
import struct
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# 定义图像处理函数
def is_aspect_ratio_valid(width, height):
    aspect_ratio = width / height
    return 0.25 <= aspect_ratio <= 4


def resize_image(img, size):
    width, height = img.size
    ratio = min(width, height) / size
    new_size = (int(width / ratio), int(height / ratio))
    return img.resize(new_size, Image.LANCZOS)


def convert_to_webp(img):
    buffer = io.BytesIO()
    img.save(buffer, format='webp')
    return buffer.getvalue()


def process_image(root, filename, queue):
    img_path = os.path.join(root, filename)

    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if is_aspect_ratio_valid(img.width, img.height):
                img = resize_image(img, 640)
                img_byte = convert_to_webp(img)
                key = struct.pack('q', int(filename.split('.')[0]))
                queue.put((key, img_byte))
                return 1  # 图像处理成功
    except (IOError, ValueError, UnidentifiedImageError) as e:
        print(f"Error processing {img_path}: {e}")
        return 0  # 图像处理失败

def writer(queue, env_path, batch_size=1000):
    env = lmdb.open(env_path, map_size=1024 ** 4, max_spare_txns=100)
    c=0
    with env.begin(write=True) as txn:
        while True:
            key, img_byte = queue.get()  # 从队列中获取数据
            if key is None:
                txn.commit()
                print('break')
                break  # 如果收到None，则结束写入进程

            txn.put(key, img_byte)
            c+=1
            if c % batch_size == 0:
                print('save')
                txn.commit()
                txn = env.begin(write=True)
                print(c)
    env.close()

def get_saved_ids(env_path):
    id_set = set()

    with lmdb.open(env_path, readonly=True) as env:
        with env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                id_set.add(struct.unpack('q', key)[0])
    return id_set

def main(env_path, images_dir):
    num_processes = cpu_count()  # 使用CPU的核心数

    queue = Manager().Queue()
    writer_process = Pool(processes=1, initializer=writer, initargs=(queue, env_path))

    ids = get_saved_ids(env_path)
    print('saved:', len(ids))

    # 收集所有有效的文件路径
    files = [(root, name, queue) for root, dirs, files in os.walk(images_dir) for name in files
             if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')) and int(name.split('.')[0]) not in ids]
    # files = [(part, filename, queue) for part in range(10000)
    #          for filename in os.listdir(os.path.join('images_part', str(part).zfill(4)))
    #          if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')) and int(filename.split('.')[0]) not in ids]


    total_files = len(files)

    # 使用 tqdm 显示进度
    pbar = tqdm(total=total_files, desc="Processing Images")

    def update_pbar(result):
        pbar.update(1)

    with Pool(processes=num_processes) as pool:
        for file in files:
            pool.apply_async(process_image, file, callback=update_pbar)
        pool.close()
        pool.join()

        #queue.put((None, None))

    # 完成后关闭进度条
    pbar.close()


if __name__ == "__main__":
    #env_path = './images_2023'
    env_path = '/home/public/dzy/images_2023'
    images_dir = 'images_new'
    main(env_path, images_dir)