import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml
from PIL import Image
import numpy as np
import io
from pathlib import Path
from yaml import CLoader as Loader
from torch.utils import data
import csv


# Single File Dataset (.sfd) format
# All numbers are little endian
#
# struct Sfd {
#     uint32_t length;                // number of images
#     uint64_t pos[length];           // position of images, as offset from the beginning
#     struct Img[length];             // images
# }
#
# struct Img {
#     uint32_t size;                  // image size in bytes
#     uint8_t data[size];             // image file data
# }


class Sfd:
    def __init__(self, filename, workers=0):
        self.filename = filename
        self.f = open(filename, 'rb')
        self.length = int.from_bytes(self.f.read(4), 'little', signed=False)
        self.positions = np.frombuffer(self.f.read(self.length * 8), dtype=np.uint64, count=self.length)

        # for multithreading
        self.files = [open(filename, 'rb') for i in range(workers)]

    def __getitem__(self, item):
        pos = self.positions[item]
        worker_info = data.get_worker_info()
        if worker_info is None:
            f = self.f
        else:
            f = self.files[worker_info.id]
        f.seek(pos, 0)
        size = int.from_bytes(f.read(4), 'little', signed=False)
        return Image.open(io.BytesIO(f.read(size)))

    def __len__(self):
        return self.length

    def __del__(self):
        self.f.close()

        # for multithreading
        [f.close() for f in self.files]

    @staticmethod
    def create(filename, img_list, img_root='', new_size=None):
        length = len(img_list)

        positions = np.empty([length], dtype=np.uint64)
        sizes = np.empty([length], dtype=np.uint64)
        cur_pos = 4 + 8 * length

        for i, img in enumerate(img_list):
            size = Path(os.path.join(img_root, img)).stat().st_size
            positions[i] = cur_pos
            sizes[i] = size
            cur_pos += 4 + size

        with open(filename, 'wb') as out_f:
            out_f.write(length.to_bytes(4, 'little', signed=False))
            out_f.write(positions.tobytes())
            for i, img in enumerate(img_list):
                if not new_size:
                    with open(os.path.join(img_root, img), 'rb') as img_f:
                        image_data = img_f.read()
                else:
                    img = Image.open(os.path.join(img_root, img))
                    width, height = img.size
                    resize_factor = max(width, height) / new_size
                    if resize_factor > 1:
                        img = img.resize(size=(round(width / resize_factor), round(height / resize_factor)))
                    img_byte_stream = io.BytesIO()
                    img.save(img_byte_stream, format='JPEG')
                    image_data = img_byte_stream.getvalue()
                out_f.write(len(image_data).to_bytes(4, 'little', signed=False))
                out_f.write(image_data)
                if i % 10 == 0:
                    print(f'\r{i + 1}/{len(img_list)}', end=' ')

# def create_isic_2020():
# from isic_classification_dataset import ISIC

# data_root = '/nas/softechict-nas-1/sallegretti/data/ISIC/SIIM-ISIC'
#
# isic_train = ISIC(split_name='training_v1_2020', classes=[[0], [1]])
# Sfd.create(os.path.join(data_root, 'train.sfd'), isic_train.imgs)
#
# isic_val = ISIC(split_name='val_v1_2020', classes=[[0], [1]])
# Sfd.create(os.path.join(data_root, 'val.sfd'), isic_val.imgs)
#
# isic_test = ISIC(split_name='test_v1_2020', classes=[[0], [1]])
# Sfd.create(os.path.join(data_root, 'test.sfd'), isic_test.imgs)


def create_isic2020_test_sfd():
    data_root = '/nas/softechict-nas-1/sallegretti/data/ISIC/SIIM-ISIC'

    img_list = []
    with open(os.path.join(data_root, 'test.csv'), 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == 'image_name':
                continue
            img_list.append(row[0] + '.jpg')

    Sfd.create(filename=os.path.join(data_root, 'submission_test.sfd'),
               img_list=img_list,
               img_root=os.path.join(data_root, 'images'))


def create_isic2020_sfd(new_size=None, prefix=''):
    data_root = '/nas/softechict-nas-1/sallegretti/data/ISIC/SIIM-ISIC'

    csv_dict = {
        '2k20_train_partition.csv': 'train.sfd',
        '2k20_validation_partition.csv': 'val.sfd',
        '2k20_test_partition.csv': 'test.sfd',
    }

    img_list = []
    for key, value in csv_dict.items():
        with open(os.path.join(data_root, key), 'r') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0] == 'image_name':
                    continue
                img_list.append(row[0] + '.jpg')

        Sfd.create(filename=os.path.join(data_root, f'{prefix}{value}'),
                   img_list=img_list,
                   img_root=os.path.join(data_root, 'images'),
                   new_size=new_size)


if __name__ == '__main__':
    #create_isic2020_test_sfd()
    create_isic2020_sfd(new_size=1024, prefix='small')

    img_root = '/nas/softechict-nas-1/sallegretti/data/ISIC/SIIM-ISIC'

    sfd = Sfd(os.path.join(img_root, 'train.sfd'))
    img = sfd[0]
    img.show()

    # create_isic_2020()

    #
    # with open('/nas/softechict-nas-2/fpollastri/data/ISIC_dataset/Task_3/isic.yml', 'r') as stream:
    #     try:
    #         d = yaml.load(stream, Loader=Loader)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #
    # img_list = [img['location'] for img in d['images']]
    # imgcat_path = f'/nas/softechict-nas-2/sallegretti/dataset/isic.sfd'

    # Sfd.create(imgcat_path, img_list, img_root)

    # img_cat = Sfd(imgcat_path)
    # img = img_cat[5]
    # img.show()
