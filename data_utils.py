
import os
import glob as gb
import cv2
import numpy as np
import torch
from torch.utils import data


def denorm(tensor):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    return np.uint8((tensor*std + mean)*255)

def load_dset(data_path, dataset=None):

    ''' Load images '''

    if dataset == 'train':
        path = os.path.join(data_path, 'data', 'Train_data', 'images')
    elif dataset == 'validation':
        path = os.path.join(data_path, 'data', 'Val_data', 'images')
    elif dataset == 'test':
        path = os.path.join(data_path, 'data', 'Test_data', 'images')
    else:
        print('Specify dataset type: [train, valdiation, test]')

    image_list = gb.glob(os.path.join(path, '*.jpg'))

    return image_list


def get_data(image_list, index):

    try:

        # read one image
        img_p = image_list[index]
        img = cv2.imread(img_p)[...,::-1]
        img = cv2.resize(img, (256, 256))

        # read the correspondent annotation
        name = img_p.split('/')[-1].split('.')[0]
        path_anns = '/'.join(img_p.split('/')[:-2])
        f = open(os.path.join(path_anns, 'anns', name + '.txt'), "r")
        line = f.readline()

        # extract the annotation
        square = line.split(',')[0].split(':')[-1]
        circle = line.split(',')[1].split(':')[-1]
        triangle = line.split(',')[2].split(':')[-1].split('\n')[0]


        shapes = [int(square), int(circle), int(triangle)]
        # we restrict the model to regress on numbers between 0-6.
        shapes = np.clip(shapes, 0, 6)

    except Exception  as e:
        img, square, circle, triangle = None, None, None, None
        print('Exception in get_data()')

    # scale to [0 1] and normalize the input image. We are not that interested about color info
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = (img / 255 - mean) / std
    img = img.astype(np.float32)

    return img, shapes

class custom_dset(data.Dataset):
    def __init__(self, data_path, dataset):
        self.image_list  = load_dset(data_path, dataset)
    def __getitem__(self, index):
        img, anns = get_data(self.image_list,  index)
        return img, anns

    def __len__(self):
        return len(self.image_list)

def collate_fn(batch):
    img, anns = zip(*batch)
    images = []

    squares = []
    circles = []
    triangles = []


    for i in range(len(img)):
        if img[i] is not None:
            a = torch.from_numpy(img[i])
            a = a.permute(2, 0, 1)
            images.append(a)
            # squares
            squares.append(anns[i][0])
            # circles
            circles.append(anns[i][1])
            #triangles
            triangles.append(anns[i][2])

    images = torch.stack(images, 0)
    squares = torch.LongTensor(squares)
    circles = torch.LongTensor(circles)
    triangles = torch.LongTensor(triangles)

    return images, squares, circles, triangles
