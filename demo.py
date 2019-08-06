import torch
import torch.nn as nn
import os
import numpy as np
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import cv2
from model import ShapeRecognizer
import glob  as gb




def img_toTensor(p):

    '''
    Torchvision pre-trained models use an input tensor, which scaled between [0 1].
    And the normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

    This function reads an image and pre-process it for the task of counting shapes.
    '''

    img = cv2.imread(p)
    img = cv2.resize(img, (256,256))
    im = img[...,::-1]

    # scale [0 1] and normalize the input image. We are not that interested about color info
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    im = (im / 255 - mean) / std
    im = im.astype(np.float32)
    im = torch.from_numpy(im)

    im = im.permute(2,0,1).unsqueeze(0).cuda()
    return img, im


def run_folder(net, lista, out_path):

    '''
    A function to run the network through a folder of images. Results are written in ./data/demo_output .
    '''
    # loop over image paths
    for p in lista:


        start = time.time()
        # read one image from path p.
        img, im = img_toTensor(p)

        # run net
        # We restrict the model to predict numbers between 0-6.
        pred_squar, pred_circ, pred_trgle, sa, ca, ta = net(im)
        _, ps = torch.max(pred_squar, dim=-1)
        _, pc = torch.max(pred_circ, dim=-1)
        _, pt = torch.max(pred_trgle, dim=-1)


        # save on disk, input image along with the attention masks
        con_masks = torch.cat(( sa[0], ca[0], ta[0]), 2).data.cpu().numpy().transpose(1,2,0)*255
        con_masks = cv2.resize(con_masks, (256*3, 256)).astype(np.uint8)
        # apply opencv pseudo-color
        con_masks = cv2.applyColorMap(con_masks, 11)

        #
        concat = np.concatenate((img, con_masks), 1)

        # name comprehensive of the current epoch and followed by the output predictions
        name = p.split('/')[-1].split('.')[0]
        name = name + '___' + str(ps[0].item()) + str(pc[0].item()) + str(pt[0].item()) + '.jpg'
        cv2.imwrite(os.path.join(out_path, name), concat)

        tempo = round((time.time() - start), 4)
        print('Input image of size {}x{} and processing time of {} seconds'.format(img.shape[0], img.shape[1], tempo))
        print('The input image has {} squares, {} circles and {} triangles.'.format(ps.item(), pc.item(), pt.item()))


def run_image(net, img_path):
        '''
        A function to run the network through one single image.
        '''
        # read one image.
        _, im = img_toTensor(img_path)
        # run net
        # We restrict the model to predict numbers between 0-6.
        pred_squar, pred_circ, pred_trgle, sa, ca, ta = net(im)
        _, ps = torch.max(pred_squar, dim=-1)
        _, pc = torch.max(pred_circ, dim=-1)
        _, pt = torch.max(pred_trgle, dim=-1)

        print('The input image has {} squares, {} circles and {} triangles.'.format(ps.item(), pc.item(), pt.item()))

def main():

    '''
    Inference on images resised to 256x256. Reads jpg format images.
    '''

    # Collect arguments (if any)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='path to the model')
    parser.add_argument('-i', '--image', type=str, default=None, help='run demo on a single image')
    parser.add_argument('-f', '--folder', type=str, default=None, help='run demo on a specific folder of images')
    args = parser.parse_args()


    # general settings
    root = os.getcwd()
    model_path =  os.path.join(root,  args.model)

    # network loading
    net = ShapeRecognizer()
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
    net.eval()

    # two options
    if args.folder != None:
        # directory to write results in case
        out_path = os.path.join(root, 'data', 'demo_output')
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        # directory where to read images
        path_imgs = os.path.join(root, args.folder)
        l = gb.glob(os.path.join(path_imgs, '*.jpg'))

        run_folder(net, l, out_path)
        # end results
        print('\n\n Output results saved to:  {}'.format(out_path))

    elif args.image != None:
        image_path = os.path.join(root, args.image)
        run_image(net, image_path)

    else:
        print('Provide an input image or a folder preceded by -i or -f respectively')



if __name__ == "__main__":
    main()
