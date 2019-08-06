import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from model import ShapeRecognizer
from data_utils import custom_dset, collate_fn, denorm
from torch.autograd import Variable
import time
import cv2
import random


def check_val(epoch, train_it, val_loader, crit, net, out_path):

    # true positives
    tp_s, tp_c, tp_t = 0,0,0

    # validation stage
    for i, (img, squares, circles, triangles) in enumerate(val_loader):

        img = img.cuda()
        squares, circles, triangles = squares.cuda(), circles.cuda(), triangles.cuda()

        # run input through the network
        pred_squar, pred_circ, pred_trgle, sa, ca, ta = net(img)

        # CrossEntropyLoss
        ls = crit(pred_squar, squares)
        lc = crit(pred_circ, circles)
        lt = crit(pred_trgle, triangles)

        # A regularization that the visualization of the attention masks
        reg = (sa.sum() + ca.sum() + ta.sum()) * 1e-05

        val_loss = ls + lc + lt + reg


        #  evalutate over a metrics
        _, ps = torch.max(pred_squar, dim=-1)
        _, pc = torch.max(pred_circ, dim=-1)
        _, pt = torch.max(pred_trgle, dim=-1)


        tp_s += torch.sum(squares == ps).item()
        tp_c += torch.sum(circles == pc).item()
        tp_t += torch.sum(triangles == pt).item()

    # this is the only metric adopted
    tp_s = tp_s/val_len
    tp_c = tp_c/val_len
    tp_t = tp_t / val_len
    avg_acc = (tp_s + tp_c + tp_t)/3
    print('\nAccuracy on squares {}  circles {}  triangles {} and Average {}'.format( tp_s, tp_c, tp_t, round(avg_acc, 2) ))
    print('Batch of GT on Validation: {}'.format(list(squares.data.cpu().numpy())))
    print('Batch of predictions on Validation: {}\n'.format(list(ps.data.cpu().numpy())))

    # save on disk, input image along with the attention masks
    img = img[0].data.cpu().numpy()
    img = denorm(img.transpose(1,2,0))
    img = cv2.resize(img, (256, 256))

    #
    con_masks = torch.cat(( sa[0], ca[0], ta[0]), 2).data.cpu().numpy().transpose(1,2,0)*255
    con_masks = cv2.resize(con_masks, (256*3, 256)).astype(np.uint8)
    con_masks = cv2.applyColorMap(con_masks, 11)
    concat = np.concatenate((img, con_masks), 1)

    # name comprehensive of the current epoch and followed by the output predictions
    name = str(epoch) + '__' +  str(train_it) + '___' + str(ps[0].item()) + str(pc[0].item()) + str(pt[0].item()) + '.jpg'
    cv2.imwrite(os.path.join(out_path, name), concat)

    return val_loss, reg


def train(epochs, net, train_loader, val_loader, optimizer,
          save_step, out_path):

    crit = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        print('*'* 100)
        print('Epoch {} / {}'.format(e + 1, epochs))
        net.train()

        # training stage
        for it, (img, squares, circles, triangles) in enumerate(train_loader):
            optimizer.zero_grad()

            img = Variable(img.cuda())
            squares = Variable(squares.cuda())
            circles = Variable(circles.cuda())
            triangles = Variable(triangles.cuda())

            # run input through the network
            pred_squar, pred_circ, pred_trgle, sa, ca, ta = net(img)

            # CrossEntropyLoss
            ls = crit(pred_squar, squares)
            lc = crit(pred_circ, circles)
            lt = crit(pred_trgle, triangles)

            # add to the loss a regularization factor. Mostly it helps for the visualization of the attention masks
            reg = (sa.sum() + ca.sum() + ta.sum())*1e-05

            train_loss = ls  + lc + lt + reg

            train_loss.backward()
            optimizer.step()

            if (it + 1) % 20 == 0:
                net.eval()
                val_loss, val_reg = check_val(e, it, val_loader, crit, net, out_path)
                print('Training loss: {} Train reg: {} Validation loss: {} Val reg: {} '.format(round(train_loss.item(), 3),
                 round(reg.item(), 3),round(val_loss.item(), 3), round(val_reg.item(), 3)))
                net.train()

        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(net.state_dict(), './checkpoints/net_{}.pth'.format(e + 1))

def main():


    # seed for random generator libraries
    global seed
    #seed = np.random.randint(0, 10000)
    seed = 9345
    print(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # deterministic cudnn
    print('Additional cudnn determinism')
    torch.backends.cudnn.deterministic = True

    # get the working directory
    root = os.getcwd()


    out_path = os.path.join(root, 'masks')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # Load dataset
    trainset = custom_dset(root, 'train')
    valset = custom_dset(root, 'validation')
    train_loader = DataLoader(
                    trainset, batch_size=28, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # for some technical issues I had to specify shuffle=True and  use a low batch size for validation.
    val_loader = DataLoader(
                    valset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # global variables
    global val_len
    global train_len
    val_len = len(valset)

    # ShapeRecognizer model is able to count three type shapes
    net = ShapeRecognizer()
    net = net.cuda()
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


    train(epochs=11, net=net, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
               save_step=2, out_path=out_path)

if __name__ == "__main__":
    main()
