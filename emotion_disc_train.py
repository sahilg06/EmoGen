import argparse
import json
import os
from tqdm import tqdm
import random as rn
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from models import emo_disc
from datagen_aug import Dataset

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-path", type=str, help="Input folder containing train data", default=None, required=True)
    # parser.add_argument("-v", "--val-path", type=str, help="Input folder containing validation data", default=None, required=True)
    parser.add_argument("-o", "--out-path", type=str, help="output folder", default='../models/def', required=True)

    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument('--lr_emo', type=float, default=1e-06)

    parser.add_argument("--gpu-no", type=str, help="select gpu", default='1')
    parser.add_argument('--seed', type=int, default=9)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    args.batch_size = args.batch_size * max(int(torch.cuda.device_count()), 1)
    args.steplr = 200

    args.filters = [64, 128, 256, 512, 512]
    #-----------------------------------------#
    #           Reproducible results          #
    #-----------------------------------------#
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)
    torch.manual_seed(args.seed)
    #-----------------------------------------#
   
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    else:
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)

    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.cuda = torch.cuda.is_available() 
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu") 
    args.kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    return args

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def enableGrad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
  

def train():
    args = initParams()
    
    trainDset = Dataset(args)

    train_loader = torch.utils.data.DataLoader(trainDset,
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)
    
    device_ids = list(range(torch.cuda.device_count()))
    
    disc_emo = emo_disc.DISCEMO().to(args.device)
    disc_emo.apply(init_weights)
    #disc_emo = nn.DataParallel(disc_emo, device_ids)

    emo_loss_disc = nn.CrossEntropyLoss()

    num_batches = len(train_loader)
    print(args.batch_size, num_batches)

    global_step = 0
    
    for epoch in range(args.num_epochs):
        print('Epoch: {}'.format(epoch))
        prog_bar = tqdm(enumerate(train_loader))
        running_loss = 0.
        for step, (x, y) in prog_bar:
            video, emotion = x.to(args.device), y.to(args.device)

            disc_emo.train()

            disc_emo.opt.zero_grad() # .module is because of nn.DataParallel 

            class_real = disc_emo(video)

            loss = emo_loss_disc(class_real, torch.argmax(emotion, dim=1))

            running_loss += loss.item()

            loss.backward()
            disc_emo.opt.step() # .module is because of nn.DataParallel 

            if global_step % 1000 == 0:
                print('Saving the network')
                torch.save(disc_emo.state_dict(), os.path.join(args.out_path, f'disc_emo_{global_step}.pth'))
                print('Network has been saved')
            
            prog_bar.set_description('classification Loss: {}'.format(running_loss / (step + 1)))

            global_step += 1

        writer.add_scalar("classification Loss", running_loss/num_batches, epoch)
        
        disc_emo.scheduler.step() # .module is because of nn.DataParallel 

if __name__ == "__main__":

    writer = SummaryWriter('runs/emo_disc_exp4')
    train()