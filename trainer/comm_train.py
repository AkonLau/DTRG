import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time
import logging
from utils import *
from colorama import Fore

def train(train_loader, model, criterion, optimizer, dtrg, mid_dtrg, conf, epoch,
          center_loss=None, optimizer_centloss=None, wmodel=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dtrg_losses = AverageMeter('Loss', ':.4e')

    # scores = AverageMeter('Acc@1', ':6.2f')
    end = time.time()
    model.train()

    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader),
                ascii=True, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
    mixmethod = None

    if 'mixmethod' in conf:
        if 'baseline' not in conf.mixmethod:
            mixmethod = conf.mixmethod
            if wmodel is None:
                wmodel = model

    for idx, (input, target) in enumerate(pbar):

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        if 'baseline' not in conf.mixmethod:
            input,target_a,target_b,lam_a,lam_b = eval(mixmethod)(input,target,conf,wmodel)

            output,_,moutput,[xf, pool4_1] = model(input)

            loss_a = criterion(output, target_a)
            loss_b = criterion(output, target_b)
            loss = torch.mean(loss_a* lam_a + loss_b* lam_b)

            if conf.ocl or conf.graph:
                # compute online center loss with weight_cent = 1e-3
                dtrg_loss = dtrg(xf, target_a, target_b, lam_a, lam_b, epoch)
                dtrg_losses.update(dtrg_loss.item(), input.size(0))
                loss += dtrg_loss

            if 'midlevel' in conf:
                if conf.midlevel:
                    loss_ma = criterion(moutput, target_a)
                    loss_mb = criterion(moutput, target_b)
                    loss += torch.mean(loss_ma* lam_a + loss_mb* lam_b)

                    if conf.ocl or conf.graph:
                        # compute online center loss with weight_cent = 1e-3
                        mid_dtrg_loss = mid_dtrg(pool4_1, target_a, target_b, lam_a, lam_b, epoch)
                        loss += mid_dtrg_loss

        else:
            output,_,moutput,[xf, pool4_1] = model(input)
            loss = torch.mean(criterion(output, target))

            if center_loss is not None:
                # ablation study with center loss
                loss += center_loss(xf, target) * conf.weight_cent
                optimizer_centloss.zero_grad()

            if conf.ocl or conf.graph:
                # compute online center loss with weight_cent = 1e-3
                dtrg_loss = dtrg(xf, target, epoch)
                dtrg_losses.update(dtrg_loss.item(), input.size(0))
                loss += dtrg_loss

            if 'midlevel' in conf and conf.midlevel is True:
                loss += torch.mean(criterion(moutput,target))

                if conf.ocl or conf.graph:
                    # compute online center loss with weight_cent = 1e-3
                    mid_dtrg_loss = mid_dtrg(pool4_1, target, epoch)
                    loss += mid_dtrg_loss

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update center loss
        if center_loss is not None:
            for param in center_loss.parameters():
                param.grad.data *= ( 1.0 / conf.weight_cent)
            optimizer_centloss.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(batch_time=batch_time.avg, data_time=data_time.avg, loss=losses.avg, dtrg_loss=dtrg_losses.avg)

    return losses.avg