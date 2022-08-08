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

def validate(train_loader, model, criterion, conf):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    scores_top1 = AverageMeter('Acc@1', ':6.2f')
    scores_top5 = AverageMeter('Acc@5', ':6.2f')

    mscores = AverageMeter('Acc@1', ':6.2f')
    ascores = AverageMeter('Acc@1', ':6.2f')
    end = time.time()
    model.eval()

    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader), ascii=True)

    for idx, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        output,_,moutput,_ = model(input)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        scores_top1.update(acc1[0], input.size(0))
        scores_top5.update(acc5[0], input.size(0))

        if 'midlevel' in conf:
            if conf.midlevel:
                mscores_acc1 = accuracy(moutput, target, topk=(1,))
                mscores.update(mscores_acc1[0][0], input.size(0))
                ascores_acc1 = accuracy(output+moutput, target, topk=(1,))
                ascores.update(ascores_acc1[0][0], input.size(0))

        loss = torch.mean(criterion(output, target))
        losses.update(loss.item(), input.size(0))
        del loss,output
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        pbar.set_postfix(batch_time=batch_time.avg, data_time=data_time.avg, loss=losses.avg)

    return scores_top1.avg, scores_top5.avg, losses.avg, mscores.avg, ascores.avg
