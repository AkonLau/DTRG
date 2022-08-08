import warnings
import torch
import torch.nn as nn

import time
import networks
import logging
import numpy as np
from trainer.update_dtrg import update_dtrg

from utils import get_config,set_env,set_logger,set_outdir,get_dataloader
from utils import get_train_setting,load_checkpoint,get_proc,save_checkpoint

from coding_functions.dtrg import DTRG, DTRG_AUG
from coding_functions.center_loss import CenterLoss
from coding_functions.label_smooth import LabelSmoothLoss
from coding_functions.online_label_smooth import OnlineLabelSmoothing

def main(conf):
    dimdict = {'resnet50': 2048, 'resnet101': 2048, 'densenet121': 1024, 'densenet161': 2208, 'densenet201': 1920}
    mid_dimdict = {'resnet50': 1024, 'resnet101': 1024, 'densenet121': 1024, 'densenet161': 2112, 'densenet201': 1792}

    if conf.netname in ['resnet18','resnet34']:
        indim = 512
        mid_dim = 256
    elif conf.netname in ['resnet32']:
        mid_dim = 32
        indim = 64
    else:
        indim = dimdict[conf.netname]
        mid_dim = mid_dimdict[conf.netname]
    warnings.filterwarnings("ignore")
    best_score = 0.
    epoch_start = 0

    # dataloader
    train_loader, val_loader, ds_train = get_dataloader(conf)

    # model
    model = networks.get_model(conf)
    model = nn.DataParallel(model).cuda()
    if conf.weightfile is not None:
        wmodel = networks.get_model(conf)
        wmodel = nn.DataParallel(wmodel).cuda()
        checkpoint_dict = load_checkpoint(wmodel, conf.weightfile)

        if 'best_score' in checkpoint_dict:
            print('best score: {}'.format(best_score))
    else:
        wmodel = model

    # training setting
    criterion, optimizer,scheduler = get_train_setting(model,conf)
    criterion_test = criterion

    if conf.ls is True:
        # using label smooth
        criterion = LabelSmoothLoss(0.1)
    if conf.ols is True:
        # using online label smooth
        criterion = OnlineLabelSmoothing(num_classes=conf.num_class, use_gpu=True)

    if conf.cl is True:
        # using center loss
        center_loss = CenterLoss(num_classes=conf.num_class, feat_dim=indim, use_gpu=True)
        optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=conf.lr_cent)
    else:
        center_loss = None
        optimizer_centloss = None

    if 'baseline' not in conf.mixmethod:
        dtrg = DTRG_AUG(conf, feat_dim=indim, use_gpu=True)
        mid_dtrg = DTRG_AUG(conf, feat_dim=mid_dim, use_gpu=True)
    else:
        dtrg = DTRG(conf, feat_dim=indim, use_gpu=True)
        mid_dtrg = DTRG(conf, feat_dim=mid_dim, use_gpu=True)

    # training and evaluate process for each epoch
    train,validate = get_proc(conf)

    if conf.resume:
        checkpoint_dict = load_checkpoint(model, conf.resume)
        epoch_start = checkpoint_dict['epoch']

        if 'best_score' in checkpoint_dict:
            best_score = checkpoint_dict['best_score']
            print('best score: {}'.format(best_score))
        print('Resuming training process from epoch {}...'.format(epoch_start))
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        print('Resuming lr scheduler')
        print(checkpoint_dict['scheduler'])

        if conf.ocl or conf.graph:
            # update class centers for resume
            print('updating center matrix from resume...')
            conf['start_dtrg'] = epoch_start
            with torch.no_grad():
                update_dtrg(train_loader, model, dtrg, mid_dtrg, conf, epoch_start-1)
            dtrg.update()
            if 'midlevel' in conf and conf.midlevel is True:
                mid_dtrg.update()
            print('start_dtrg epoch:', conf.start_dtrg+1)

    if conf.evaluate:
        print(validate(val_loader, model, criterion_test, conf))
        return

    detach_epoch = conf.epochs + 1
    if 'detach_epoch' in conf:
        detach_epoch = conf.detach_epoch

    start_eval = 0
    if 'start_eval' in conf:
        start_eval = conf.start_eval

    ## ------main loop-----
    for epoch in range(epoch_start, conf.epochs):
        start_time = time.time()

        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']

        logging.info("Epoch: [{} | {} LR: {} {}".format(epoch+1,conf.epochs,lr0, lr1))

        if epoch == detach_epoch:
            model.module.set_detach(False)

        tmp_loss = train(train_loader, model, criterion, optimizer, dtrg, mid_dtrg, conf, epoch,
                         center_loss, optimizer_centloss, wmodel)
        infostr = {'Epoch:  {}   train_loss: {}'.format(epoch+1,tmp_loss)}
        logging.info(infostr)
        scheduler.step()

        if conf.ocl or conf.graph:
            dtrg.update()
            if 'midlevel' in conf and conf.midlevel is True:
                mid_dtrg.update()
        if conf.ols is True:
            criterion.update()

        if epoch > start_eval and (epoch+1) % 1 == 0:
            with torch.no_grad():
                val_score, val_score_top5,val_loss,mscore,ascore = validate(val_loader, model,criterion_test, conf)
                comscore = val_score
                if 'midlevel' in conf:
                    if conf.midlevel:
                        comscore = ascore

                is_best = comscore > best_score
                best_score = max(comscore,best_score)
                infostr = {'Epoch:  {:.4f}   loss: {:.4f},gs: {:.4f},gs_acc5: {:.4f},ms:{:.4f},as:{:.4f},bs:{:.4f}'.format(
                    epoch+1,val_loss,val_score,val_score_top5,mscore,ascore,best_score)}
                logging.info(infostr)
                save_checkpoint(
                        {'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'best_score': best_score
                        }, is_best, outdir=conf['outdir'], iteral=conf.iteral)
        end_time = time.time()
        seconds = end_time - start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        infostr = {"Epoch Time %02d:%02d:%02d" % (h, m, s)}
        logging.info(infostr)
    logging.info({'Best val acc: {}'.format(best_score)})
    print('Best val acc: {}'.format(best_score))
    # return 0


if __name__ == '__main__':
    start_time = time.time()
    # get configs and set envs
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

    end_time = time.time()
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("During Time %02d:%02d:%02d" % (h, m, s))