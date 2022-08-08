import torch
import os
import os.path as path
from datetime import datetime
import shutil
from tqdm import tqdm
import math
from urllib.request import urlretrieve

# ---------------load checkpoint--------------------
def load_checkpoint(model, pth_file):
    print('==> Reading from model checkpoint..')

    assert os.path.isfile(pth_file), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(pth_file)

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.module.state_dict()
    model_dict.update(pretrained_dict)
    model.module.load_state_dict(model_dict)

    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(pth_file, checkpoint['epoch']))

    return checkpoint

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

# ---------------save checkpoint--------------------
def save_checkpoint(state, is_best=False, outdir='checkpoint', filename='checkpoint.pth',iteral=50):

    epochnum = state['epoch']
    filepath = os.path.join(outdir, filename)
    epochpath =  str(epochnum)+'_'+filename
    epochpath = os.path.join(outdir, epochpath)
    if epochnum % iteral == 0:
        savepath = epochpath
    else:
        savepath = filepath
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(outdir, 'model_best.pth.tar'))

def set_outdir(conf):

    default_outdir = 'results'
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir,conf.exp_name, \
            conf.net_type+'_'+conf.dataset,timestr)
    else:
        outdir = os.path.join(default_outdir,conf.exp_name, \
            conf.netname+'_'+conf.dataset)

        prefix = 'bs'+str(conf.batch_size)+'_seed'+str(conf.seed)

        if conf.weightfile:
            prefix = 'ft_'+prefix

        if not conf.pretrained:
            prefix = '_scratch_'+prefix

        if 'midlevel' in conf:
            if conf.midlevel:
                prefix += '_mid_'

        if 'mixmethod' in conf:
            if isinstance(conf.mixmethod,list):
                prefix += '_'.join(conf.mixmethod)
            else:
                prefix += '_' + (conf.mixmethod)
        if 'prob' in conf:
            prefix += '_p'+str(conf.prob)
        if 'beta' in conf:
            prefix += '_b'+str(conf.beta)

        if conf.ocl or conf.graph:
            if 'start_dtrg' in conf:
                prefix += '_dtrg_s%d' % (conf.start_dtrg)

            if conf.ocl is True:
                prefix += '_ocl'
                if 'weight_cent' in conf:
                    prefix += '_w%.e' % (conf.weight_cent)
            if conf.graph is True:
                prefix += '_graph'
                if 'distmethod' in conf:
                    prefix += '_'+str(conf.distmethod)
                    if 'tau' in conf:
                        prefix += '_t' + str(conf.tau)
                    if 'eta' in conf:
                        prefix += '_e' + str(conf.eta)

        if conf.ls is True:
            prefix += '_label_smooth'
        if conf.ols is True:
            prefix += '_online_label_smooth'
        if conf.cl is True:
            prefix += '_center_loss'

        if conf.dataset =='dtd':
            prefix += '_partition_' + str(conf.partition)

        prefix += '_epochs' + str(conf.epochs)
        conf['prefix'] = prefix
        outdir = os.path.join(outdir,prefix)

    ensure_dir(outdir)
    conf['outdir'] = outdir

    return conf


# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))


def ensure_file(file_path):

    newpath = file_path
    if os.path.exists(file_path):
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p_')
        newpath = path.join(path.dirname(file_path),timestr + path.basename(file_path))
    return newpath


