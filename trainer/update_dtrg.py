import time
from utils import *
from colorama import Fore

def update_dtrg(train_loader, model, dtrg, mid_dtrg, conf, epoch):
    '''For updating center feature when resuming from middle epoch'''
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    end = time.time()
    model.train()

    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader),
                ascii=True, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))

    for idx, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        output,_,moutput,[xf, pool4_1] = model(input)

        if conf.ocl or conf.graph:
            _ = dtrg(output, xf, target, conf, epoch)

        if 'midlevel' in conf and conf.midlevel is True:

            if conf.ocl or conf.graph:
                _ = mid_dtrg(moutput, pool4_1, target, conf, epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(batch_time=batch_time.avg, data_time=data_time.avg)
