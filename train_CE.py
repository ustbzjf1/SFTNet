import os
import random
import logging
import numpy as np
import time
import setproctitle
import math
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from config import opts
import models
from models import criterions
from data.BraTS import BraTS
import matplotlib.pyplot as plt



def main():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', config.experiment+config.data)
    log_file = log_dir + '.txt'
    log_config(log_file)
    logging.info('-------------------------------------------This is all configurations-----------------------------------------')
    logging.info(config)
    logging.info('---------------------------------------------This is a halving line-------------------------------------------')
    logging.info('{}'.format(config.description))


    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    #model = generate_model(config)
    model = getattr(models, config.model_name)()
    #model = getattr(models, config.model_name)(c=4,n=32,channels=128, groups=16,norm='sync_bn', num_classes=4,output_func='softmax')
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    # criterion = getattr(criterions, config.criterion)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', config.experiment+config.data)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    resume = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.resume)
    if os.path.isfile(resume) and config.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        config.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'.format(config.resume, config.start_epoch))

    else:
        logging.info('re-training!!!')

    train_list = os.path.join(config.root, config.train_dir, config.train_file)
    train_root = os.path.join(config.root, config.train_dir)

    train_set = BraTS(train_list, train_root, config.mode)
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_iters = (len(train_set) * config.end_epoch) // config.batch_size
    num_iters -= (len(train_set) * config.start_epoch) // config.batch_size
    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)


    for epoch in range(config.start_epoch, config.end_epoch):
        loss_epoch = []
        area1 = []
        area2 = []
        area4 = []
        

        setproctitle.setproctitle('{}:{} {}/{}'.format(config.user, config.model_name, epoch+1, config.end_epoch))
        start_epoch = time.time()

        for i, data in enumerate(train_loader):
        
            

            adjust_learning_rate(optimizer, epoch, config.end_epoch, config.lr)
            #warm_up_learning_rate_adjust2(config.lr, epoch, config.warm, config.end_epoch, optimizer)
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data
            output = model(x)
            
            target[target==4]=3

            loss = criterion(output, target)
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f} ||'.format(epoch, i, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            




        end_epoch = time.time()

        if (epoch + 1) % int(config.save_freq) == 0 \
                or (epoch + 1) % int(config.end_epoch - 1) == 0 \
                or (epoch + 1) % int(config.end_epoch - 2) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (config.end_epoch-epoch-1)*epoch_time_minute/60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))



    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': config.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('-----------------------------------The training process finished!------------------------------------')

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)
        
def warm_up_learning_rate_adjust1(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(epoch+1)/(warm_epoch+1)
        else:
            param_group['lr'] = init_lr*(math.cos(math.pi*(epoch-warm_epoch)/max_epoch)+1)/2

def warm_up_learning_rate_adjust2(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(1-math.cos(math.pi/2*(epoch+1)/(warm_epoch)))
        else:
            param_group['lr'] = init_lr*(math.cos(math.pi*(epoch-warm_epoch)/max_epoch)+1)/2

def log_config(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # config FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # config StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

if __name__ == '__main__':
    config = opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    cudnn.benchmark = True
    main()
