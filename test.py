import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
cudnn.benchmark = True

from config import opts
from data.BraTS import BraTS
from predict import validate_softmax
import models



def main():

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    #model = generate_model(config)
    model = getattr(models, config.model_name)()
    # model = getattr(models, config.model_name)(c=4,n=32,channels=128, groups=16,norm='sync_bn', num_classes=4,output_func='softmax')
    model = torch.nn.DataParallel(model).cuda()

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'checkpoint', config.experiment+config.test_date, config.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        config.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(config.experiment+config.test_date, config.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(config.root, config.valid_dir, config.valid_file)
    valid_root = os.path.join(config.root, config.valid_dir)
    valid_set = BraTS(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.output_dir,
                              config.submission, config.experiment+config.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.output_dir,
                          config.visual, config.experiment+config.test_date)
    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         savepath=submission,
                         visual=visual,
                         names=valid_set.names,
                         scoring=False,
                         use_TTA=config.use_TTA,
                         save_format=config.save_format,
                         snapshot=True,
                         postprocess=True
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(config.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


