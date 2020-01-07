import time
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class opts():
    def __init__(self):
        self.user = 'zjf'
        self.experiment = 'Unet_sft(34)_'
        self.data = local_time.split(' ')[0]
        self.warm = 30

        self.root = '/data1/zjf'
        self.train_dir = 'MICCAI_BraTS_2019_Data_Training'
        self.valid_dir = 'MICCAI_BraTS_2019_Data_Validation'
        self.mode = 'train_all'  #choice['train_all', 'train_fold', 'test'] set to train_all while train and change to train_fold while valid
        #self.train_file = 'train_small.txt'
        self.train_file = 'all.txt'
        self.valid_file = 'valid.txt'
        self.dataset = 'brats'

        self.model = 'resnet'
        self.model_name = 'Unet_3D_SFT'
        self.model_depth = 34

        self.input_C = 4
        self.input_H = 240
        self.input_W = 240
        self.input_D = 160
        self.output_D = 155
        self.resnet_shortcut = 'A'  #['10':B , '18':A, '34':A, '50':B, '101':B, '152':B, '200':B]
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.amsgrad = True
        self.criterion = 'Generalized_dice'  #choice=['Generalized_dice','softmax_dice','Dual_focal_loss']
        self.num_class = 4

        self.description = 'DMFNet16x training on the whole dataset with 128!'
        self.seed = 1
        self.no_cuda = False
        self.resume = 'checkpoint/DMFNet_T128_xavier_2019-11-19/model_epoch_139.pth'
        self.load = False
        self.valid_transform = 'Compose([Pad((0, 0, 0, 5, 0)),NumpyType((np.float32, np.int64))])'
        self.GPU = '4, 5, 6, 7'
        self.num_workers = 16
        self.batch_size = 4
        self.start_epoch = 0
        self.end_epoch = 500
        self.save_freq = 50

        #params for test
        self.output_dir = 'output'
        self.submission = 'submission'
        self.visual = 'visualization'
        self.test_date = '2020-01-02'
        self.test_file = 'model_epoch_last.pth'
        self.snapshot = True
        self.use_TTA = False
        self.post_process = True
        self.save_format = 'nii'  #choice=['npy', 'nii']

        self.crop_H = 128
        self.crop_W = 128
        self.crop_D = 128

    def gatherAttrs(self):
        return ",".join("\n{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

    def __str__(self):
        return "{}:{}".format(self.__class__.__name__, self.gatherAttrs())



