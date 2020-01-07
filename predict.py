import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from medpy import metric
import nibabel as nib
import scipy.misc
from config import opts

config = opts()
cudnn.benchmark = True


def tailor_and_concat(x, model):
    temp = []
    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    for i in range(len(temp)):
        temp[i] = model(temp[i])
    x[..., :128, :128, :128] = temp[0]
    x[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    x[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    x[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    x[..., :128, :128, 128:155] = temp[4][..., 96:123]
    x[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    x[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    x[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return x[..., :155]


def dice_score(o, t,eps = 1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    # print('All_voxels:240*240*155 | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}'.format(int(num),int(den),o.sum(),int(t.sum())))
    return num/den


def mIOU(o,t,eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

    return ret


keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        cfg='',
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        scoring=True,  # If true, print the dice score.
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        cpu_only=False,
        valid_in_train=False,  # if you are valid when train
        ):

    H, W, T = config.input_H, config.input_W, config.output_D
    model.eval()



    runtimes = []
    # vals = AverageMeter()
    # mIOUs = AverageMeter()
    for i, data in enumerate(valid_loader):
        if valid_in_train:
            target_cpu = data[1][0, :H, :W, :T].numpy()
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
        else:
            x = data
            x.cuda()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit = tailor_and_concat(x, model)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            runtimes.append(elapsed_time)

            output = F.softmax(logit, dim=1)
        else:
            logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  #flip H
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  #flip W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  #flip D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  #flip H, W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  #flip H, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  #flip W, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  #flip H, W, D
            output = logit / 8.0  # mean

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)

        if postprocess == True:
            ET_voxels = (output == 3).sum()
            if ET_voxels < 500:
                output[np.where(output == 3)] = 1

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name + '.nii.gz')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

                seg_img[np.where(output == 1)] = 1
                seg_img[np.where(output == 2)] = 2
                seg_img[np.where(output == 3)] = 4
                if verbose:
                    print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
                    print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
                          np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
                nib.save(nib.Nifti1Image(seg_img, None), oname)
                print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])

#----------------------------------------Don't need now--------------------------------------------------------------
#         if scoring:
#             scores = softmax_output_dice(output, target_cpu)
#             # mIOU_score = softmax_mIOU_score(output, target_cpu)
#             vals.update(np.array(scores))
#             # mIOUs.update(np.array(mIOU_score))
#             msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])
#             # logging.info('[mIOU] label_1:{} | label_2:{} | label_4:{}'.format(mIOU_score[0],mIOU_score[1],mIOU_score[2]))
#
#             if snapshot:
#                 # red: (255,0,0) green:(0,255,0) blue:(0,0,255) 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.
#                 gap_width = 2  # boundary width = 2
#                 Snapshot_img = np.zeros(shape=(H, W*2+gap_width, 3, T), dtype=np.uint8)
#                 Snapshot_img[:, W:W+gap_width, :] = 255  # white boundary
#
#                 empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 empty_fig[np.where(output == 1)] = 255
#                 Snapshot_img[:, :W, 0, :] = empty_fig
#                 empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 empty_fig[np.where(target_cpu == 1)] = 255
#                 Snapshot_img[:, W+gap_width:, 0, :] = empty_fig
#
#                 empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 empty_fig[np.where(output == 2)] = 255
#                 Snapshot_img[:, :W, 1, :] = empty_fig
#                 empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 empty_fig[np.where(target_cpu == 2)] = 255
#                 Snapshot_img[:, W+gap_width:, 1, :] = empty_fig
#
#                 empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 empty_fig[np.where(output == 3)] = 255
#                 Snapshot_img[:, :W, 2, :] = empty_fig
#                 empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 empty_fig[np.where(target_cpu == 4)] = 255
#                 Snapshot_img[:, W+gap_width:, 2, :] = empty_fig
#
#                 for frame in range(T):
#                     os.makedirs(os.path.join('snapshot',cfg, name), exist_ok=True)
#                     scipy.misc.imsave(os.path.join('snapshot',cfg, name, str(frame) + '.png'), Snapshot_img[:,:,:,frame])
#
#         logging.info(msg)
#
#     if scoring:
#         msg = 'Average scores:'
#         msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
#         logging.info(msg)
#         # logging.info('Average mIOU:', mIOUs.avg)
#
#     print(runtimes)
#     computational_runtime(runtimes)
#
#     model.train()
#     return vals.avg
#
# def computational_runtime(runtimes):
#     # remove the maximal value and minimal value
#     runtimes = np.array(runtimes)
#     maxvalue = np.max(runtimes)
#     minvalue = np.min(runtimes)
#     nums = runtimes.shape[0] - 2
#     meanTime = (np.sum(runtimes) - maxvalue - minvalue ) / nums
#     fps = 1 / meanTime
#     print('mean runtime:', meanTime, 'fps:', fps)
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count