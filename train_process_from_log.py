import numpy as np
import matplotlib.pyplot as plt

x_axis = range(0, 500)
loss_all = []
area1 = []
area2 = []
area4 = []

loss = []
loss1 = []
loss2 = []
loss3 = []

experiment = None
data = None
with open('./log/DMFNet_T128_fold_2019-11-15.txt') as f:
    i = 0
    for line in f:

        if 'Epoch' in line:
            all = line.split(':')
            if (i+1) % 34 == 0:
                loss_all.append(np.mean(loss))
                area1.append(np.mean(loss1))
                area2.append(np.mean(loss2))
                area4.append(np.mean(loss3))
                loss = []
                loss1 = []
                loss2 = []
                loss3 = []
            else:
                loss.append(float(all[5][:7]))
                loss1.append(float(all[6][:6]))
                loss2.append(float(all[7][:6]))
                loss3.append(float(all[8][:6]))

            i = i+1
        if 'experiment' in line:
            experiment = line.split('=')[-1].split(',')[0]
        if 'data=' in line:
            data = line.split('=')[-1].split(',')[0]


plt.plot(x_axis, loss_all, color='red', linewidth=1, label='train_loss')
plt.plot(x_axis, area1, color='blue', linewidth=1, label="accuracy_1")
plt.plot(x_axis, area2, color='purple', linewidth=1, label="accuracy_2")
plt.plot(x_axis, area4, color='green', linewidth=1, label="accuracy_4")
plt.title('Training process')
plt.ylabel('loss/accuracy')
plt.legend()
save_path = './train_process/' + experiment + data + '.jpg'
plt.savefig(save_path)