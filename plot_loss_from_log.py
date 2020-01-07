import numpy as np
import matplotlib.pyplot as plt
import math

def adjust_learning_rate(epoch, max_epoch, init_lr, power=0.9):
        return round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

def warm_up_learning_rate_adjust1(init_lr, epoch, warm_epoch, max_epoch):
    if epoch < warm_epoch:
        return init_lr*(epoch+1)/(warm_epoch+1)
    else:
        return init_lr*(math.cos(math.pi*(epoch-warm_epoch)/(max_epoch-warm_epoch))+1)/2

def warm_up_learning_rate_adjust2(init_lr, epoch, warm_epoch, max_epoch):
    if epoch < warm_epoch:
        return init_lr*(1-math.cos(math.pi/2*(epoch+1)/(warm_epoch)))
    else:
        return init_lr*(math.cos(math.pi*(epoch-warm_epoch)/(max_epoch-warm_epoch))+1)/2

# 设置x,y轴的数值（y=sinx）
x = range(0, 500)

y1 = []
y2 = []
y3 = []
for i in x:
    y1.append(warm_up_learning_rate_adjust1(0.001, i, 50, 500))
    y2.append(warm_up_learning_rate_adjust2(0.001, i, 50, 500))
    y3.append(adjust_learning_rate(i, 500, 0.001))

# 创建绘图对象，figsize参数可以指定绘图对象的宽度和高度，单位为英寸，一英寸=80px
# plt.figure(figsize=(8, 4))
# plt.subplot(3, 1, 1)
plt.plot(x, y1, color='red', linewidth=1, label='warm_up1')
plt.title('warm up 1')
plt.ylabel('learning rate')

# plt.subplot(3, 1, 2)
plt.plot(x, y2, color='blue', linewidth=1, label='warm_up2')
plt.title('warm up 2')
plt.ylabel('learning rate')

# plt.subplot(3, 1, 3)
plt.plot(x, y3, color='yellow', linewidth=1, label='linear_adjust')
plt.title('adjust')
plt.ylabel('learning rate')

# # 在当前绘图对象中画图（x轴,y轴,给所绘制的曲线的名字，画线颜色，画线宽度）
# plt.plot(x, y1, label="$sin(x)$", color="red", linewidth=2)
#
# # X轴的文字
# plt.xlabel("epoch")
#
# # Y轴的文字
# plt.ylabel("loss or accuracy")
#
# # 图表的标题
# plt.title("Training process")

# Y轴的范围
# plt.ylim(-0.2, 1.2)
# plt.ylim(-0.2, 1.2)
# 显示图示
plt.legend()

# 保存图
plt.savefig("warm_up.jpg")

# 显示图
plt.show()

