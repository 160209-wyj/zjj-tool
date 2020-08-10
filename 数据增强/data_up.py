# '''
# 数据增强代码
# '''

from PIL import Image 
import random
import numpy as np 
import matplotlib.pyplot as plt
import tqdm
import os
import cv2
import tensorflow as tf

        


# 随机裁剪
'''
random_crop
对图片随机进行crop
'''

def random_crop(img,w,h,c):
    img = cv2.imread(img)
    # print img
    cv2.imshow("inputaa", img)
    sess = tf.InteractiveSession()	
    # height, width = img.shape[:2]
    for x in range(4):
        cropped_image = tf.random_crop(img, (w, h,c))
        cv2.imshow("output" + str(x), cropped_image.eval())
        cv2.imwrite('./img/img/random{}.jpg'.format(x),cropped_image.eval())
    cv2.waitKey(0) 
    sess.close()

random_crop("./img/img/11.jpg",400,400,3)



# 翻转
# mode: 1=水平翻转 / 0=垂直 / -1=水平垂直
def flip(img, mode=1):
    
    assert mode in (0, 1, -1), "mode is not right"
    return cv2.flip(img, flipCode=mode)


# 随机锐化增强
# USM锐化增强算法可以去除一些细小的干扰细节和图像噪声，比一般直接使用卷积锐化算子得到的图像更可靠。
# output = 原图像−w∗高斯滤波(原图像)/(1−w)
# 其中w为上面所述的系数，取值范围为0.1~0.9，一般取0.6。
def random_USM(img, gamma=0.):
 
    blur = cv2.GaussianBlur(img, (0, 0), 25)
    img_sharp = cv2.addWeighted(img, 1.5, blur, -0.3, gamma)
    return img_sharp

# 2.1 随机扰动
#     噪声(高斯、自定义)  noise
def random_noise(img, rand_range=(3, 20)):
    
    img = np.asarray(img, np.float)
    sigma = random.randint(*rand_range)
    nosie = np.random.normal(0, sigma, size=img.shape)
    img += nosie
    img = np.uint8(np.clip(img, 0, 255))
    return img


# 高斯模糊
# 各种滤波原理介绍：https://blog.csdn.net/hellocsz/article/details/80727972
# ks:  卷积核      stdev: 标准差
def gaussianBlue(img, ks=(7, 7), stdev=1.5):

    return cv2.GaussianBlur(img, (7, 7), 1.5)


# 随机滤波
def ranndom_blur(img, ksize=(3, 3)):
   
    blur_types = ['gaussian', 'median', 'bilateral', 'mean', 'box']
    assert len(blur_types) > 0
    blur_func = None
    blur_index = random.choice(blur_types)
    if blur_index == 0:  # 高斯模糊, 比均值滤波更平滑，边界保留更加好
        blur_func = cv2.GaussianBlur
    elif blur_index == 1:  # 中值滤波, 在边界保存方面好于均值滤波，但在模板变大的时候会存在一些边界的模糊。对于椒盐噪声有效
        blur_func = cv2.medianBlur
    elif blur_index == 2:  # 双边滤波, 非线性滤波，保留较多的高频信息，不能干净的过滤高频噪声，对于低频滤波较好，不能去除脉冲噪声
        blur_func = cv2.bilateralFilter
    elif blur_index == 3:  # 均值滤波, 在去噪的同时去除了很多细节部分，将图像变得模糊
        blur_func = cv2.blur
    elif blur_index == 4:  # 盒滤波器
        blur_func = cv2.boxFilter
    img_blur = blur_func(src=img, ksize=ksize)
    return img_blur


# 直方图均衡化
def equalize_hist(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.equalizeHist(gray)
    rgb = cv2.cvtColor(hist, cv2.COLOR_GRAY2RGB)
    return rgb


#随机旋转
# angle_range:  旋转角度范围 (min,max)   >0 表示逆时针
def random_rotate(img, angle_range=(-10, 10)):
    
    height, width = img.shape[:2]  # 获取图像的高和宽
    center = (width / 2, height / 2)  # 取图像的中点
    angle = random.randrange(*angle_range, 1)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 获得图像绕着某一点的旋转矩阵
    # cv2.warpAffine()的第二个参数是变换矩阵,第三个参数是输出图像的大小
    rotated = cv2.warpAffine(img, M, (height, width))
    return rotated

# 偏移  shift
# 偏移，向右 向下
#  x_offset:  >0表示向右偏移px, <0表示向左
#  y_offset:  >0表示向下偏移px, <0表示向上
def shift(img, x_offset, y_offset):
    h, w, _ = img.shape
    M = np.array([[1, 0, x_offset], [0, 1, y_offset]], dtype=np.float)
    return cv2.warpAffine(img, M, (w, h))

# 缩放  scale
def resize_img(img, resize_w, resize_h):
    height, width = img.shape[:2]  # 获取图片的高和宽
    return cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)


def rgb2hsv_cv(img):
    # from https://blog.csdn.net/qq_38332453/article/details/89258058
    h = img.shape[0]
    w = img.shape[1]
    H = np.zeros((h,w),np.float32)
    S = np.zeros((h, w), np.float32)
    V = np.zeros((h, w), np.float32)
    r,g,b = cv2.split(img)
    r, g, b = r/255.0, g/255.0, b/255.0
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            V[i, j] = mx
            if V[i, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = (V[i, j] - mn) / V[i, j]
            if mx == mn:
                H[i, j] = 0
            elif V[i, j] == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))+360
            elif V[i, j] == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (V[i, j] - mn) + 120
            elif V[i, j] == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (V[i, j] - mn) + 240
            H[i,j] = H[i,j] / 2
    return H, S, V


# 颜色抖动(亮度\色度\饱和度\对比度)  color jitter
def adjust_contrast_bright(img, contrast=1.2, brightness=100):
    """
    调整亮度与对比度
    dst = img * contrast + brightness
    :param img:
    :param contrast: 对比度   越大越亮
    :param brightness: 亮度  0~100
    :return:
    """
    # 像素值会超过0-255， 因此需要截断
    return np.uint8(np.clip((contrast * img + brightness), 0, 255))

def pytorch_color_jitter(img):
    return torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)








