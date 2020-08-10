# ---------------------贴图-------------------------------------------

from PIL import Image
import os
import random
read_path = r'./贴图/'
save_path = './save/'
# 背景图片
ground = '0.jpg'

def resize_img(img,add_img,file):
    add_img_w = add_img.size[0]
    add_img_h = add_img.size[1]

    img_w = img.size[0]
    img_h = img.size[1]
    while True:
        if add_img_w<img_w and add_img_h<img_h:
            h = int((img_h-add_img_h)/2)
            w = int((img_w-add_img_w)/2)
            img.paste(add_img,(w,h))
            img.save('{}/{}.jpg'.format(save_path,random.random()))
            break
        else:
            add_img = add_img.resize((int(add_img_w/2),int(add_img_h/2)))
            add_img_w = add_img.size[0]
            add_img_h = add_img.size[1]
            continue


dire = os.listdir(read_path)
for file in dire:
    path = ground
    img = Image.open(path)
    markImg = Image.open(read_path+file)
    markImg_w = markImg.size[0]
    markImg_h = markImg.size[1]
    # 随机对图片进行缩放
    count = random.uniform(2,3)
    w = round(markImg_w*count,1)
    h = round(markImg_h*count,1)

    # 原图
    resize_img(img,markImg,file)
    # 将图像旋转指定尺寸
    markImg = markImg.resize((int(w),int(h)))
    add_img = markImg.rotate(45)
    resize_img(img,add_img,file)
    # 上下镜像
    markImg = markImg.resize((int(w),int(h)))
    add_img = markImg.transpose(Image.FLIP_TOP_BOTTOM)
    resize_img(img,add_img,file)
    # 左右镜像
    markImg = markImg.resize((int(w),int(h)))
    add_img = markImg.transpose(Image.FLIP_LEFT_RIGHT)
    resize_img(img,add_img,file)
    # 90度镜像
    path = ground
    img = Image.open(path)
    markImg = markImg.resize((int(w),int(h)))
    add_img = markImg.transpose(Image.ROTATE_90)
    resize_img(img,add_img,file)
    # 180度镜像
    path = ground
    img = Image.open(path)
    markImg = markImg.resize((int(w),int(h)))
    add_img = markImg.transpose(Image.ROTATE_180)
    resize_img(img,add_img,file)
    # 颠倒
    path = ground
    img = Image.open(path)
    markImg = markImg.resize((int(w),int(h)))
    add_img = markImg.transpose(Image.TRANSPOSE)
    resize_img(img,add_img,file)
    



 
# 读入原图片
# import os
# import tqdm
# import cv2
# path = r'./normal/'
# dir = os.listdir(path)

# for file in tqdm.tqdm(dir):
#     # print(img)
#     img = cv2.imread(path+file)
#     #print(img)
#     try:
#         img_test1 = cv2.resize(img, (224,224))
#     except:
#         cv2.imwrite('normal_resize/{}'.format(file),img_test1)
#     cv2.imwrite('normal_resize/{}'.format(file),img_test1)

