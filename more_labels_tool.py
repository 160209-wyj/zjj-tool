
#------------------------读取txt文件行数-------------------------------
# filename = r"csv_多标签.txt"
# myfile = open(filename) 
# lines = len(myfile.readlines()) 
# print(lines)




#-----------------打乱txt每一行--------------

import os
import random
out = open("train_shuffle.txt",'w')
lines=[]
with open("train.txt", 'r',encoding='utf-8') as infile:
	for line in infile:
		lines.append(line)
random.shuffle(lines)
for line in lines:
    out.write(line)




#------------------------------随机从txt传到另一个txt--------------------------
# read_filename = r"normal.txt"
# write_filename = r"train.txt"
# with open(read_filename,'r',encoding='utf-8')  as read:
#     with open(write_filename,'a')  as writer:
#         data = read.readlines()
#         for line_data in data:
#             writer.write(line_data)


#------------------------------------在图片之上添加文字------------------------

# import cv2

# #加载背景图片
# bk_img = cv2.imread("1.jpg")

# w = bk_img.shape[0]
# h = bk_img.shape[1]
# y0, dy = int(w*0.1), int(h*0.1)
# #在图片上添加文字信息
# text = "[('red', 0.665), ('file', 0.843)]"
# for i, txt in enumerate(text.split(')')):
#     y = y0+i*dy
#     cv2.putText(bk_img,txt, (int(w*0.1),y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)
# #显示图片
# cv2.imshow("add_text",bk_img)
# cv2.waitKey()
# #保存图片
# cv2.imwrite("add_text.jpg",bk_img)


#----------------------pands读取csv  并写入另一个csv----------------------
# import pandas as pd
# import csv
# with open('test.txt','a') as txt:
#     with open('labels.csv', 'r', encoding="utf-8") as f:
#         # data = pd.read_csv("labels.csv")
#         reader = csv.reader(f)
#         # print(type(reader))
    
#         for row in list(reader)[1::10]:
#             row1 = ','.join(row)
#             # print(row)
#             txt.write(row1)
#             txt.write('\n')
#             # data.drop(row)


# import cv2
# import numpy
# from PIL import Image, ImageDraw, ImageFont

 
# def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=5):
#     if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "font/simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


# if __name__ == '__main__':
#     img = cv2ImgAddText(cv2.imread('1.jpg'), "大家好，我是片天边的云彩", 10, 65, (0, 0 , 139), 10)
#     cv2.imshow('show', img)
#     if cv2.waitKey(100000) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()


# ------------------------建个文件夹中图片加载到txt     xxx.jpg  id ----------------------
# import os
# path = r'负样本'
# dire = os.listdir(path)
# write_filename = r"{}.txt".format(path)
# with open(write_filename,'w')  as writer:
#     for file in dire:
    
#         writer.write('{},0,0,0,0,0,0,1'.format(file))
#         writer.write('\n')



#-------------------将多个文件夹中的文件汇总到一个文件夹-----------------------

# import os
# import shutil
# import random
# mypath = r"./图片助手"
# write_path = './负样本/'.format(mypath)
# if not os.path.isdir(write_path):
#     os.makedirs(write_path)
  
# for root,dirs,files in os.walk(mypath):
#     # for dr in dirs:
#         # print(dr)
#     for name in files:
#         # if name.endswith(".txt"):
#         filename = os.path.join(root, name)
#         # print(filename)
#         shutil.copy(filename,'{}{}.jpg'.format(write_path,random.random()))

    

# ---------------------贴图-------------------------------------------

# from PIL import Image
# import os
# import random
# read_path = r'./贴图/'
# save_path = './save/'
# # 背景图片
# ground = '0.jpg'

# def resize_img(img,add_img,file):
#     add_img_w = add_img.size[0]
#     add_img_h = add_img.size[1]

#     img_w = img.size[0]
#     img_h = img.size[1]
#     while True:
#         if add_img_w<img_w and add_img_h<img_h:
#             h = int((img_h-add_img_h)/2)
#             w = int((img_w-add_img_w)/2)
#             img.paste(add_img,(w,h))
#             img.save('{}/{}.jpg'.format(save_path,random.random()))
#             break
#         else:
#             add_img = add_img.resize((int(add_img_w/2),int(add_img_h/2)))
#             add_img_w = add_img.size[0]
#             add_img_h = add_img.size[1]
#             continue


# dire = os.listdir(read_path)
# for file in dire:
#     path = ground
#     img = Image.open(path)
#     markImg = Image.open(read_path+file)
#     markImg_w = markImg.size[0]
#     markImg_h = markImg.size[1]
#     # 随机对图片进行缩放
#     count = random.uniform(2,3)
#     w = round(markImg_w*count,1)
#     h = round(markImg_h*count,1)

#     # 原图
#     resize_img(img,markImg,file)
#     # 将图像旋转指定尺寸
#     markImg = markImg.resize((int(w),int(h)))
#     add_img = markImg.rotate(45)
#     resize_img(img,add_img,file)
#     # 上下镜像
#     markImg = markImg.resize((int(w),int(h)))
#     add_img = markImg.transpose(Image.FLIP_TOP_BOTTOM)
#     resize_img(img,add_img,file)
#     # 左右镜像
#     markImg = markImg.resize((int(w),int(h)))
#     add_img = markImg.transpose(Image.FLIP_LEFT_RIGHT)
#     resize_img(img,add_img,file)
#     # 90度镜像
#     path = ground
#     img = Image.open(path)
#     markImg = markImg.resize((int(w),int(h)))
#     add_img = markImg.transpose(Image.ROTATE_90)
#     resize_img(img,add_img,file)
#     # 180度镜像
#     path = ground
#     img = Image.open(path)
#     markImg = markImg.resize((int(w),int(h)))
#     add_img = markImg.transpose(Image.ROTATE_180)
#     resize_img(img,add_img,file)
#     # 颠倒
#     path = ground
#     img = Image.open(path)
#     markImg = markImg.resize((int(w),int(h)))
#     add_img = markImg.transpose(Image.TRANSPOSE)
#     resize_img(img,add_img,file)
    



 
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


