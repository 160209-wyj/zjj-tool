import os
import random
import cv2
import pandas as pd
import csv

import os
import shutil
import random
#------------------------读取txt文件行数-------------------------------
def read_txtline(txt_file):
	filename = txt_file
	myfile = open(filename) 
	lines = len(myfile.readlines()) 
	print(lines)




#-----------------打乱txt每一行--------------

def shuffle_txt(read_txt,shuffle_txt):
	out = open(shuffle_txt,'w')
	lines=[]
	with open(read_txt, 'r',encoding='utf-8') as infile:
		for line in infile:
			lines.append(line)
	random.shuffle(lines)
	for line in lines:
		out.write(line)






#------------------------------随机从txt传到另一个txt--------------------------
def tran_txt(read_txt,write_txt):
	read_filename = read_txt
	write_filename = write_txt
	with open(read_filename,'r',encoding='utf-8')  as read:
		with open(write_filename,'a')  as writer:
			data = read.readlines()
			for line_data in data:
				writer.write(line_data)


#------------------------------------在图片之上添加文字------------------------


def puttext(pic):
	#加载背景图片
	bk_img = cv2.imread(pic)

	w = bk_img.shape[0]
	h = bk_img.shape[1]
	y0, dy = int(w*0.1), int(h*0.1)
	#在图片上添加文字信息
	text = "[('red', 0.665), ('gz', 0.843)]"
	for i, txt in enumerate(text.split(')')):
		y = y0+i*dy
		cv2.putText(bk_img,txt, (int(w*0.1),y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)
	#显示图片
	cv2.imshow("add_text",bk_img)
	cv2.waitKey()
	#保存图片
	cv2.imwrite("add_text.jpg",bk_img)




#----------------------读取csv  并写入另一个csv----------------------
def tran_csv(read_csv,write_csv):
	with open(write_csv,'a') as txt:
		with open(read_csv, 'r', encoding="utf-8") as f:
			# data = pd.read_csv("labels.csv")
			reader = csv.reader(f)
			# print(type(reader))
		
			for row in list(reader)[1::10]:
				row1 = ','.join(row)
				# print(row)
				txt.write(row1)
				txt.write('\n')
				# data.drop(row)




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
def add_normal(path):
	
	dire = os.listdir(path)
	write_filename = r"{}.txt".format(path)
	with open(write_filename,'w')  as writer:
		for file in dire:
		
			writer.write('{},0,0,0,0,0,0,1'.format(file))
			writer.write('\n')



#-------------------将多个文件夹中的文件汇总到一个文件夹-----------------------
def copy_pic(read_path,write_path):
	mypath = read_path
	write_path = write_path
	if not os.path.isdir(write_path):
		os.makedirs(write_path)
	
	for root,dirs,files in os.walk(mypath):
		# for dr in dirs:
			# print(dr)
		for name in files:
			# if name.endswith(".txt"):
			filename = os.path.join(root, name)
			# print(filename)
			shutil.copy(filename,'{}{}.jpg'.format(write_path,random.random()))







# read_txtline("csv_more_labels.txt")    #读取txt文件行数



shuffle_txt("train.txt","train_shuffle.txt")   # 打乱txt每一行

#   
# tran_txt('read.txt','write.txt')    #  随机从txt传到另一个txt


# puttext('1.jpg')   # 在图片之上添加文字


# tran_csv('read.csv','write.txt')   # 读取csv  并写入另一个csv


# add_normal('./normal')    # 建个文件夹中图片加载到txt     xxx.jpg  id


# copy_pic(r'./图片助手','./normal')   # 将多个文件夹中的文件汇总到一个文件夹

    

 


