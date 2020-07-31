'''
数据增强代码
'''

# # from PIL import Image 
# # import numpy as np 
# # import matplotlib.pyplot as plt
# # import tqdm
# # import os
# # import cv2

# # #将图片数据增强 图片反转
# # path = './images/'
# # fileList = os.listdir(path)
# # # count = len(fileList)
# # # print(fileList)
# # for i,file in enumerate(tqdm.tqdm(fileList)):
# #     print('file:',file)
# #     if file.endswith('.jpg') or file.endswith('.png'):
# #       img = cv2.imread(path+file)
# #       flipped_img = np.fliplr(img)
# #       cv2.imwrite('./fu/{}.jpg'.format(i),flipped_img)




from PIL import Image
import os
import os.path

rootdir = '../images'  # 指明被遍历的文件夹
for parent, dirnames, filenames in os.walk(rootdir):
    for count,filename in enumerate(filenames):
        # print('parent is :' + parent)
        # print('filename is :' + filename)
        # currentPath = os.path.join(parent, filename)
        currentPath = rootdir+'/'+filename
        # print('the fulll name of the file is :' + currentPath)
        
        im = Image.open(currentPath)
        
        # 进行上下颠倒
        out = im.transpose(Image.FLIP_TOP_BOTTOM)
        newname = '../save/nf/sx{}new.jpg'.format(count)
        out.save(newname)
        #进行左右颠倒
        # out =im.transpose(Image.FLIP_LEFT_RIGHT)
        # newname = '../save/gun/sx{}new.jpg'.format(count)
        # out.save(newname)
        # # 进行旋转90
        # out = im.transpose(Image.ROTATE_90)
        # newname = './aaa/90'+ filename
        # out.save(newname)
        # # 进行旋转180
        # out = im.transpose(Image.ROTATE_180)
        # newname = './aaa/180'+ filename
        # out.save(newname)
        # # 进行旋转270
        # out = im.transpose(Image.ROTATE_270)
        # newname = './aaa/270'+ filename
        # out.save(newname)
        # #将图片重新设置尺寸
        # out= out.resize((1280,720))
        
        


















