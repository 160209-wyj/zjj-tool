'''
灰度图也有3通道，将1通道转为3通道  其他两个通道为0
彩图也有4通道，将4通道转为3通达    多了alpha通道
'''

import tqdm
import imghdr
import os
from PIL import Image
# 通道转换
def change_image_channels(image, image_path):
 # 4通道转3通道
    if image.mode == 'RGBA':
         print('image:',image,'image_path:',image_path)
         r, g, b, a = image.split()
         image = Image.merge("RGB", (r, g, b))
         image.save(image_path)
#1 通道转3通道
    elif image.mode != 'RGB':
        print('image:',image,'image_path:',image_path)
        image = image.convert("RGB")
        os.remove(image_path)
        image.save(image_path)

if __name__ == "__main__":
    
    for curdir, subdirs, files in os.walk('./train'):
        # print(files)
        for file in tqdm.tqdm(files):
            # if file.endswith('.jpg') and file.endswith('.jpeg'):
            # print('===================================')
            path = os.path.join(curdir, file)
            file_ = path.split('\\')
            file = '/'.join(file_)
            typ = imghdr.what('./{}'.format(file))
            if typ == 'gif':  # 填写规则
                os.remove('./{}'.format(file))
                continue
            image = Image.open('./{}'.format(file))
            
            new_image = change_image_channels(image, "{}".format(file))