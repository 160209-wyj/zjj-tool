# coding: utf-8
import os
import pandas as pd
import json
from PIL import Image
import urllib
import requests
from io import BytesIO
import tqdm
import cv2
import numpy as np
import urllib.request

def txt2xml(img,labels_dict):

 
    # write the region of image on xml file
    
    # spt = img_each_label.split(' ') #这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
    xml_file.write('    <object>\n')
    xml_file.write('        <name>' + str(labels_dict["label"]) + '</name>\n')
    xml_file.write('        <pose>Unspecified</pose>\n')
    xml_file.write('        <truncated>0</truncated>\n')
    xml_file.write('        <difficult>0</difficult>\n')
    xml_file.write('        <bndbox>\n')
    xml_file.write('            <xmin>' + str(labels_dict["x1"]) + '</xmin>\n')
    xml_file.write('            <ymin>' + str(labels_dict["y1"]) + '</ymin>\n')
    xml_file.write('            <xmax>' + str(labels_dict["x2"]) + '</xmax>\n')
    xml_file.write('            <ymax>' + str(labels_dict["y2"]) + '</ymax>\n')
    xml_file.write('        </bndbox>\n')
    xml_file.write('    </object>\n')
 
    # xml_file.write('</annotation>')
    return xml_file

def urllib_download(IMAGE_URL,save_path):
    from urllib.request import urlretrieve
    urlretrieve(IMAGE_URL, save_path)
claess = ['红头文件','公章','显示器_有内容','正常文件','工程图纸','投影仪_有内容']
with open("labels.txt") as f:
    line_list = f.readlines()
    for line in tqdm.tqdm(line_list):
        labels_list = []
        labels_dict = {}
        data = json.loads(line)
        content = data["content"]
        taskName = data["taskName"] #公章
        name = data["name"] #图片名
        path = data["path"] #图片地址
        # response = requests.get(path)
        # tmpIm = BytesIO(response.content)
        try:
            resp = urllib.request.urlopen(path)
        except:
            print(path)
            continue
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # img = Image.open(tmpIm)
        file_name = name.split(".")
        width = img.shape[0]
        height = img.shape[1]
        src_xml_dir = 'E:\\所有文件的多标签\\save_img\\gctz_xml'
        xml_file = open((src_xml_dir + '\\' + file_name[0] + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str(file_name[0]) + '.png' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        for i in content:
            if i["labelName"] == '红头文件':
                labelName = 'red_file'
                labels_dict["label"] = labelName
            if i["labelName"] == '公章':
                labelName = 'gz'
                labels_dict["label"] = labelName
            if i["labelName"] == '显示器_有内容':
                labelName = 'xxq'
                labels_dict["label"] = labelName
            if i["labelName"] == '正常文件':
                labelName = 'file'
                labels_dict["label"] = labelName
            if i["labelName"] == '工程图纸':
                labelName = 'gctz'
                labels_dict["label"] = labelName
            if i["labelName"] == '投影仪_有内容':
                labelName = 'tyy'
                labels_dict["label"] = labelName
            if i["labelName"] not in claess:
                labelName = 'normal'
                labels_dict["label"] = labelName
            if labelName in labels_list:
                continue
            else:
                labels_list.append(labelName)
            file_name = name.split(".")
            area = i["area"]
            try:
                x1 = area['x']
                y1 = area['y']
                x2 = x1 + area['xlen']
                y2 = y1 + area['ylen']
                labels_dict["x1"] = x1
                labels_dict["y1"] = y1
                labels_dict["x2"] = x2
                labels_dict["y2"] = y2
            except:
                print(file_name)
            # out_file = open('D:\\data\\logo\\gongzhang\\txt/%s.txt' % (file_name[0]), 'w')
            # out_file.write(str(102) + '\n' + str(x1) + '\n' + str(y1) + '\n' + str(x2) + '\n' + str(y2) + '\n')
            # out_file.close()
            # img = Image.open(urllib.request.urlopen(path))
            
            xml_file = txt2xml(file_name[0],labels_dict)
        xml_file.write('</annotation>')
        urllib_download(path,r'E:\\所有文件的多标签\\save_img\\img\\{}'.format(name))