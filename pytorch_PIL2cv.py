import os
import numpy as np
import torch
from models import resnet_50
import cv2
from torchvision import datasets
np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def get_img_np_nchw(filename):  # 图片预处理
    image = cv2.imread(filename)
    # image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ### 省去RGB转换，直接使用opencv默认BGR
    image_cv = cv2.resize(image, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])

    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw.astype('float32')


def main_opencv():
    img_dir = './three_image_test'
    model_path = './copy_model_opencv.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load model
    model = resnet_50()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    a = []
    with torch.no_grad():
        for i, img_file in enumerate(os.listdir(img_dir)):
            print(img_file)
            input = get_img_np_nchw(os.path.join(img_dir, img_file))
            feature = model(torch.from_numpy(input).to('cuda'))
            a.append(feature.cpu().numpy().round(8)[0])
            print(a[i][:9])

    print("cosine similarity :", a[0].dot(a[1]), a[1].dot(a[2]), a[0].dot(a[2]))



def read_image(filename):
    image = cv2.imread(filename)
    return image


def preprocess(image):  # 图片预处理
    # image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ### 省去RGB转换，直接使用opencv默认BGR
    image_cv = cv2.resize(image, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])

    # img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return torch.from_numpy(img_np_t.astype('float32'))


def main():
    # img_dir = '/home/zhangchao/hd1/data/100Wpic/train1/'
    img_dir = '/home/zhangchao/disk3/data/100Wpic/test1'
    # img_dir = '/home/zhangchao/hd1/data/20w_video_pic'


    model_path = './copy_model_opencv.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size=512
    fileList = datasets.ImageFolder(img_dir, loader= read_image ,transform=preprocess)
    file_loader = torch.utils.data.DataLoader(fileList,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=10,pin_memory=True)

    #load model
    model = resnet_50()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # a = []
    with torch.no_grad():
        for i, files in enumerate(file_loader):

            feature = model(files[0].to('cuda'))  # 0 is image 1 is label
            # a.append(feature.cpu().numpy())
            for j in range(feature.size()[0]):
                # print(a[i][j][:9])
                print(f'{fileList.imgs[i*batch_size+j][0].split("/")[-1]} : {list(feature[j].cpu().numpy().round(4))}')


    # print("cosine similarity :", a[0].dot(a[1]), a[1].dot(a[2]), a[0].dot(a[2]))


if __name__ == "__main__":
    # main_opencv()
    main()
