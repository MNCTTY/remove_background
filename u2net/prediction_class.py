import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
class Predict:
    def __init__(self, model_name):
        model_dir = os.path.join(os.getcwd(), 'saved_models','u2net',  model_name + '.pth')
        self.model_name = model_name

        self.net = U2NET(3, 1)


        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.net.eval()
        print('Model loaded!')
    # normalize the predicted SOD probability map

    def save_output(self, image_name,pred,d_dir):

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np*255).convert('RGB')
        img_name = image_name.split(os.sep)[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

        pb_np = np.array(imo)

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        imo.save(d_dir+self.model_name+imidx+'.png')

    def normPRED(self,d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn

    def predict(self, image_path):

        # --------- 1. get image path and name ---------
        #image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
        #prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)

        print(image_path)
        img_name_list = []
        img_name_list.append(image_path)
        prediction_dir = (os.path.dirname(os.path.abspath(image_path)) + os.sep)
        # --------- 2. dataloader ---------
        #1. dataloader
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)




        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= self.net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            pred = self.normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            self.save_output(img_name_list[i_test],pred,prediction_dir)

            del d1,d2,d3,d4,d5,d6,d7


if __name__ == '__main__':
    image = '/home/ubuntu/removebg/test.jpg'
    model_name = 'u2net'
    prediction = Predict(model_name)
    prediction.predict(image)
