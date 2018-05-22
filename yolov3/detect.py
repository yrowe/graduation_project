from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from hyper_para import cfg
from ipdb import set_trace

def init():
    #set_trace()
    batch_size = cfg.batch_size
    confidence = cfg.confidence
    nms_thresh = cfg.nms_thresh
    num_classes = cfg.num_classes
    classes = cfg.classes_name
    

    CUDA = torch.cuda.is_available()
    assert CUDA, "this code only support CUDA version"

    model = Darknet()
    model.module_list.load_state_dict(torch.load('yolov3.pth'))
    print("YOLO has been loaded")
    #set_trace()
    model.net_info["height"] = cfg.img_height
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0, "size of input image should be divided exactly by the maximum feature map size." 
    assert inp_dim > 32, "size of input image should greater than maximum feature map size."

    model.cuda()
    model.eval()

    return (batch_size, confidence, nms_thresh, num_classes, classes, CUDA, model, inp_dim)

batch_size, confidence, nms_thresh, num_classes, classes, CUDA, model, inp_dim = init()

def get_all_predict(img):
    #set_trace()
    model_inp = prep_image(img)
    im_dim = img.shape[1], img.shape[0]
    im_dim = torch.FloatTensor(im_dim).repeat(1,2)
    im_dim = im_dim.cuda()

    model_inp = model_inp.cuda()
    with torch.no_grad():
        output = model(model_inp, CUDA)

    output = write_results(output, confidence, num_classes, nms_conf = nms_thresh)
    if type(output) == int:
        return output

    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)

    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    #output = output[:, [0, 1, 3, 2, 4]]
    return output

def letterbox_image(img):
    '''resize image with unchanged aspect ratio using padding'''

    img_w, img_h = img.shape[1], img.shape[0]
    dim = (inp_dim, inp_dim)
    w, h = dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((dim[1], dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    #set_trace()
    img = (letterbox_image(img))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0


def print_rectangle(img, loc):
    for rec in loc:
        if rec[7] != 0.0:
            continue
        intRec = [int(i) for i in rec]
        cv2.rectangle(img, (intRec[1], intRec[2]), (intRec[3], intRec[4]), (255, 0, 0), 3)

    return img

if __name__ == '__main__':
    img = cv2.imread("img1.jpg")
    loc = get_all_predict(img)
    #print(type(loc))
    if type(loc) == int:
        print("no detections.")
        torch.cuda.empty_cache()
        exit(0)

    img = print_rectangle(img, loc)
    torch.cuda.empty_cache()
    #torch.cuda.empty_cache()
    save_img = 'tt.jpg'
    
    cv2.imwrite("{}".format(save_img), img)
    print("save at {}".format(save_img))
    
    cv2.namedWindow("detection",0)
    cv2.resizeWindow("detection", 1440, 900);
    
    cv2.imshow("detection",img)
    cv2.waitKey(0)
    exit(0)
    
    #print(loc)
    