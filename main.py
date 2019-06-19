import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import keras
import cv2
import boto3
from skimage.measure import compare_ssim, compare_psnr
from foolbox.models import KerasModel

import utils

def setup_local_model():
    #sets up local ResNet50 model, to use for local testing
    keras.backend.set_learning_phase(0)
    kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    model = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing, predicts='logits')
    return model

log_every_n_steps = 100 #log progress to console every n steps
query_limit = 5000 #set to None for queryless setting
epsilon = 64
size = 8

setting = 'targeted'
target_system = 'AWS'

################## -- START OF ATTACK -- #######################

x_val = np.load("data/x_val_1000.npy") #loads 1000 instances of the ImageNet validation set
y_val = np.load("data/y_val_1000.npy") #loads labels of the 1000 instances of the ImageNet validation set

print('loading untargeted and targeted splits...')
local_untargeted_split = utils.pickle_load('data/untargeted_split.pickle') #the indices of the random split of the images we will be testing
local_targeted_split = utils.pickle_load('data/targeted_split.pickle') #the indices of the random split of the images we will be testing
api_classifiers_untargeted_split = utils.pickle_load('data/online_api_classifiers_untargeted_split.pickle')
api_classifiers_targeted_split = utils.pickle_load('data/online_api_classifiers_targeted_split.pickle')

print('starting simba attack...')
print('epsilon: {} ({:.2%})'.format(epsilon, epsilon/255))
print('size: {}, {}, max directions: {} ({:.2%})'.format(size, size, 224*224*3/size/size, 1/size/size))

if target_system == 'local_resnet50':
    #local model testing
    local_model = setup_local_model()
    untargeted_split = local_untargeted_split
    targeted_split = local_targeted_split
elif target_system == 'AWS' or target_system == 'GCV':
    local_model = None
    untargeted_split = api_classifiers_untargeted_split
    targeted_split = api_classifiers_targeted_split
else:
    raise Exception('target_system should be set to "local_resnet50", or "AWS" or "GCV"')

if setting == 'untargeted':
    target_class = None
    split = untargeted_split
elif setting == 'targeted':
    split = targeted_split.T

print(split)

for i in split: #loop through all images in the split
    df_filename = 'pickles/{}_{}_SimBA_{}_{}_img{}.pickle'.format(str(target_system), str(setting), str(epsilon), str(size), str(i))
    file_exists = os.path.isfile(df_filename)
    if file_exists:
        print('file for image {} already exists'.format(i))
    else:
        if setting == 'targeted':
            #unpack targeted labels
            target_class = i[1]
            i = int(i[0])
        print(i)
        original_image = x_val[i] #img must be bgr
        original_class = y_val[i]
        start = time.time()
        
        adv, total_calls, info_df = utils.run_sparse_simba(original_image, size=size, epsilon=epsilon, setting=setting,
                                            query_limit=query_limit, target_system=target_system,
                                            target_class=target_class, local_model=local_model,
                                            log_every_n_steps=log_every_n_steps) #set size=1 for original SimBA
        
        print('total time taken: {}s'.format(time.time()-start))
        print('total queries made: {}'.format(total_calls))
        
        #improve these last 3 before AWS
        utils.pickle_save(info_df, df_filename)
        # img_filename = df_filename[:-7] + '.png'
        # save_adv_details_simba(original_image, adv, total_calls, aws_model, img_filename)
