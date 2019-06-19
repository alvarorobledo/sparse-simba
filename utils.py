#functions and utils go here
import os
import sys
import io
import numpy as np
import pandas as pd
import pickle
import time
import keras
import cv2
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr
import boto3 #AWS python client library
from google.cloud import vision #GCV python client library

#set GCV credentials as environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Alvaro\Dropbox\Imperial\FYP\GCV\credentials.json"

def pickle_save(x, file_name):
    #custom function to save variables locally
    pickle_out = open(file_name, 'wb')
    pickle.dump(x, pickle_out, -1)
    pickle_out.close()

def pickle_load(file_name):
    #custom function to load variables
    pickle_in = open(file_name, 'rb')
    x = pickle.load(pickle_in)
    pickle_in.close()
    return x

def run_sparse_simba(original_image, size=1, epsilon=64, setting='untargeted', query_limit=5000, target_system='local_resnet50', target_class=None, local_model=None, log_every_n_steps=200):
    sys.setrecursionlimit(max(1000, int(224*224*3/size/size))) #for deep recursion diretion sampling
    top_preds = get_top_preds(original_image, target_system, local_model)
    top_1_idx = np.argmax(top_preds[:,1].astype(np.float))
    original_class = top_preds[top_1_idx][0]
    if setting == 'untargeted':
        p = top_preds[top_1_idx][1]
        loss_label = original_class
    elif setting == 'targeted':
        idx = np.argwhere(target_class==top_preds[:,0]) #positions of occurences of label in preds
        if len(idx) == 0:
            print("{} does not appear in top_preds".format(target_class))
            p = 0
        p = top_preds[idx[0][0]][1]
        loss_label = target_class
    info_df = pd.DataFrame(columns=['iterations','total calls',
                                'epsilon','size', 'is_adv',
                                'ssim', 'psnr', 'image', 'probs'])
    total_calls = 0
    delta = 0
    is_adv = 0
    iteration = 0
    done = []

    #save step 0 in df
    adv = np.clip(original_image + delta, 0, 255)
    ssim = compare_ssim(original_image, adv, multichannel=True)
    psnr = compare_psnr(original_image, adv, data_range=255)
    info_df = info_df.append({
        "iterations": iteration,
        "total calls": total_calls,
        "epsilon": epsilon,
        "size": size,
        "is_adv": is_adv,
        "ssim": ssim,
        "psnr": psnr,
        "image": original_image,
        "top_preds": top_preds
    }, ignore_index=True)

    start = time.time()

    while ((not is_adv) & (total_calls <= query_limit+5)): #buffer of 5 calls
        if iteration % log_every_n_steps == 0:
            print('iteration: {}, new p is: {}, took {:.2f} s'.format(str(iteration), str(p), time.time()-start))
        iteration += 1    

        q, done = new_q_direction(done, size=size)

        delta, p, top_preds, success = check_pos(original_image, delta, epsilon, q, p, loss_label, setting, target_system, local_model)
        total_calls += 1
        if not success:
            delta, p, top_preds, _ = check_neg(original_image, delta, epsilon, q, p, loss_label, setting, target_system, local_model)
            total_calls += 1

        #update data on df
        adv = np.clip(original_image + delta, 0, 255)
        ssim = compare_ssim(original_image, adv, multichannel=True)
        psnr = compare_psnr(original_image, adv, data_range=255)


        if iteration % 100 == 0: #only save image and probs every 100 steps, to save memory space
            image_save = adv
            preds_save = top_preds
        else:
            image_save = None
            preds_save = None
            
        info_df = info_df.append({
            "iterations": iteration,
            "total calls": total_calls,
            "epsilon": epsilon,
            "size": size,
            "is_adv": is_adv,
            "ssim": ssim,
            "psnr": psnr,
            "image": image_save,
            "top_preds": preds_save
        }, ignore_index=True)

        #check if image is now adversarial
        if (not is_adv) and (is_adversarial(adv, top_preds, setting, original_class, target_class)):
            is_adv=1
            info_df = info_df.append({
                "iterations": iteration,
                "total calls": total_calls,
                "epsilon": epsilon,
                "size": size,
                "is_adv": is_adv,
                "ssim": ssim,
                "psnr": psnr,
                "image": adv,
                "top_preds": top_preds
            }, ignore_index=True)   
            return adv, total_calls, info_df #remove this to continue attack even after adversarial is found
    
    return adv, total_calls, info_df

def check_pos(x, delta, epsilon, q, p, loss_label, setting, target_system, local_model):
    success = False #initialise as False by default
    pos_x = x + delta + epsilon * q
    pos_x = np.clip(pos_x, 0, 255)
    top_preds = get_top_preds(pos_x, target_system, local_model)
    idx = np.argwhere(loss_label==top_preds[:,0]) #positions of occurences of label in preds
    if len(idx) == 0:
        print("{} does not appear in top_preds".format(loss_label))
        return delta, p, top_preds, success
    idx = idx[0][0]
    p_test = top_preds[idx][1]
    if setting == 'untargeted':
        if p_test < p:
            delta = delta + epsilon*q #add new perturbation to total perturbation
            p = p_test #update new p
            success = True
        return delta, p, top_preds, success
    elif setting == 'targeted':
        if p_test > p:
            delta = delta + epsilon*q #add new perturbation to total perturbation
            p = p_test #update new p
            success = True
        return delta, p, top_preds, success
    raise Exception('setting should be set to either "untargeted" or "targeted"')

def check_neg(x, delta, epsilon, q, p, loss_label, setting, target_system, local_model):
    success = False #initialise as False by default
    neg_x = x + delta - epsilon * q
    neg_x = np.clip(neg_x, 0, 255)
    top_preds = get_top_preds(neg_x, target_system, local_model)
    idx = np.argwhere(loss_label==top_preds[:,0]) #positions of occurences of label in preds
    if len(idx) == 0:
        print("{} does not appear in top_preds".format(loss_label))
        return something
    idx = idx[0][0]
    p_test = top_preds[idx][1]
    if setting == 'untargeted':
        if p_test < p:
            delta = delta - epsilon*q #add new perturbation to total perturbation
            p = p_test #update new p 
            success = True
        return delta, p, top_preds, success
    elif setting == 'targeted':
        if p_test > p:
            delta = delta - epsilon*q #add new perturbation to total perturbation
            p = p_test #update new p
            success = True
        return delta, p, top_preds, success
    raise Exception('setting should be set to either "untargeted" or "targeted"')


def is_adversarial(image, top_preds, setting, original_class, target_class):
    #returns whether image is adversarial, according to setting
    top_probs = top_preds[:,1]
    top_1_idx = np.argmax(top_probs.astype(np.float))
    top_1_label = top_preds[top_1_idx][0]
    if setting == 'untargeted':
        if top_1_label != original_class: #remove this in final code
            print('image is now adversarial!')
            print(top_preds[top_1_idx])
            return True
        return top_1_label != original_class
    elif setting == 'targeted':
        if top_1_label == target_class:
            print('image is now adversarial!')
            print(top_preds[top_1_idx])
            return True
        return top_1_label == target_class
    raise Exception('setting should be set to either "untargeted" or "targeted"')

def get_top_preds(image, target_system, local_model):
    if target_system == 'AWS':
        top_preds = aws_predict(image)
    elif target_system == 'GCV':
        top_preds = gcv_predict(image)
    elif target_system == 'local_resnet50':
        probs = local_model.predictions(image[:,:,::-1])
        top_preds = np.array([np.arange(len(probs), dtype=np.int32), probs]).T
    return top_preds

def new_q_direction(done, size=1):
  [a,b,c] = sample_nums(done, size)
  done.append([a,b,c])
  if len(done) >= 224*224*3/size/size-2:
    done = [] #empty it before it hits recursion limit
  q = np.zeros((224,224,3))
  for i in range(size):
    for j in range(size):
      q[a*size+i, b*size+j, c] = 1
  q = q/np.linalg.norm(q)
  return q, done

def sample_nums(done, size=1):
  #samples new pixels without replacement
  [a,b] = np.random.randint(0, high=224/size, size=2)
  c = np.random.randint(0, high=3, size=1)[0]
  if [a,b,c] in done:
    #sample again (recursion)
    [a,b,c] = sample_nums(done, size)
  return [a,b,c]

def AWS_query_bytes_image(image):
    client=boto3.client('rekognition')
    response = client.detect_labels(Image={'Bytes': image}, MinConfidence=5)
    top_preds = [] 
    for label in response['Labels']:
        top_preds.append([label['Name'], label['Confidence']])
    top_preds = np.array(top_preds)
    return response, top_preds

def aws_predict(image):
    success, encoded_image = cv2.imencode('.png', image)
    response, top_preds = AWS_query_bytes_image(encoded_image.tobytes())
    return top_preds

def gcv_predict(image):
    pb_image = numpy_to_pb(image) #converts numpy image to protobuf
    client = vision.ImageAnnotatorClient()
    response = client.label_detection(image=pb_image, max_results=100)
    top_preds = []
    for label in response.label_annotations:
        top_preds.append([label.description, label.score])
    top_preds = np.array(top_preds)
    return top_preds

def numpy_to_pb(image):
    #converts a numpy RGB image into a protobuf
    assert type(image) is np.ndarray
    image_pil = Image.fromarray(image.astype(np.uint8))
    image_bytes_io = io.BytesIO()
    image_pil.save(image_bytes_io, format='PNG')
    content = image_bytes_io.getvalue()
    image_pb = vision.types.Image(content=content)
    return image_pb