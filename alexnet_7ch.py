from __future__ import division, print_function, absolute_import
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
# import pandas as pd
import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
import scipy
from PIL import Image
from sklearn.metrics import accuracy_score
import os, random, time
from scipy.misc import imread, imresize
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.metrics import average_precision_score, precision_recall_curve
from img_aug import central_scale_images, rotate_images, flip_images, add_salt_pepper_noise

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

run=1

# storing
count=[]
train_error = []
valid_error = []
train_statistics = {'precision:': [], 'recall': [], 'fscore': []}
valid_statistics = {'precision:': [], 'recall': [], 'fscore': []}



def prasad_map(t,count,iter=0):
    k=1
    num_class=2
    count1=np.zeros(1+num_class).tolist()
    count1[0]=iter
    for i in range(1,1+num_class):
        for lbl in t:
            if lbl==str(i-1):
                count1[i] += 1
    if len(count)>0:
        count1[1]+=count[-1][1]
        count1[2]+=count[-1][2]
    count.append(count1)
    # plt.figure()
    count_arr=np.array(count).T
    # if iter==0 or iter%k==0: #try not to plot all, only the ones divisible by k
    #     plt.bar(count_arr[0], label='Benign', height=count_arr[1], color='g', align='edge')
    #     plt.bar(count_arr[0], label='Malignant', height=count_arr[2], color='b', align='edge')
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig('prasad_map_alex_ere1.png')
        # plt.close()
    return count
#
# def load_info(dirpath):  # image, labels, filename, features
#     files = os.listdir(dirpath)
#     im = []
#     lbl = []
#     filenames = []
#     for f in files:
#         # print(f)
#         im.append(imresize(imread(dirpath + f),(224,224,3)))
#         lbl.append(f[0])
#         filenames.append(dirpath + f)
#     return im, lbl, filenames

def load_info(dirpath):  # image, labels, filename, features
    files = os.listdir(dirpath)
    im = []
    lbl = []
    filenames = []
    lbl_names=['neun','gfap','s100','apc','iba','reca1']
    for f in files:
        # print(f)
        img=Image.open(dirpath+f)
        imgArray=np.zeros((img.n_frames,50,50),np.uint8)
        for frame in range(img.n_frames):
            img.seek(frame)
            img1 = img.resize((50,50))#,Image.ANTIALIAS)
            imgArray[frame,:,:] = img1
            # frame = frame + 1
        im.append(imgArray.reshape(50,50,img.n_frames))
        cl_indices=[f.index(ll)+len(ll)+1 for ll in lbl_names]
        cl_lbl=[int(f[id]) for id in cl_indices]
        if cl_lbl[0]==1:
            lbl.append(1)
        elif cl_lbl[1]==1:
            lbl.append(2)
        elif cl_lbl[3]==1:
            lbl.append(3)
        elif cl_lbl[4]==1:
            lbl.append(4)
        elif cl_lbl[5]==1:
            lbl.append(5)
        else:
            lbl.append(0)
        filenames.append(dirpath + f)
    scaled_imgs = central_scale_images(im, [0.9])
    flip_imgs = flip_images(im)
    rotate_imgs = rotate_images(im, -90, 90, 2)
    # salt_imgs = add_salt_pepper_noise(im)
    im.extend(scaled_imgs)
    im.extend(flip_imgs)
    im.extend(rotate_imgs)
    # im.extend(salt_imgs)
    files1 = []
    labels = []
    files1.extend(filenames)
    labels.extend(lbl)
    for i in range(4):
        files1.extend(filenames)
        labels.extend(lbl)
    return im, labels, files1
    # return im, lbl, filenames

def get_statistics(y_score, y_pred, y_lbl, label):
    y_lbl1=[str(ll) for ll in np.argmax(y_lbl,axis=1)]
    precision1, recall1, fscore1, support1 = score(y_lbl1, [str(i) for i in y_pred])
    df = {label + 'precision': precision1.tolist(), label + 'recall': recall1.tolist(),
          label + 'fscore': fscore1.tolist()}
    np.save(label + '_statistics_prasad_'+str(run)+'.npy', df)
    print('-------------------')
    print(label + ' Performance')
    print('-------------------')
    print('Nucleus Neurons Astrocytes Oligodendrocytes Microglia Endothelials')
    print('Precision: ', precision1)
    print('Recall   : ', recall1)
    print('F-score  : ', fscore1)


def get_error(model,f,t,label):
    pred_probs=[model.predict(f[i*500:min((i+1)*500,len(f))]) for i in range(int(len(f)/500)+1)]
    y=[val for sublist in pred_probs for val in list(sublist)]
    # y=model.predict(f)
    yy=np.argmax(y,axis=1)
    acc=accuracy_score(t,to_categorical(yy,6))
    get_statistics(y, yy, t, label)
    return 1-acc

tf.reset_default_graph()
base_path='/home/cougarnet.uh.edu/asingh42/Aditi/new_crops_7channel/'

# v_path=base_path+'valid/'
# t_path=base_path+'test/'
d_path=base_path
# g_path=base_path+'train/golden/'
# s_path=base_path+'train/seed/'

# v_im, v_lbl, _ = load_info(v_path) #validation data
# t_im, t_lbl, _ = load_info(t_path) #testing data
d_im, d_lbl, _ = load_info(d_path) #mturk data
# g_im, g_lbl, _ = load_info(g_path) #golden data
# s_im, s_lbl, _ = load_info(s_path) #seed data

d_im=np.array(d_im)
#Training alexnet

#normalize data
# mean_d=np.mean(d_im,axis=0)
# std_d=np.std(d_im,axis=0)
#
# for i in range(len(d_im)):
#     d_im[i]=1.0*(d_im[i]-mean_d)/(std_d+1e-8)

# g_im=(g_im-mean_d)/std_d
# s_im=(s_im-mean_d)/std_d
#
# for i in range(len(t_im)):
#     t_im[i]=(t_im[i]-mean_d)/std_d
#
# for i in range(len(v_im)):
#     v_im[i]=(v_im[i]-mean_d)/std_d


# tflearn.init_graph(gpu_memory_fraction=1)

shuffle_ids = random.sample(range(len(d_lbl)), int(0.8*len(d_lbl)))

tr_im = [d_im[idx] for idx in shuffle_ids]
tr_lbl = [d_lbl[idx] for idx in shuffle_ids]

v_im = [d_im[idx] for idx in range(len(d_lbl)) if idx not in shuffle_ids]
v_lbl = [d_lbl[idx] for idx in range(len(d_lbl)) if idx not in shuffle_ids]

# tr_files = [g_files[idx] for idx in shuffle_ids]
# tr_info = [patient_mag_info[idx] for idx in shuffle_ids]

t1 = to_categorical(tr_lbl,6)
f1 = np.array(tr_im)

t2 = to_categorical(v_lbl,6)
f2 = np.array(v_im)


# img_aug=ImageAugmentation()
# img_aug.add_random_flip_leftright()
# img_aug.add_random_blur(sigma_max=3.)
# img_aug.add_random_rotation(max_angle=25.)
# img_aug.add_random_flip_updown()
network = input_data(shape=[None, 50,50, 7])#, data_augmentation=img_aug)
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.6)
network1 = fully_connected(network, 4096, activation='tanh',name='layer')
network = dropout(network1, 0.6)
network = fully_connected(network, 6, activation='softmax')
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)
model = tflearn.DNN(network,tensorboard_verbose=3)

model.load('alexnet_new_data_' + str(run) + '.tflearn')
tr_err=get_error(model,f1,t1,'training_new_data')
val_err=get_error(model,f2,t2,'validation_new_data')

model.fit(f1, t1, n_epoch=150, validation_set=0.2, shuffle=True, show_metric=True, batch_size=2048, snapshot_step=200, snapshot_epoch=False, run_id='alexnet_training1')
model.save('alexnet_new_data_' + str(run) + '.tflearn')

#model.load('alexnet_new_data_' + str(run) + '.tflearn')



# count=prasad_map(g_lbl,count,0)
#
tr_err=get_error(model,f1,t1,'training_new_data')
val_err=get_error(model,f2,t2,'validation_new_data')
# tr_err=get_error(model,f3,t3,'training_prasad')
#
#
# # training_errs.append(tr_err)
# # validation_errs.append(val_err)
#
# #testing
# t4=to_categorical(t_lbl,2)
# f4=np.array([imresize(t_im1,(224,224)) for t_im1 in t_im])
#
# # f3=resize_batch(f3,50)
#
# tst_err=get_error(model,f4,t4,'testing_prasad')
# print(tr_err,val_err,tst_err)
