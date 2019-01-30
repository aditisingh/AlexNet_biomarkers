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
    im.extend(scaled_imgs)
    im.extend(flip_imgs)
    im.extend(rotate_imgs)
    files1 = []
    labels = []
    files1.extend(filenames)
    labels.extend(lbl)
    for i in range(4):
        files1.extend(filenames)
        labels.extend(lbl)
    return im, labels, files1

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


d_path=base_path
d_im, d_lbl, _ = load_info(d_path) 
d_im=np.array(d_im)
#Training alexnet

shuffle_ids = random.sample(range(len(d_lbl)), int(0.8*len(d_lbl)))

tr_im = [d_im[idx] for idx in shuffle_ids]
tr_lbl = [d_lbl[idx] for idx in shuffle_ids]

v_im = [d_im[idx] for idx in range(len(d_lbl)) if idx not in shuffle_ids]
v_lbl = [d_lbl[idx] for idx in range(len(d_lbl)) if idx not in shuffle_ids]

t1 = to_categorical(tr_lbl,6)
f1 = np.array(tr_im)

t2 = to_categorical(v_lbl,6)
f2 = np.array(v_im)

network = input_data(shape=[None, 50,50, 7])
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

tr_err=get_error(model,f1,t1,'training_new_data')
val_err=get_error(model,f2,t2,'validation_new_data')
