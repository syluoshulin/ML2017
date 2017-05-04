import os
import sys
import argparse
from keras.models import load_model
#from termcolor import colored,cprint
from keras import backend as K
#from utils import *
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.realpath(__file__))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
orig_see_dir = os.path.join(img_dir, 'orig')
if not os.path.exists(orig_see_dir):
    os.makedirs(orig_see_dir)
model_dir = os.path.join(base_dir, 'model')

def read_data(data_path):
    with open(sys.argv[1]) as trainFile:
            trainList = trainFile.read().splitlines()
            train_arr = np.array([line.split(",") for line in trainList])
            x_arr = train_arr[1:,1]
            y_arr = train_arr[1:,0]
            x_arr = np.array([str(line).split() for line in x_arr])
            y_arr = np.array([str(line).split() for line in y_arr])

            x_train_data = x_arr.reshape(x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)
            y_train_data = y_arr.astype(np.int)#(28709,1)

    x_train_data /= 255

    ratio = 0.9
    xSize = int(x_train_data.shape[0]*ratio)
    ySize = int(y_train_data.shape[0]*ratio)
    x_val_data = x_train_data[xSize:]
    y_val_data = y_train_data[ySize:]

    return x_val_data, y_val_data

def main():
    #parser = argparse.ArgumentParser(prog='plot_saliency.py',
    #        description='ML-Assignment3 visualize attention heat map.')
    #parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    #args = parser.parse_args()
    #model_name = "model-%s.h5" %str(args.epoch)
    model_path = os.path.join(base_dir, "model.h5")
    emotion_classifier = load_model(model_path)
    #print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    #private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    private_pixels, private_label = read_data(sys.argv[1])
    #[ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) for i in range(len(private_pixels)) ]
    input_img = emotion_classifier.input
    img_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]#["image ids from which you want to make heatmaps"]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx].reshape(1,48,48,1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''

        heatmap = np.array(fn([private_pixels[idx].reshape(1,48,48,1), 1])).reshape(48,48)
        heatmap = np.absolute(heatmap)
        heatmap = (heatmap.astype(float) - np.amin(heatmap)) / (np.amax(heatmap) - np.amin(heatmap))

        '''
        step = 1
        temp = np.copy(private_pixels[idx])
        for i in range(20):
            loss_value, grads_value = fn([temp.reshape(1,48,48,1)])
            temp += grads_value * step
        heatmap = temp
        '''

        thres = 0.5
        see = np.copy(private_pixels[idx].reshape(48, 48))
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow((private_pixels[idx].reshape(48,48)), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(orig_see_dir, '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()
