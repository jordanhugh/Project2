#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import string
import random
import argparse
import csv
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.preprocessing import normalize

def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.validate_dataset is None:
        print("Please specify the path to the validation data set")
        exit(1)
        
    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)
    
    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()
        
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10, 7.5))
    ax = ax.flatten()
    fig.tight_layout(pad=1.0)
    fig.suptitle('Validation Dataset', fontsize=14)
    for itr, x in enumerate(os.listdir(args.validate_dataset)[:25]):
        ax[itr].set_xlabel(x.split('.')[0], fontsize=12)
        ax[itr].set_xticks([])
        ax[itr].set_yticks([])
        raw_data = cv2.imread(os.path.join(args.validate_dataset, x))
        ax[itr].imshow(raw_data)
    plt.show()

    model = keras.models.load_model(args.model_name + '.h5')
    
    classified = 0
    misclassified = 0
    for x in os.listdir(args.validate_dataset):
        img = cv2.imread(os.path.join(args.validate_dataset, x))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
        img = np.array(img) / 255.0
        (c, h) = img.shape
        img = img.reshape([-1, c, h, 1])
        true = x.split('.')[0]
        pred = decode(captcha_symbols, model.predict(img))
        classified += 1
        if pred != true:
            misclassified += 1
     
    accuracy = misclassified / float(classified)
    print('Accuracy: ' + str(accuracy)) 
    
    
                  
if __name__ == '__main__':
    main()