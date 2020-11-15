#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import cv2
import numpy as np
import string
import random
import argparse
import csv
import keras
import matplotlib.pyplot as plt
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

    model = keras.models.load_model(args.model_name)
    
    occurrences = np.zeros(255)
    misclassified = np.zeros(255)
    for x in os.listdir(args.validate_dataset):
        img = cv2.imread(os.path.join(args.validate_dataset, x))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        img = np.array(img) / 255.0
        (c, h) = img.shape
        img = img.reshape([-1, c, h, 1])
        
        true = x.split('.')[0]
        pred = decode(captcha_symbols, model.predict(img))
        for i in range(len(true)): 
            occurrences[ord(true[i])] += 1
        if pred not in true:
            for i in range(len(pred)):
                if pred[i] != true[i]:
                    misclassified[ord(true[i])] += 1
    
    misclassified = normalize(misclassified.reshape(-1, len(misclassified)), norm='l1', axis=1).ravel()
    occurrences = normalize(occurrences.reshape(-1, len(occurrences)), norm='l1', axis=1).ravel()
    np.seterr(divide='ignore', invalid='ignore')
    weights = np.where(occurrences==0, 0, misclassified/occurrences)
    np.seterr(divide='ignore', invalid='warn')
    weights = normalize(weights.reshape(-1, len(weights)), norm='l1', axis=1).ravel()
    
    print('%32s\n' % ('Model Analysis'))
    print('%-10s %-15s %-14s %-11s' % ('Character','Misclassified', 'Occurances', 'Weights'))
    for symbol in captcha_symbols:
        code = ord(symbol)
        print('%-10c %-15s %-14s %-11s' % (symbol, "{:.2%}".format(misclassified[code]), "{:.2%}".format(occurrences[code]), "{:.2%}".format(weights[code])))
    print()
    
    print("Writing to weights.csv")
    row = np.zeros(len(captcha_symbols))
    for itr, symbol in enumerate(captcha_symbols):
            row[itr] = weights[ord(symbol)]
    with open('weights.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    print("Finished writing to weights.csv")
    
    
                  
if __name__ == '__main__':
    main()