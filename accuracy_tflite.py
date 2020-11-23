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
import tflite_runtime.interpreter as tflite
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
    
    interpreter = tflite.Interpreter(model_path=args.model_name + '.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    classified = 0
    misclassified = 0
    for x in os.listdir(args.validate_dataset):
        raw_data = cv2.imread(os.path.join(args.validate_dataset, x))
        greyscale_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
        greyscale_data = cv2.adaptiveThreshold(greyscale_data,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        greyscale_data = np.array(greyscale_data, dtype=np.float32)
        image = np.array(greyscale_data) / 255.0
        (h, w) = image.shape
        input_data = image.reshape([-1, h, w, 1])
        true = x.split('.')[0]
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = []
        for i in range(6):
            data = interpreter.get_tensor(output_details[i]['index'])
            output_data.append(data)
        pred = decode(captcha_symbols, output_data)
        classified += 1
        if pred != true:
            misclassified += 1
     
    accuracy = misclassified / float(classified)
    print('Accuracy: ' + str(accuracy))
    
    
                  
if __name__ == '__main__':
    main()