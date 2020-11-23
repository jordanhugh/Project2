#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import string
import random
import csv
import argparse
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import matplotlib.pyplot as plt

def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    with open(args.output, 'w') as output_file:
        writer = csv.writer(output_file)

        model = tf.keras.models.load_model(args.model_name + '.h5')
        
        for itr, x in enumerate(os.listdir(args.captcha_dir)[:50]):
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            greyscale_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
            processed_data = cv2.adaptiveThreshold(greyscale_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_data = np.array(processed_data) / 255.0
            (h, w) = processed_data.shape
            input_data = processed_data.reshape([-1, h, w, 1])
            prediction = decode(captcha_symbols, model.predict(input_data))
            prediction = prediction.replace(' ', '')
            
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Prediction: [' + prediction + ']', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(raw_data)
            plt.show()
            prediction = input('Enter your Prediction: ')
            writer.writerow([x, prediction])
            print(str(itr) + ': Classified ' + x + ' as ' + prediction + " ")
            
    print("Pushing " + args.output + " to GitHub")
    os.system("git add " + args.output)
    os.system("git commit -m \"Updated Predictions\"")
    os.system("git push")
    print("Finished pushing " + args.output + " to GitHub")

    

if __name__ == '__main__':
    main()
