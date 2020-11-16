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
import tensorflow as tf
import tensorflow.keras as keras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to used for classification', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    model = keras.models.load_model(args.model_name + '.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(args.model_name + '.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Pushing " + args.model_name + ".tflite to GitHub")
    os.system("git add " + args.model_name + ".tflite")
    os.system("git commit -m \"Updated Model\"")
    os.system("git push")
    print("Finished pushing " + args.model_name + ".tflife to GitHub")

if __name__ == '__main__':
    main()