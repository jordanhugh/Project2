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
        interpreter = tflite.Interpreter(model_path=args.model_name + '.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        
        for itr, x in enumerate(os.listdir(args.captcha_dir)):
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            greyscale_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
            processed_data = cv2.adaptiveThreshold(greyscale_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_data = np.array(processed_data, dtype=np.float32)
            processed_data = np.array(processed_data) / 255.0
            (h, w) = processed_data.shape
            input_data = processed_data.reshape([-1, h, w, 1])
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = []
            for i in range(6):
                data = interpreter.get_tensor(output_details[i]['index'])
                output_data.append(data)
            prediction = decode(captcha_symbols, output_data)
            prediction = prediction.replace(' ', '')
            writer.writerow([x, prediction])
            print(str(itr) + ': Classified ' + x + ' as ' + prediction + " ")
            
    print("Pushing " + args.output + " to GitHub")
    os.system("ssh macneill 'cd ~/Documents/Project2; git add " + args.output + "; git commit -m \"Updated Predictions\"; git push'")
    print("Finished pushing " + args.output + " to GitHub")


    
if __name__ == '__main__':
    main()
