#!/usr/bin/env python3

import os
import numpy as np
import random
import string
import cv2
import argparse
import captcha.image
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--weights', help='File with the symbol weights to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    
    if args.weights is None:
        weights = np.full(len(captcha_symbols), 1)
#         characters = "!\"#$%&()*+,/-:<=>?@[\]^|~"
#         for character in characters:
#             index = captcha_symbols.find(character)
#             weights[index] = 2
    else:
        with open(args.weights, 'r') as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            weights = np.array(next(reader)) 

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    for i in range(args.count):
        random_str = ''.join(random.choices(captcha_symbols.strip(), weights=weights, cum_weights=None, k=random.randint(1, args.length)))
        random_str = '{: <6}'.format(random_str)
        image_path = os.path.join(args.output_dir, random_str+'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')
        
        random_str = random_str.replace(' ', '')
        image = np.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)
        
    print("Finished generating captchas with symbol set {" + captcha_symbols + "}")

if __name__ == '__main__':
    main()
