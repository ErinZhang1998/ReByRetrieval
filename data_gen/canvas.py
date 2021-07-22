import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
from optparse import OptionParser
from PIL import Image

parser = OptionParser()
parser.add_option("--canvas_image_file_path", dest="canvas_image_file_path")
parser.add_option("--canvas_file_path", dest="canvas_file_path")

parser.add_option("--num_canvas", dest="num_canvas", type=int, default=500)

(args, argss) = parser.parse_args()

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result

def generate_canvas(num_canvas, canvas_image_file_path, canvas_file_path):
    file_paths = []
    for idx in range(num_canvas):
        start = np.random.randint(0,256,3)
        end = []
        for s in start:
            end.append(np.random.randint(255-s,256,1)[0])
        true_false = np.random.randint(0,2,3).astype(bool)
        
        array = get_gradient_3d(test_dataset.img_w, test_dataset.img_h, \
                                start, end, true_false)
        background_img = np.uint8(array)
        
        im = Image.fromarray(background_img)
        fpath = os.path.join(canvas_image_file_path, 'background_{}.jpg'.format(idx))
        im.save(fpath)
        file_paths.append(fpath)

    with open(canvas_file_path, 'w+') as the_file:
        for fpath in file_paths:
            the_file.write('{}\n'.format(fpath))

if __name__ == "__main__":
    generate_canvas(args.num_canvas, args.canvas_image_file_path, args.canvas_file_path)
