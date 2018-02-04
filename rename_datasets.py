''' This module rename the images in each class in a consistent manner
    final name image of the datapoint will will be classname_count.extension
'''
import os
import argparse


def rename(args):
    for dir_name in os.listdir(args.dir):
        dir_path = os.path.join(args.dir, dir_name)
        count = 1
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            extension = img_name.split('.')[-1]
            os.rename(img_path, os.path.join(dir_path, dir_name + '_' + \
                                             str(count) + '.' + extension))
            count += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='renaming data to consisten name')
    parser.add_argument('-d', '--dir', type=str, required=True,
                        help='Location of Datasets')
    args = parser.parse_args()
    rename(args)
