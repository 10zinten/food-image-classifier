import os
from scipy.misc import imread, imresize, imsave
from random import shuffle

IMAGE_SIZE = 128
DATASET_DIR = 'Gaze_UPMC_Food20/images'
TRAIN_DIR = 'dataset/train_set/'
TEST_DIR = 'dataset/test_set/'

# save train and test image in train_set and test_set folder
def save_image(images, path, name):
    for index, img in enumerate(images):
        imsave(os.path.join(path, name + str(index) + '.jpg'), img)


def create_dataset(test_size):

    for index, img_dir in enumerate(sorted(os.listdir(DATASET_DIR))):
        datasets = []
        dir_path = os.path.join(DATASET_DIR, img_dir)
        for img in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img)
            img = imresize(imread(img_path), (IMAGE_SIZE, IMAGE_SIZE))
            datasets.append(img)

        shuffle(datasets)
        if not os.path.isdir(os.path.join(TRAIN_DIR, img_dir)):
            os.makedirs(os.path.join(TRAIN_DIR, img_dir))
        train_set = datasets[: len(datasets) - int(len(datasets) * test_size)]
        save_image(train_set, os.path.join(TRAIN_DIR, img_dir), img_dir)

        if not os.path.isdir(os.path.join(TEST_DIR, img_dir)):
            os.makedirs(os.path.join(TEST_DIR, img_dir))
        test_set = datasets[len(train_set):]
        save_image(test_set, os.path.join(TEST_DIR, img_dir), img_dir)
        del datasets[:]

if __name__ == '__main__':
    create_dataset(0.20)
