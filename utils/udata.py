
"""
This module implements methods to handle datasets.
"""

# External Libraries
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas
import torch
import os

# Standard Libraries
from os import path, listdir
import sys
import csv
import re

# Modules
import uimage


# RAF-DB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Adapted from DAN: https://github.com/yaoing/DAN/blob/main/rafdb.py

class RafDataSet(Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pandas.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,
                             names=['name', 'label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# RAF-DB <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# FER+ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class FERplus(Dataset):
    def __init__(self, idx_set=0, max_loaded_images_per_label=1000, transforms=None, base_path_to_FER_plus=None):
        """
            Code based on https://github.com/microsoft/FERPlus.

            :param idx_set: Labeled = 0, Validation = 1, Test = 2
            :param max_loaded_images_per_label: Maximum number of images per label
            :param transforms: transforms (callable, optional): Optional transform to be applied on a sample.
        """

        self.idx_set = idx_set
        self.max_loaded_images_per_label = max_loaded_images_per_label
        self.transforms = transforms
        self.base_path_to_FER_plus = base_path_to_FER_plus
        self.fer_sets = {0: 'FER2013Train/', 1: 'FER2013Valid/', 2: 'FER2013Test/'}

        # Default values
        self.num_labels = 8
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # Load data
        self.loaded_data = self._load()
        print('Size of the loaded set: {}'.format(self.loaded_data[0].shape[0]))

    def __len__(self):
        return self.loaded_data[0].shape[0]

    def __getitem__(self, idx):
        sample = {'image': self.loaded_data[0][idx], 'emotion': self.loaded_data[1][idx]}
        sample['image'] = Image.fromarray(sample['image'])

        if not (self.transforms is None):
            sample['image'] = self.transforms(sample['image'])

        return Normalize(mean=self.mean, std=self.std)(ToTensor()(sample['image'])), sample['emotion']

    def online_normalization(self, x):
        return Normalize(mean=self.mean, std=self.std)(ToTensor()(x))

    def norm_input_to_orig_input(self, x):
        x_r = torch.zeros(x.size())
        x_r[0] = (x[2] * self.std[2]) + self.mean[2]
        x_r[1] = (x[1] * self.std[1]) + self.mean[1]
        x_r[2] = (x[0] * self.std[0]) + self.mean[0]
        return x_r

    @staticmethod
    def get_class(idx):
        classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

        return classes[idx]

    @staticmethod
    def _parse_to_label(idx):
        """
        Parse labels to make them compatible with AffectNet.
        :param idx:
        :return:
        """
        emo_to_return = np.argmax(idx)

        if emo_to_return == 2:
            emo_to_return = 3
        elif emo_to_return == 3:
            emo_to_return = 2
        elif emo_to_return == 4:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 4

        return emo_to_return

    @staticmethod
    def _process_data(emotion_raw):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal)
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        # find the peak value of the emo_raw list
        maxval = max(emotion_raw)
        if maxval > 0.5 * sum_list:
            emotion[np.argmax(emotion_raw)] = maxval
        else:
            emotion = emotion_unknown  # force setting as unknown

        return [float(i) / sum(emotion) for i in emotion]

    def _load(self):
        csv_label = []
        data, labels = [], []
        counter_loaded_images_per_label = [0 for _ in range(self.num_labels)]

        path_folders_images = path.join(self.base_path_to_FER_plus, 'Images', self.fer_sets[self.idx_set])
        path_folders_labels = path.join(self.base_path_to_FER_plus, 'Labels', self.fer_sets[self.idx_set])

        with open(path_folders_labels + '/label.csv') as csvfile:
            lines = csv.reader(csvfile)
            for row in lines:
                csv_label.append(row)

        # Shuffle training set
        if self.idx_set == 0:
            np.random.shuffle(csv_label)

        for l in csv_label:
            emotion_raw = list(map(float, l[2:len(l)]))
            emotion = self._process_data(emotion_raw)
            emotion = emotion[:-2]

            try:
                emotion = [float(i) / sum(emotion) for i in emotion]
                emotion = self._parse_to_label(emotion)
            except ZeroDivisionError:
                emotion = 9

            if (emotion < self.num_labels) and (counter_loaded_images_per_label[int(emotion)] < self.max_loaded_images_per_label):
                counter_loaded_images_per_label[int(emotion)] += 1

                img = np.array(uimage.read(path.join(path_folders_images, l[0])), np.uint8)

                box = list(map(int, l[1][1:-1].split(',')))

                if box[-1] != 48:
                    print("[INFO] Face is not centralized.")
                    print(path.join(path_folders_images, l[0]))
                    print(box)
                    exit(-1)

                img = img[box[0]:box[2], box[1]:box[3], :]
                img = uimage.resize(img, (224, 224))

                data.append(img)
                labels.append(emotion)

            has_loading_finished = (np.sum(counter_loaded_images_per_label) >= (self.max_loaded_images_per_label * self.num_labels))

            if has_loading_finished:
                break

        return [np.array(data), np.array(labels)]

# FER+ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# AffectNet (Categorical) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class AffectNetCategorical(Dataset):
    def __init__(self, idx_set=0, max_loaded_images_per_label=1000, transforms=None, is_norm_by_mean_std=True,
                 base_path_to_affectnet=None):
        """
            This class follows the experimental methodology conducted by (Mollahosseini et al., 2017).

            Refs.
            Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. Affectnet: A database for facial expression,
            valence, and arousal computing in the wild. IEEE Transactions on Affective Computing.

            :param idx_set: Labeled = 0, Unlabeled = 1, Validation = 2, Test = Not published by
                            (Mollahosseini et al., 2017)
            :param max_loaded_images_per_label: Maximum number of images per label
            :param transforms: transforms (callable, optional): Optional transform to be applied on a sample.
        """

        self.idx_set = idx_set
        self.max_loaded_images_per_label = max_loaded_images_per_label
        self.transforms = transforms
        self.base_path_to_affectnet = base_path_to_affectnet
        self.affectnet_sets = {'supervised': 'Training_Labeled/',
                               'unsupervised': 'Training_Unlabeled/',
                               'validation': 'Validation/'}

        # Default values
        self.num_labels = 8
        if is_norm_by_mean_std:
            self.mean = [149.35457 / 255., 117.06477 / 255., 102.67609 / 255.]
            self.std = [69.18084 / 255., 61.907074 / 255., 60.435623 / 255.]
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]

        # Load data
        self.loaded_data = self._load()
        print('Size of the loaded set: {}'.format(self.loaded_data[0].shape[0]))

    def __len__(self):
        return self.loaded_data[0].shape[0]

    def __getitem__(self, idx):
        sample = {'image': self.loaded_data[0][idx], 'emotion': self.loaded_data[1][idx]}
        sample['image'] = Image.fromarray(sample['image'])

        if not (self.transforms is None):
            sample['image'] = self.transforms(sample['image'])

        return Normalize(mean=self.mean, std=self.std)(ToTensor()(sample['image'])), sample['emotion']

    def online_normalization(self, x):
        return Normalize(mean=self.mean, std=self.std)(ToTensor()(x))

    def norm_input_to_orig_input(self, x):
        x_r = torch.zeros(x.size())
        x_r[0] = (x[2] * self.std[2]) + self.mean[2]
        x_r[1] = (x[1] * self.std[1]) + self.mean[1]
        x_r[2] = (x[0] * self.std[0]) + self.mean[0]
        return x_r

    @staticmethod
    def get_class(idx):
        classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

        return classes[idx]

    @staticmethod
    def _parse_to_label(idx):
        """
            The file name follows this structure: 'ID_s_exp_s_val_s_aro_.jpg' Ex. '0000000s7s-653s653.jpg'.

            Documentation of labels adopted by AffectNet's authors:
            Expression: expression ID of the face (0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face)
            Valence: valence value of the expression in interval [-1,+1] (for Uncertain and No-face categories the value is -2)
            Arousal: arousal value of the expression in interval [-1,+1] (for Uncertain and No-face categories the value is -2)

            :param idx: File's name
            :return: label
        """

        label_info = idx.split('s')
        discrete_label = np.int(label_info[1])

        return discrete_label if (discrete_label < 8) else -1

    def _load(self):
        data_affect_net, labels_affect_net = [], []
        counter_loaded_images_per_label = [0 for _ in range(self.num_labels)]

        if self.idx_set == 0:
            path_folders_affect_net = path.join(self.base_path_to_affectnet,
                                                self.affectnet_sets['supervised'])
        elif self.idx_set == 1:
            path_folders_affect_net = path.join(self.base_path_to_affectnet,
                                                self.affectnet_sets['unsupervised'])
        else:
            path_folders_affect_net = path.join(self.base_path_to_affectnet,
                                                self.affectnet_sets['validation'])

        folders_affect_net = sort_numeric_directories(listdir(path_folders_affect_net))
        # Randomize folders
        if self.idx_set < 2:
            np.random.shuffle(folders_affect_net)

        for f_af in folders_affect_net:
            path_images_affect_net = path.join(path_folders_affect_net, f_af)

            images_affect_net = np.sort(np.array(listdir(path_images_affect_net)))
            # Randomize images
            if self.idx_set < 2:
                np.random.shuffle(images_affect_net)

            for file_name_image_affect_net in images_affect_net:
                lbl = self._parse_to_label(file_name_image_affect_net)

                if (lbl >= 0) and (counter_loaded_images_per_label[int(lbl)] < self.max_loaded_images_per_label):
                    img = np.array(uimage.read(path.join(path_images_affect_net, file_name_image_affect_net)), np.uint8)

                    data_affect_net.append(img)
                    labels_affect_net.append(lbl)

                    counter_loaded_images_per_label[int(lbl)] += 1

                has_loading_finished = (np.sum(counter_loaded_images_per_label) >= (
                            self.max_loaded_images_per_label * self.num_labels))

                if has_loading_finished:
                    break

            if has_loading_finished:
                break

        return [np.array(data_affect_net), np.array(labels_affect_net)]

# AffectNet (Categorical) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Other methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def sort_numeric_directories(dir_names):
    return sorted(dir_names, key=lambda x: (int(re.sub("\D", "", x)), x))


def _generate_single_file_name(img_id, expression, valence, arousal):
    valence = int(valence * 1000)
    arousal = int(arousal * 1000)
    return '%07ds%ds%ds%d.jpg' % (img_id, expression, valence, arousal)


def pre_process_affect_net(base_path_to_images, base_path_to_annotations, base_destination_path, set_index,
                           image_size):
    """
    Pre-process the AffectNet dataset. Faces are cropped and resized to 96 x 96 pixels.
    The images are organized in folders with 500 images each. The test set had not been released
    when this experiment was carried out.

    :param base_path_to_images: (string) Path to images.
    :param base_path_to_annotations: (string) Path to annotations.
    :param base_destination_path: (string) destination path to save preprocessed data.
    :param set_index: (int = {0, 1, 2}) set_index = 0 process the automatically annotated images.
                                        set_index = 1 process the manually annotated images: training set.
                                        set_index = 2 process the manually annotated images: validation set.
    :param image_size: (int = {96,224}) the size of cropped image. It is 96 for ESR and 224 for MA-Net
    :return: (void)
    """

    print('preprocessing started')
    assert ((set_index < 3) and (set_index >= 0)), "set_index must be 0, 1 or 2."

    annotation_folders = ['Automatically_Annotated_extracted/', 'Manually_Annotated_extracted/',
                          'Manually_Annotated_extracted/']
    destination_set_folders = ['Training_Unlabeled/', 'Training_Labeled/',
                               'Validation/']
    annotation_file_names = ['automatically_annotated.csv', 'training.csv', 'validation.csv']

    image_id = 0
    error_image_id = []
    img_size = (image_size, image_size)
    num_images_per_folder = 500

    annotation_file = pandas.read_csv(path.join(base_path_to_annotations, annotation_file_names[set_index]))
    print('annotation file loaded')

    for line in range(image_id, annotation_file.shape[0]):
        try:
            # Read image
            img_file_name = annotation_file.get('subDirectory_filePath')[line]
            img_file_name = img_file_name.split("/")[-1]
            img_full_path = path.join(base_path_to_images, annotation_folders[set_index], img_file_name)
            img = uimage.read(img_full_path)

            # Crop face
            x = int(annotation_file.get('face_x')[line])
            y = int(annotation_file.get('face_y')[line])
            w = int(annotation_file.get('face_width')[line])
            h = int(annotation_file.get('face_height')[line])
            img = img[x:x + w, y:y + h, :]

            # Resize image
            img = uimage.resize(img, img_size)

            # Save image
            folder = str(image_id // num_images_per_folder)
            exp = annotation_file.get('expression')[line]
            val = annotation_file.get('valence')[line]
            aro = annotation_file.get('arousal')[line]
            file_name = _generate_single_file_name(image_id, exp, val, aro)
            uimage.write(img, path.join(base_destination_path, destination_set_folders[set_index], folder), file_name)
            image_id += 1
        except Exception:
            print('ERROR: The image ID %d is corrupted.' % image_id)
            error_image_id.append(image_id)

    print('Dataset has been processed.')
    print('Images successfully processed: %d' % (image_id - len(error_image_id)))
    print('Images processed with error: %d' % len(error_image_id))
    print('Image IDs processed with error: %s' % error_image_id)


if __name__ == '__main__':
    base_path_to_images = '/mnt/archive/common/datasets/facial_datasets/AffectNet'
    base_path_to_annotations = '/mnt/archive/common/datasets/facial_datasets/AffectNet/Manually_Annotated_file_lists'
    base_destination_path = '/mnt/archive/common/datasets/facial_datasets/AffectNet/preprocessed_MANet'
    pre_process_affect_net(base_path_to_images=base_path_to_images, base_path_to_annotations=base_path_to_annotations,
                           base_destination_path=base_destination_path, set_index=1, image_size=224)

