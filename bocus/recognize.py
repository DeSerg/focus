import os
import sys
import time
import math
import argparse
import numpy as np
import random
import hdbscan
import cv2
from sklearn import svm
import pickle
from matplotlib import pyplot as plt

from .network import Network
from .utils import utils
from .utils.dataset import Dataset
from .utils.imageprocessing import preprocess

from .retinaface import RetinaFace

import bocus.align.crop_ijba as crop
import bocus.align.align_dataset as align
import bocus.extern as ext


IMAGE_SIZE = [96, 112]

MAX_HEIGHT=2500
MAX_WIDTH=2500

BATCH_SIZE = 512

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
PFE_PATH = '/home/spopov/focus/pfe'

detector = RetinaFace(os.path.join(PFE_PATH, 'model/R50'), 0, 0, 'net3')

network = Network()
network.load_model(os.path.join(PFE_PATH, 'log/sphere64_casia_am_PFE/20200515-123838'))


class Template:
    def __init__(self, template_id, filename, img, box):
        self.template_id = template_id
        self.filename = filename
        self.img = img
        self.box = box

    def draw(self):
        if self.img is None:
            return

        imgg = self.img / np.max(self.img) # normalize the data to 0 - 1
        imgg = 255 * imgg # Now scale by 255
        imgg = imgg.astype(np.uint8)
        align_rgb = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(align_rgb)
        plt.show()

    def __str__(self):
        return 'template_id: {}, filename: {},\nimg: {},\nbox: {},\nlandmarks: {}'.format(\
            self.template_id, self.filename, self.img, self.box, self.landmarks)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def force_compare(compare_func, verbose=False):
    def compare(t1, t2):
        score_vec = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] is None or t2[i] is None:
                score_vec[i] = 0
            else:
                score_vec[i] = compare_func(t1[i][None], t2[i][None])
            if verbose and i % 1000 == 0:
                sys.stdout.write('Matching pair {}/{}...\t\r'.format(i, len(t1)))
        if verbose:
            print('')
        return score_vec
    return compare

def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)

    r = random.randint(0, 10)
    if r % 7 == 0:
#         print('pair_MLS_score: ', type(-dist[0]), -dist[0])
        pass

    return -dist[0]


fc = force_compare(pair_MLS_score)

def compare(x, y):
     return fc([x], [y])[0]

def max_similarity(features, verbose=False):
    max_sim = 0
    for i in range(len(features)):
        if i % 200 == 0:
            print('{} of {} processed ({})'.format(i, len(features), i/len(features)*100))
        for j in range(len(features)):
            sim = compare(features[i], features[j])
            if sim > max_sim:
                max_sim = sim
    return max_sim

def distance(x, y, max_sim):
    return 1 - compare(x, y) / max_sim

def confidence_sigma(sigma_sq):
    return -np.sum(np.log(sigma_sq), axis=0)

def confidence(x):
    x = np.array(x)
    D = int(x.shape[0] / 2)
    sigma_sq = x[D:]
    return confidence_sigma(sigma_sq)

def build_templates(filepaths):

    templates = []
    template_id = 0
    for filepath_i, filepath in enumerate(filepaths, 1):
        print('Loading img ', filepath)
        img = cv2.imread(filepath)
        height, width = img.shape[:2]

        if height > MAX_HEIGHT:
            img = image_resize(img, height=MAX_HEIGHT)
        height, width = img.shape[:2]
        if width > MAX_WIDTH:
            img = image_resize(img, width=MAX_WIDTH)

        height, width = img.shape[:2]
        print('image size: {}x{}'.format(width, height))

        if filepath_i % 200 == 0:
            print('\r{} of {} images scanned for faces'.format(filepath_i, len(filepaths)))
        if img is None or img.ndim == 0:
            print('Invalid image: %s' % impath)
            continue

        faces, landmarks = detector.detect(img)

        if faces is None:
            continue

        for face_i in range(faces.shape[0]):
            box = faces[face_i].astype(np.int)

            bbox = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])

            face_landmarks = landmarks[face_i]
            src_pts = [[face_landmarks[j][0], face_landmarks[j][1]] for j in range(5)]
            align_img, _, _ = align.align(img, src_pts, align.ref_pts, IMAGE_SIZE)

            imgg = align_img / np.max(align_img) # normalize the data to 0 - 1
            imgg = 255 * imgg # Now scale by 255
            imgg = imgg.astype(np.uint8)

            template = Template(template_id, filepath, imgg, bbox)
            template_id += 1
            templates.append(template)

    return np.array(templates)



class ModelInfo:
    def __init__(self, svm_clf, labels, image_paths):
        self.svm_clf = svm_clf
        self.labels = labels
        self.image_paths = image_paths

class RecognizeFace:

    def __init__(self, album_path):
        self.album_path = album_path
        self.pkl_filepath = os.path.join(self.album_path, ext.PKL_FILENAME)

    def perform_clustering(self):

        print('Building templates...')
        image_paths = []
        for name in os.listdir(self.album_path):
            filepath = os.path.join(self.album_path, name)
            if os.path.isfile(filepath) and filepath != self.pkl_filepath:
                image_paths.append(filepath)

        templates = build_templates(image_paths)

        proc_func = lambda templates: preprocess(np.array([template.img for template in templates]), network.config)

        mu, sigma_sq = network.extract_feature(templates, BATCH_SIZE, proc_func=proc_func, verbose=True)

        print('Concatenating features')
        features = np.concatenate([mu, sigma_sq], axis=1)

        print(features.shape)
        print(features)

        print('Measuring max similarity')
        MAX_SIMILARITY = max_similarity(features)
        print('MAX_SIMILARITY: ', MAX_SIMILARITY)

        def distance_func(x, y):
            d = distance(x, y, MAX_SIMILARITY)
            return d

        print('Performing clustering...')
        clt = hdbscan.HDBSCAN(core_dist_n_jobs=-1,\
                min_cluster_size=2, metric=distance_func)
        clt.fit(features)

        labels = np.array(clt.labels_)
        max_label = np.max(labels)
        print('Max label: ', max_label)
        no_cl_idx = np.where(labels == -1)[0]

        cur_label = max_label + 1
        for idx in no_cl_idx:
            labels[idx] = cur_label
            cur_label += 1
        print('Labels: ', labels)

        print('Training SVM...')
        svm_clf = svm.SVC()
        svm_clf.fit(features, labels)

        print('Storing model to file...')
        mi = ModelInfo(svm_clf, labels, [t.filename for t in templates])
        with open(self.pkl_filepath, 'wb') as pkf:
            pickle.dump(mi, pkf)

        print('Clustering finished!')

    def perform_recognition(self, face_path):
        print('Building template...')
        face_templates = build_templates([face_path])

        proc_func = lambda templates: preprocess(np.array([template.img for template in templates]), network.config)
        mu, sigma_sq = network.extract_feature(face_templates, BATCH_SIZE, proc_func=proc_func, verbose=True)
        face_features = np.concatenate([mu, sigma_sq], axis=1)

        print('Loading model...')
        with open(self.pkl_filepath, 'rb') as pkf:
            mi = pickle.load(pkf)

        print('Predicting...')
        face_label = mi.svm_clf.predict(face_features)[0]
        # face_labels, _ = hdbscan.approximate_predict(clt, face_features)
        # face_label = face_labels[0]
        print('Image label: {}'.format(face_label))

        inds = np.where(mi.labels == face_label)[0]
        print('template inds: ', inds)

        paths = np.unique(np.array(mi.image_paths)[inds])
        print('image paths: ', paths)

        return paths
