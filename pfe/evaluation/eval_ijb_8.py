"""Test PFE on IJB-A.
"""
# MIT License
#
# Copyright (c) 2019 Yichun Shi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import time
import math
import argparse
import numpy as np

from utils import utils
from utils.dataset import Dataset
from utils.imageprocessing import preprocess
from network import Network

from retinaface import RetinaFace
from sklearn.cluster import DBSCAN

import align.crop_ijba as crop
import align.align_dataset as align

from evaluation.ijbc import IJBBTest


def aggregate_templates(templates, features, method):
    for i,t in enumerate(templates):
        if len(t.indices) > 0:
            if method == 'mean':
                t.feature = utils.l2_normalize(np.mean(features[t.indices], axis=0))
            if method == 'PFE_fuse':
                t.mu, t.sigma_sq = utils.aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
                t.feature = t.mu
            if method == 'PFE_fuse_match':
                if not hasattr(t, 'mu'):
                    t.mu, t.sigma_sq = utils.aggregate_PFE(features[t.indices], normalize=True, concatenate=False)
                t.feature = np.concatenate([t.mu, t.sigma_sq])
        else:
            t.feature = None
        if i % 1000 == 0:
            sys.stdout.write('Fusing templates {}/{}...\t\r'.format(i, len(templates)))
    print('')


def force_compare(compare_func, verbose=False):
    def compare(t1, t2):
        score_vec = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] is None or t2[i] is None:
                score_vec[i] = -9999
            else:
                score_vec[i] = compare_func(t1[i][None], t2[i][None])
            if verbose and i % 1000 == 0:
                sys.stdout.write('Matching pair {}/{}...\t\r'.format(i, len(t1)))
        if verbose:
            print('')
        return score_vec
    return compare

def detect_faces(image_paths):
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')


def main(args):

    network = Network()
    network.load_model(args.model_dir)
    proc_func = lambda x: preprocess(x, network.config, False)

    testset = Dataset(args.dataset_path)
    tester = IJBBTest(testset['abspath'].values)
    tester.init_proto(args.protocol_path)

    templates = detect_faces(tester.image_paths)
    crop_aligned_templates = crop_align_templates(templates)

    mu, sigma_sq = network.extract_feature(crop_aligned_templates, args.batch_size, proc_func=proc_func, verbose=True)
    features = np.concatenate([mu, sigma_sq], axis=1)


    clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])

    print('---- Average pooling')
    aggregate_templates(tester.verification_templates, features, 'mean')
    TARs, std, FARs = tester.test_verification(force_compare(utils.pair_euc_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))

    print('---- Uncertainty pooling')
    aggregate_templates(tester.verification_templates, features, 'PFE_fuse')
    TARs, std, FARs = tester.test_verification(force_compare(utils.pair_euc_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))


    print('---- MLS comparison')
    aggregate_templates(tester.verification_templates, features, 'PFE_fuse_match')
    TARs, std, FARs = tester.test_verification(force_compare(utils.pair_MLS_score))
    for i in range(len(TARs)):
        print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str, default=None)
    parser.add_argument("--protocol", help="The dataset to test",
                        type=str, default='ijba')
    parser.add_argument("--dataset_path", help="The path to the IJB-A dataset directory",
                        type=str, default='data/ijba_mtcnncaffe_aligned')
    parser.add_argument("--protocol_path", help="The path to the IJB-A protocol directory",
                        type=str, default='proto/IJB-A')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=256)
    parser.add_argument("-j", "--jobs", type=int, default=-1,
                    help="# of parallel jobs to run (-1 will use all CPUs)")
    args = parser.parse_args()
    main(args)
