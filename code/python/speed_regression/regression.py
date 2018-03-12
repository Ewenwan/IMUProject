import time
import argparse
import warnings
import os
import sys
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas
import cv2

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')
import speed_regression.training_data as td

args = None


def load_datalist(path, option):
    root_dir = os.path.dirname(path)
    with open(path) as f:
        dataset_list = [s.strip() for s in f.readlines() if s[0] != '#']
    feature_all = []
    target_all = []
    for dataset in dataset_list:
        info = dataset.split(',')
        data_path = root_dir + '/' + info[0] + '/processed/data.csv'
        if not os.path.exists(data_path):
            warnings.warn('File ' + data_path + ' does not exist, omit the folder')
            continue

        print('Loading dataset ' + data_path)
        data_all = pandas.read_csv(data_path)

        extra_args = {'target_smooth_sigma': 30.0,
                      'feature_smooth_sigma': 2.0}

        feature, target = td.get_training_data(data_all=data_all, option=option, extra_args=extra_args)
        feature_all.append(feature)
        target_all.append(target)

    return np.concatenate(feature_all, axis=0), np.concatenate(target_all, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--validation', default=None, type=str)
    parser.add_argument('--window', default=200, type=int)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--feature', default='direct_gravity', type=str)
    parser.add_argument('--target', default='local_speed_gravity', type=str)
    parser.add_argument('--split_ratio', default=0.3, type=float)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--e', default=None, type=float)
    parser.add_argument('--g', default=None, type=float)
    parser.add_argument('--grid', action='store_true')

    args = parser.parse_args()

    options = td.TrainingDataOption(sample_step=args.step, window_size=args.window,
                                    feature=args.feature, target=args.target)

    print("-----------Training set--------------")
    training_feature_all, training_target_all = load_datalist(args.list, options)
    n_feature = training_feature_all.shape[1]
    if training_target_all.ndim == 1:
        training_target_all = training_target_all[:, None]
    print("{} training samples in total. feature dimension: {}".format(training_target_all.shape[0], n_feature))

    validation_feature_all, validation_target_all = None, None
    if args.validation is not None:
        print("-----------Validation set--------------")
        validation_feature_all, validation_target_all = load_datalist(args.validation, options)
        assert validation_feature_all.shape[1] == n_feature
        if validation_target_all.ndim == 1:
            validation_target_all = validation_target_all[:, None]
        print("{} validation samples in total. feature dimension: {}".format(validation_target_all.shape[0], n_feature))

    assert len(training_feature_all) > 0, 'No data was loaded'

    best_C = [5.0, 5.0, 5.0]
    best_e = [0.01, 0.01, 0.01]

    if args.C is not None:
        best_C = [args.C, args.C, args.C]
    if args.e is not None:
        best_e = [args.e, args.e, args.e]

    for chn in [0, 2]:
        print('Training SVM for target ', chn)

        regressor = None
        regressor_cv = None

        if args.grid:
            print('Running grid search')
            
            search_dict = {'C': [0.01, 0.1, 1.0, 10.0, 50.0],
                           'epsilon': [0.001, 0.01, 0.1, 1.0],
                           'kernel': ['rbf']}

            grid_searcher = GridSearchCV(svm.SVR(), search_dict, n_jobs=6, verbose=3, scoring='neg_mean_squared_error')

            start_t = time.clock()
            grid_searcher.fit(training_feature_all, training_target_all[:, chn])

            time_passage = time.clock() - start_t
            print('All done, time usage: {:.3f}s ({:.3f}h)'.format(time_passage, time_passage / 3600.0))
            print('Optimal parameter: ', grid_searcher.best_params_)
            print('Best score: ', grid_searcher.best_score_)
            regressor = grid_searcher.best_estimator_

        else:
            print('Training with OpenCV, C: {}, epsilon: {}'.format(best_C[chn], best_e[chn]))
            regressor_cv = cv2.ml.SVM_create()
            regressor_cv.setType(cv2.ml.SVM_EPS_SVR)
            regressor_cv.setKernel(cv2.ml.SVM_RBF)
            regressor_cv.setC(best_C[chn])
            regressor_cv.setP(best_e[chn])
            regressor_cv.setGamma(1.0 / training_feature_all.shape[1])
            regressor_cv.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-09))
            training_feature_cv = training_feature_all.astype(np.float32)
            training_target_cv = training_target_all[:, chn].astype(np.float32)

            regressor_cv.train(training_feature_cv, cv2.ml.ROW_SAMPLE, training_target_cv)

            predicted_training = regressor_cv.predict(training_feature_cv)[1]
            score_r2 = r2_score(training_target_all[:, chn], predicted_training)
            score_l2 = mean_squared_error(training_target_all[:, chn], predicted_training)
            print('Training score by OpenCV: {}(r2), {}(l2), support vectors: {}'.
                  format(score_r2, score_l2, regressor_cv.getSupportVectors().shape[0]))

            if validation_feature_all is not None:
                validation_feature_cv = validation_feature_all.astype(np.float32)
                validation_target_cv = validation_target_all[:, chn].astype(np.float32)
                predicted_validation = regressor_cv.predict(validation_feature_cv)[1]
                valid_score_r2 = r2_score(validation_target_cv, predicted_validation)
                valid_score_l2 = mean_squared_error(validation_target_cv, predicted_validation)
                print('Validation score by OpenCV: {}(r2), {}(l2)'. format(valid_score_r2, valid_score_l2))

            if len(args.output) > 0:
                cv_model_path = '{}_w{}_s{}_{}.yml'.format(args.output, args.window, args.step, chn)
                print('CV model written into ', cv_model_path)
                regressor_cv.save(cv_model_path)
