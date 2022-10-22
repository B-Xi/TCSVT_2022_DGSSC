import numpy as np
import torchvision
import os
import scipy.io as sio
from torch.utils.data import DataLoader
# from torchvision import transforms
from functions import *
from sklearn import preprocessing
# from .CelebA import CelebA
import h5py

_dataset_ratio_mapping = {
    'paviau': [66, 186, 20, 30, 13, 50, 13, 36, 9],
    'chikusei': [28, 28, 2, 48, 42, 11, 205, 65, 133, 12, 59, 21, 12, 76, 4, 2, 10, 8, 1],
    'hyrank': [14, 3, 27, 4, 70, 11, 25, 54, 190, 140, 20, 24, 70, 23]
}


def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, len(dataset))
    x, y = next(iter(loader))
    return x.numpy(), y.numpy()


def _shuffle(x, y, seed):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y

def load_data_HSI(name, seed, numComponents, window_size, train_ratio, imbalance=None, data_dir=None):
    name = name.lower()
    if data_dir is None:
        data_dir = './data/%s/' % name
    # load the data set
    data_IN = sio.loadmat(data_dir+name+'.mat')[name].astype('float32')
    gt_IN = sio.loadmat(data_dir+name+'_gt.mat')[name+'_gt']
    rows, cols, depth = data_IN.shape
    classes_num = np.max(gt_IN)
    # split the training and test locations
    trainingMap = np.zeros_like(gt_IN)
    testMap = np.zeros_like(gt_IN)
    train_num_perclass=np.zeros([classes_num,1])
    train_num = 0
    test_num = 0
    for i in range(1, classes_num + 1):
        index = np.where(gt_IN == i)
        n_sample = len(index[0])
        array = np.random.permutation(n_sample)
        n_per = int(n_sample*train_ratio)
        if i == 1:
            array1_train = index[0][array[:n_per]]
            array2_train = index[1][array[:n_per]]
            array1_test = index[0][array[n_per:]]
            array2_test = index[1][array[n_per:]]
        else:
            array1_train = np.concatenate((array1_train, index[0][array[:n_per]]))
            array2_train = np.concatenate((array2_train, index[1][array[:n_per]]))
            array1_test = np.concatenate((array1_test, index[0][array[n_per:]]))
            array2_test = np.concatenate((array2_test, index[1][array[n_per:]]))
        train_num_perclass[i-1,0]=int(len(array1_train))
        trainingMap[index[0][array[:n_per]], index[1][array[:n_per]]] = i
        testMap[index[0][array[n_per:]], index[1][array[n_per:]]] = i
        train_num += n_per
        test_num += n_sample - n_per
    train_indices_rows,train_indices_cols = np.where(trainingMap!=0)
    trainingMap_flatten = trainingMap.reshape(np.prod(trainingMap.shape[:2]),)
    train_indices = np.where(trainingMap_flatten!=0)
    y_train = trainingMap_flatten[train_indices]-1

    test_indices_rows,test_indices_cols = np.where(testMap!=0)
    testMap_flatten = testMap.reshape(np.prod(testMap.shape[:2]),)
    test_indices = np.where(testMap_flatten!=0)
    y_test = testMap_flatten[test_indices]-1
    #assign the training and testing samples
    data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
    data = preprocessing.scale(data)

    whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
    whole_data, pca = applyPCA(whole_data, numComponents = numComponents)
    whole_data = NormalizationEachBand(whole_data, unit=False)

    PATCH_LENGTH = int((window_size-1)/2)
    padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

    train_data = np.zeros((train_indices_rows.shape[0], window_size, window_size, numComponents))
    test_data = np.zeros((test_indices_rows.shape[0], window_size, window_size, numComponents))
    train_assign = indexToAssignment(train_indices_rows, train_indices_cols, PATCH_LENGTH)
    test_assign = indexToAssignment(test_indices_rows,test_indices_cols, PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data,train_assign[i][0],train_assign[i][1],PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data,test_assign[i][0],test_assign[i][1],PATCH_LENGTH)
    X_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], numComponents)
    X_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], numComponents)

    X_train, y_train = _shuffle(X_train, y_train, seed)#X_test=test_data.mat
    # X_train = np.transpose(X_train, axes=[0, 2, 3, 1])#y_test=test_label.mat
    # X_test = np.transpose(X_test, axes=[0, 2, 3, 1])#X_train, y_train = _shuffle(X_train, y_train, seed)
    n_classes = len(np.unique(y_test))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print([np.count_nonzero(y_train == i) for i in range(n_classes)])
    if imbalance is None or imbalance is False:
        return (X_train, y_train), (X_test, y_test)
    if imbalance is True:
        ratio = _dataset_ratio_mapping[name]
        # ratio = train_num_perclass.tolist()
    else:
        ratio = imbalance
    X_train = [X_train[y_train == i][:num] for i, num in enumerate(ratio)]
    y_train = [y_train[y_train == i][:num] for i, num in enumerate(ratio)]
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_train, y_train = _shuffle(X_train, y_train, seed)
    # X_train =X_train.transpose([0,3,1,2])
    # X_test = X_test.transpose([0,3,1,2])
    X_train = np.expand_dims(X_train,4)
    X_test = np.expand_dims(X_test, 4)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test), test_indices, gt_IN

