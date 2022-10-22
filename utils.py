import os
from operator import truediv
import numpy as np
import tensorflow as tf
from sklearn import metrics

def shuffle(x, y, seed):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y


def create_dir(d):
    if not tf.gfile.IsDirectory(d):
        tf.gfile.MakeDirs(d)


class File(tf.gfile.GFile):
    def __init__(self, *args):
        super(File, self).__init__(*args)

    def seek(self, position, whence=0):
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)


def o_gfile(filename, mode):
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)


def get_batch_size(inputs):
    return tf.cast(tf.shape(inputs)[0], tf.float32)


def get_test_metrics(y_true, y_pred, verbose=True):
    """
    :return: asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    cs_accuracy = TP / cnf_matrix.sum(axis=1)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (FP + TN)

    f1_macro = (2 * precision * recall / (precision + recall)).mean()
    f1_micro = 2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum())

    g_marco = ((recall * specificity) ** 0.5).mean()
    g_micro = ((TP.sum() / (TP.sum() + FN.sum())) * (TN.sum() / (TN.sum() + FP.sum()))) ** 0.5
    return cs_accuracy.mean(), precision.mean(), recall.mean(), specificity.mean(), f1_macro, f1_micro, g_marco, g_micro

def pdis(A, B):
    A = tf.squeeze(A, axis=4)
    B = tf.squeeze(B, axis=4)
    A = tf.transpose(A, perm=[0,3,1,2])
    B = tf.transpose(B, perm=[0,3,1,2])
    AA = tf.tile(tf.expand_dims(A, -1), [1, 1, 1, 1, A.shape[2] * A.shape[3]])
    BB = tf.tile(tf.expand_dims(B, -1), [1, 1, 1, 1, B.shape[2] * B.shape[3]])
    AB = tf.tile(tf.reshape(A, [-1, A.shape[1], 1, 1, A.shape[2] * A.shape[3]]), [1, 1, A.shape[2], A.shape[3], 1])
    BA = tf.tile(tf.reshape(B, [-1, B.shape[1], 1, 1, B.shape[2] * B.shape[3]]), [1, 1, B.shape[2], B.shape[3], 1])
    tf.nn.top_k
    da = tf.reduce_min(tf.reduce_min(tf.reduce_mean(tf.square(AA - BA), axis=1), axis=1), axis=1)
    db = tf.reduce_min(tf.reduce_min(tf.reduce_mean(tf.square(BB - AB), axis=1), axis=1), axis=1)
    dist = tf.reduce_mean(tf.reduce_max(tf.stack([tf.reshape(da, [-1]), tf.reshape(db, [-1])], axis=0), axis=0))
    return dist

def AA_andEachClassAccuracy(predicts,test_labels):
    confusion_matrix = metrics.confusion_matrix(predicts,test_labels)
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=0)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    precision = np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=1)
    average_precision = np.sum(precision)/confusion_matrix.shape[0]
    kappa = metrics.cohen_kappa_score(predicts,test_labels)
    overall_acc = metrics.accuracy_score(predicts,test_labels)
    return each_acc, average_acc, overall_acc, kappa, precision,average_precision