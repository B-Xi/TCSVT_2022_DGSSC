import math
import numpy as np

from dataset.dataset import load_data_HSI
from utils import shuffle

class DataHandler(object):
    def __init__(self, opts, seed):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.test_labels = None
        self.train_dataset = None
        self.test_dataset = None
        self.test_indices = None
        self.gt = None
        self.class_counts = None
        self._load_data(opts, seed)

    def _load_data(self, opts, seed):
        if opts['dataset'].lower() in ('hyrank', 'cks','paviau'):
            (self.data, self.labels), (self.test_data, self.test_labels), self.test_indices, self.gt = load_data_HSI(opts['dataset'],seed, opts['num_pcs'], opts['window_size'],
                                                                                         opts['train_ratio'], imbalance=True)
            if 'augment_x' in opts and opts['augment_x']:
                self.data, self.labels = self.oversampling(opts, self.data, self.labels, seed)
            self.num_points = len(self.data)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        self.class_counts = [np.count_nonzero(self.labels == c) for c in range(opts['n_classes'])]
        print("[ statistic ]")
        print("Total train: ", self.num_points)
        print(self.class_counts)
        print("Total test: ", len(self.test_labels))
        print([np.count_nonzero(self.test_labels == c) for c in range(opts['n_classes'])])

    def oversampling(self, opts, x, y, seed):
        n_classes = opts['n_classes']
        class_cnt = [np.count_nonzero(y == c) for c in range(n_classes)]
        max_class_cnt = max(class_cnt)
        x_aug_list = []
        y_aug_list = []
        aug_rate = opts['aug_rate']
        if aug_rate <= 0:
            return x, y
        aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
        rep_nums = [aug_num / class_cnt[i] for i, aug_num in enumerate(aug_nums)]
        for i in range(n_classes):
            idx = (y == i)
            if rep_nums[i] <= 0.:
                x_aug_list.append(x[idx])
                y_aug_list.append(y[idx])
                continue
            n_c = np.count_nonzero(idx)
            if n_c == 0:
                continue
            x_aug_list.append(
                np.repeat(x[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
            y_aug_list.append(
                np.repeat(y[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
        if len(x_aug_list) == 0:
            return x, y
        x_aug = np.concatenate(x_aug_list, axis=0)
        y_aug = np.concatenate(y_aug_list, axis=0)
        x_aug, y_aug = shuffle(x_aug, y_aug, seed)
        print([np.count_nonzero(y_aug == c) for c in range(n_classes)])
        return x_aug, y_aug
