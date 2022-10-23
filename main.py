import argparse
import logging
import os
import subprocess
from io import StringIO

import pandas as pd

import configs
import utils
from datahandler import DataHandler
from dgc import DGC

def main(tag, seed, dataset):
    opts = getattr(configs, 'config_%s' % dataset)
    opts['work_dir'] = './results/%s/' % tag

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                                  'checkpoints'))

    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    data = DataHandler(opts, seed)
    model = DGC(opts, tag)
    model.train(data)

def get_free_gpu(num=1):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf8')),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip('[MiB]')))
    gpu_df = gpu_df.sort_values(by='memory.free', ascending=False)
    print('GPU usage:\n{}'.format(gpu_df))
    free_gpus = []
    for i in range(num):
        print('Returning GPU{} with {} free MiB'.format(gpu_df.index[i], gpu_df.iloc[i]['memory.free']))
        free_gpus.append(str(gpu_df.index[i]))
    return ','.join(free_gpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='hyrank',
                        help='dataset [cks/paviau/hyrank]')
    parser.add_argument("--seed", type=int, default=1,
                        help='random seed for imbalance data generation')
    FLAGS = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["OMP_NUM_THREADS"] = "8"
    dataset_name = FLAGS.exp
    ITER = 1
    seeds = [1220, 1330, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
    for index_iter in range(ITER):
        seed = seeds[index_iter]
        tag = '%s_seed%02d' % (dataset_name, seed)
        main(tag, seed, dataset_name)
