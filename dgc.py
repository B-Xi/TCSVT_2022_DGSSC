import logging
import math
import os
import time
import save_classification_map
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio
import utils
from models import encoder, decoder, classifier

class DGC(object):

    def __init__(self, opts, tag):
        tf.reset_default_graph()
        logging.error('Building the Tensorflow Graph')
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.opts = opts

        shape = [opts['window_size'], opts['window_size'], opts['num_pcs'], 1]
        self.sample_points = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        self.labels = tf.placeholder(tf.int64, shape=[None], name='label_ph')
        self.sample_noise = tf.placeholder(
            tf.float32, [None, opts['zdim']], name='noise_ph')

        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.mean_ph = tf.placeholder(tf.float32, shape=[None, opts['zdim']])
        self.sigma_ph = tf.placeholder(tf.float32, shape=[None, opts['zdim']])

        # Build training computation graph
        sample_size = tf.shape(self.sample_points)[0]
        enc_mean, enc_sigmas = encoder(opts, inputs=self.sample_points,
                                       is_training=self.is_training, y=self.labels)
        enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
        self.enc_mean, self.enc_sigmas = enc_mean, enc_sigmas

        self.encoded = self.get_encoded(opts, self.enc_mean, self.enc_sigmas) #z
        self.encoded2 = self.get_encoded(opts, self.mean_ph, self.sigma_ph) #augmented_z

        self.reconstructed = decoder(opts, noise=self.encoded, is_training=self.is_training) #x_hat, theta_1; self.encoded=(?,64)
        self.probs1 = classifier(opts, self.encoded) #theta_2
        self.probs2 = classifier(opts, self.encoded2) #theta_2

        self.correct_sum = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.probs1, axis=1), self.labels), tf.float32))
        self.decoded = decoder(opts, noise=self.sample_noise, is_training=self.is_training) #self.sample_noise=(?,64)

        self.loss_cls = self.cls_loss(self.labels, self.probs1)
        self.loss_cls2 = self.cls_loss(self.labels, self.probs2)
        self.loss_mmd = self.mmd_penalty(self.sample_noise, self.encoded)
        self.loss_recon = self.reconstruction_loss(self.opts, self.sample_points, self.reconstructed)
        self.objective = self.loss_recon + opts['lambda'] * self.loss_mmd + self.loss_cls

        # Build evaluate computation graph
        logpxy = []
        dimY = opts['n_classes']
        N = sample_size
        S = opts['sampling_size']
        x_rep = tf.tile(self.sample_points, [S, 1, 1, 1, 1])
        for i in range(dimY):
            y = tf.fill((N,), i)
            mu, log_sig = encoder(opts, inputs=self.sample_points, is_training=False, y=y)
            mu = tf.tile(mu, [S, 1])
            log_sig = tf.tile(log_sig, [S, 1])
            y = tf.tile(y, [S, ])
            z = self.get_encoded(opts, mu, log_sig)
            z_sample = tf.random_normal((tf.shape(z)[0], opts['zdim']), 0., 1., dtype=tf.float32)

            mu_x = decoder(opts, noise=z, is_training=False)
            logit_y = classifier(opts, z)
            logp = -tf.reduce_sum((x_rep - mu_x) ** 2, axis=[1, 2, 3, 4])
            log_pyz = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit_y)
            mmd_loss = self.mmd_penalty(z_sample, z)
            bound = 0.5 * logp + log_pyz + opts['lambda'] * mmd_loss
            bound = tf.reshape(bound, [S, N])
            bound = self.logsumexp(bound) - tf.log(float(S))
            logpxy.append(tf.expand_dims(bound, 1))
        logpxy = tf.concat(logpxy, 1)
        y_pred = tf.nn.softmax(logpxy)
        self.eval_probs = y_pred

        self.loss_pretrain = self.pretrain_loss() if opts['e_pretrain'] else None
        self.add_optimizers()
        self.add_savers()
        self.tag = tag

    def get_encoded(self, opts, mu, sig):
        sample_size = tf.shape(mu)[0]
        eps = tf.random_normal((sample_size, opts['zdim']),
                               0., 1., dtype=tf.float32)
        z = mu + tf.multiply(eps, tf.sqrt(1e-8 + tf.exp(sig)))
        return z

    def log_gaussian_prob(self, x, mu=0.0, log_sig=0.0):
        logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                  - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
        ind = list(range(1, len(x.get_shape().as_list())))
        return tf.reduce_sum(logprob, ind)

    def logsumexp(self, x):
        x_max = tf.reduce_max(x, 0)
        x_ = x - x_max
        tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
        return tmp + x_max

    def pretrain_loss(self):
        opts = self.opts
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= opts['e_pretrain_sample_size'] - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= opts['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def add_savers(self):
        saver = tf.train.Saver(max_to_keep=11)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
        tf.add_to_collection('encoder', self.encoded)
        tf.add_to_collection('decoder', self.decoded)

        self.saver = saver

    def cls_loss(self, labels, logits):
        return tf.reduce_mean(tf.reduce_sum(  # FIXME
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)))

    def mmd_penalty(self, sample_pz, sample_qz):
        opts = self.opts
        sigma2_p = 1.
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        Cbase = 2. * opts['zdim'] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat

    def reconstruction_loss(self, opts, real, reconstr):
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.5 * tf.reduce_mean(loss)
        elif opts['cost'] == 'pdis':
            loss = utils.pdis(real,reconstr)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        return tf.train.AdamOptimizer(lr)

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        ae_vars = encoder_vars + decoder_vars + classifier_vars

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.objective,
                                   var_list=ae_vars)
        self.cls_opt = opt.minimize(loss=self.loss_cls,
                                    var_list=classifier_vars)
        self.cls_opt2 = opt.minimize(loss=self.loss_cls2,
                                     var_list=classifier_vars)
        # Encoder optimizer
        if opts['e_pretrain']:
            opt = self.optimizer(lr)
            self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
                                             var_list=encoder_vars)
        else:
            self.pretrain_opt = None
        if opts['LVO']:
            self.lvo_opt = opt.minimize(loss=self.objective, var_list=encoder_vars)

    def sample_pz(self, num=100, z_dist=None, labels=None):
        opts = self.opts
        if z_dist is None:
            mean = np.zeros(opts["zdim"])
            cov = np.identity(opts["zdim"])
            noise = np.random.multivariate_normal(mean, cov, num).astype(np.float32)
            return noise
        assert labels is not None
        means, covariances = z_dist
        noise = np.array([np.random.multivariate_normal(means[e], covariances[e]) for e in labels])
        return noise

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 200
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            train_size = data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids]
            batch_labels = data.labels[data_ids]
            batch_noise = self.sample_pz(batch_size)

            [_, loss_pretrain] = self.sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch_images,
                           self.labels: batch_labels,
                           self.sample_noise: batch_noise,
                           self.is_training: True})

            if opts['verbose']:
                logging.error('Step %d/%d, loss=%f' % (
                    step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break

    def augment_batch(self, x, y, class_cnt):
        max_class_cnt = max(class_cnt)
        n_classes = len(class_cnt)
        x_aug_list = [[] for _ in range(len(x))]
        y_aug_list = []
        aug_rate = self.opts['aug_rate']
        if aug_rate <= 0:
            return x, y
        aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
        rep_nums = [aug_num / class_cnt[i] for i, aug_num in enumerate(aug_nums)]
        for i in range(n_classes):
            idx = (y == i)
            if rep_nums[i] <= 0.:
                for j, x_aug in enumerate(x_aug_list):
                    x_aug.append(x[j][idx])
                y_aug_list.append(y[idx])
                continue
            n_c = np.count_nonzero(idx)
            if n_c == 0:
                continue
            for j, x_aug in enumerate(x_aug_list):
                x_aug.append(
                    np.repeat(x[j][idx],
                              repeats=math.ceil(1 + rep_nums[i]),
                              axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
            y_aug_list.append(
                np.repeat(y[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
        if len(x_aug_list[0]) == 0:
            return x, y
        aug = [np.concatenate(x_aug, axis=0) for x_aug in x_aug_list]
        y_aug = np.concatenate(y_aug_list, axis=0)
        return aug, y_aug

    def get_lr_decay(self, opts, epoch):
        decay = 1.
        if opts['lr_schedule'] == "manual":
            if epoch == 30:
                decay = decay / 2.
            if epoch == 50:
                decay = decay / 5.
            if epoch == 100:
                decay = decay / 10.
        elif opts['lr_schedule'] == "manual_smooth":
            enum = opts['epoch_num']
            decay_t = np.exp(np.log(100.) / enum)
            decay = decay / decay_t
        return decay

    def train(self, data):
        opts = self.opts
        class_cnt = [np.count_nonzero(data.labels == n) for n in range(opts['n_classes'])]
        if opts['verbose']:
            logging.error(opts)
        losses = []
        losses_rec = []
        losses_match = []
        losses_cls = []

        batches_num = math.ceil(data.num_points / opts['batch_size'])
        self.sess.run(tf.global_variables_initializer())

        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(data)
            logging.error('Pretraining the encoder done.')

        counter = 0
        train_start_time = time.time()
        for epoch in range(opts["epoch_num"]):
            # Update learning rate if necessary
            start_time = time.time()
            decay = self.get_lr_decay(opts, epoch)

            # Save the model
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'], 'checkpoints', 'trained'),
                                global_step=counter)

            acc_total = 0.
            loss_total = 0.

            for it in tqdm(range(batches_num)):
                start_idx = it * opts['batch_size']
                end_idx = start_idx + opts['batch_size']
                batch_images = data.data[start_idx:end_idx]
                batch_labels = data.labels[start_idx:end_idx]
                train_size = len(batch_labels)
                batch_noise = self.sample_pz(len(batch_images))

                feed_dict = {
                    self.sample_points: batch_images,
                    self.sample_noise: batch_noise,
                    self.labels: batch_labels,
                    self.lr_decay: decay,
                    self.is_training: True}

                if opts['LVO'] is True:
                    _ = self.sess.run(self.lvo_opt, feed_dict=feed_dict)

                (_, mu, sigma, loss, loss_rec, loss_cls, loss_match, correct) = self.sess.run(
                    [self.ae_opt,
                     self.enc_mean, self.enc_sigmas,
                     self.objective, self.loss_recon, self.loss_cls, self.loss_mmd, self.correct_sum],
                    feed_dict=feed_dict)

                if opts['augment_z'] is True:
                    [mu_aug, sigma_aug], y_aug = self.augment_batch([mu, sigma], batch_labels, class_cnt)
                    (_, loss_cls2) = self.sess.run([self.cls_opt2, self.loss_cls2],
                                                   feed_dict={self.mean_ph: mu_aug,
                                                              self.sigma_ph: sigma_aug,
                                                              self.labels: y_aug,
                                                              self.lr_decay: decay,
                                                              self.is_training: True
                                                              })
                    loss_cls += loss_cls2
                    loss += loss_cls2

                acc_total += correct / train_size
                loss_total += loss

                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                losses_cls.append(loss_cls)

                counter += 1

            # Print debug info
            now = time.time()
            debug_str = 'EPOCH: %d/%d, BATCH/SEC:%.2f' \
                        % (epoch + 1, opts['epoch_num'], float(counter) / (now - start_time))
            debug_str += ' (TOTAL_LOSS=%.5f, RECON_LOSS=%.5f, MATCH_LOSS=%.5f, CLS_LOSS=%.5f' % (
                losses[-1], losses_rec[-1], losses_match[-1], losses_cls[-1])
            logging.error(debug_str)

            training_acc = acc_total / batches_num
            avg_loss = loss_total / batches_num
            print("Train loss: %.5f, Train acc: %.5f, Time: %.5f" % (avg_loss, training_acc, time.time() - start_time))

            if (self.opts['eval_strategy'] == 1 and (epoch + 1) % 200 == 0) or \
                    self.opts['eval_strategy'] == 2 and ((0 < epoch <= 20) or (epoch > 20 and epoch % 3 == 0)):
                train_time=time.time()-train_start_time
                test_start_time=time.time()
                self.evaluate(data, epoch)
                test_time = time.time() - test_start_time
                print('Training_time = ',train_time)
                print('Testing_time = ',test_time)

            if epoch > 0 and epoch % 10 == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'],
                                             'checkpoints',
                                             'trained-final'),
                                global_step=epoch)

    def evaluate(self, data, epoch):
        batch_size = self.opts['batch_size']
        batches_num = math.ceil(len(data.test_data) / batch_size)
        probs = []
        for it in tqdm(range(batches_num)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            prob = self.sess.run(
                self.eval_probs,
                feed_dict={self.sample_points: data.test_data[start_idx:end_idx],
                           self.is_training: False})
            probs.append(prob)
        probs = np.concatenate(probs, axis=0)
        predicts = np.argmax(probs, axis=-1)
        asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi = utils.get_test_metrics(data.test_labels, predicts)
        print("EPOCH: %d, ASCA=%.5f, PRE=%.5f, REC=%.5f, SPE=%.5f, F1_ma=%.5f, F1_mi=%.5f, G_ma=%.5f, G_mi=%.5f" % (
            epoch, asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi))
        each_acc, average_acc, overall_acc, kappa, precision, average_precision = utils.AA_andEachClassAccuracy(predicts,data.test_labels)
        for i in each_acc:
            print(i)
        print(average_acc)
        print(overall_acc)
        print(kappa)
        print()
        # print('ua =')
        for i in precision:
            print(i)
        print(average_precision)

        metrics = np.hstack((each_acc, average_acc.reshape(1, ), overall_acc.reshape(1, ), kappa.reshape(1, ), precision,
                            average_precision.reshape(1, )))
        np.savetxt('records/hyrank_result'+'_'+repr(int(average_acc * 10000))+'_'+repr(int(overall_acc * 10000))+'.txt', metrics.astype(str), fmt='%s', delimiter="\t", newline='\n')

        gt_flatten = data.gt.reshape(np.prod(data.gt.shape[:2]),)
        gt_NEW = gt_flatten.copy()
        gt_NEW[data.test_indices] = predicts + 1
        classification_map = np.reshape(gt_NEW, (data.gt.shape[0], data.gt.shape[1]))
        sio.savemat('figure/hyrank/DGSSC_hyrank'+'_'+repr(int(average_acc * 10000))+ '_' + repr(int(overall_acc * 10000)) + '.mat',
                    {'classification_map': classification_map})
        hsi_pic = np.zeros((classification_map.shape[0], classification_map.shape[1], 3))
        for i in range(classification_map.shape[0]):
            for j in range(classification_map.shape[1]):
                if classification_map[i][j] == 0:
                    hsi_pic[i, j, :] = [0, 0, 0]
                if classification_map[i][j] == 1:
                    hsi_pic[i, j, :] = [0, 205, 0]
                if classification_map[i][j] == 2:
                    hsi_pic[i, j, :] = [127, 255, 0]
                if classification_map[i][j] == 3:
                    hsi_pic[i, j, :] = [46, 139, 87]
                if classification_map[i][j] == 4:
                    hsi_pic[i, j, :] = [0, 139, 0]
                if classification_map[i][j] == 5:
                    hsi_pic[i, j, :] = [160, 82, 45]
                if classification_map[i][j] == 6:
                    hsi_pic[i, j, :] = [0, 255, 255]
                if classification_map[i][j] == 7:
                    hsi_pic[i, j, :] = [255, 255, 255]
                if classification_map[i][j] == 8:
                    hsi_pic[i, j, :] = [216, 191, 216]
                if classification_map[i][j] == 9:
                    hsi_pic[i, j, :] = [255, 0, 0]
                if classification_map[i][j] == 10:
                    hsi_pic[i, j, :] = [139, 0, 0]
                if classification_map[i][j] == 11:
                    hsi_pic[i, j, :] = [100, 0, 255]
                if classification_map[i][j] == 12:
                    hsi_pic[i, j, :] = [255, 255, 0]
                if classification_map[i][j] == 13:
                    hsi_pic[i, j, :] = [238, 154, 0]
                if classification_map[i][j] == 14:
                    hsi_pic[i, j, :] = [85, 26, 139]
        save_classification_map.save_classification_map(hsi_pic / 255, classification_map, 24,
                                                        'figure/hyrank/DGSSC_hyrank'+'_'+repr(int(average_acc * 10000)) + '_' + repr(int(overall_acc* 10000)) + '.png')

