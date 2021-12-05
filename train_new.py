import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import utils_new
import cit_gan
import decimal
import gan_utils
import utils
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(42)
np.random.seed(42)


def main():
    # number of samples
    sample_size = 1000
    batch_size = 64
    alpha = 0.1
    alpha1 = 0.05
    # z_dims_scheme = [20, 50, 100, 250, 500]
    z_dims_scheme = [100]
    test = 'power'
    # test = 'type1error'

    saved_file = "{}-{}{}-{}-{}".format('GCIT', datetime.now().strftime("%h"), datetime.now().strftime("%d"),
                                        datetime.now().strftime("%H"), datetime.now().strftime("%M"))
    log_dir = "./trained/{}/log".format(saved_file)

    base_path = './trained/{}/'.format(saved_file)

    train_writer = tf.summary.create_file_writer(logdir=log_dir)
    n_test = 1

    test_count = 0

    for z_dim in z_dims_scheme:
        p_values = []
        for n in range(n_test):
            start_time = datetime.now()
            p_value = utils_new.gcit_new(n=sample_size, z_dim=z_dim, simulation=test,
                                         batch_size=batch_size, n_iter=1000, train_writer=train_writer,
                                         current_iters=test_count * n_test)
            p_values.append(p_value)
            test_count += 1
            print("--- The %d'th iteration take %s seconds ---" % (test_count, (datetime.now() - start_time)))

            fp = [pval < alpha / 2.0 for pval in p_values]
            final_result = tf.reduce_sum(tf.cast(fp, tf.float32)) / len(fp)
            fp1 = [pval < alpha1 / 2.0 for pval in p_values]
            final_result1 = tf.reduce_sum(tf.cast(fp1, tf.float32)) / len(fp1)

            if test == 'type1error':
                print('Type 1 error: {} for z dimension {} with significance level {}'.format(final_result, z_dim, alpha))
                print('Type 1 error: {} for z dimension {} with significance level {}'.format(final_result1, z_dim, alpha1))
            else:
                print('Power: {} for z dimension {} with significance level {}'.format(final_result, z_dim, alpha))
                print('Power: {} for z dimension {} with significance level {}'.format(final_result1, z_dim,
                                                                                              alpha1))

            if n % 200 == 0 and n != 0:
                filename = '{}_z_dims{}_and_iterations{}.npz'.format(test, z_dim, n)
                np.savez(os.path.join(base_path, filename), np.asarray(p_values))


if __name__ == '__main__':
    main()