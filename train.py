import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import utils
import cit_gan
import decimal
tf.keras.backend.set_floatx('float64')
# tf.random.set_seed(1)


def main():
    # number of samples
    sample_size = 1000
    batch_size = 64
    alpha = 0.1
    alpha1 = 0.05
    # z_dims_scheme = [20, 50, 100, 250, 500]
    z_dims_scheme = [200]
    # lamda_scheme = [0.0, 5.0, 10.0, 15.0, 20.0]
    lamda_scheme = [0.0]
    p_vals_lamda = []

    saved_file = "{}-{}{}-{}:{}".format('GCIT', datetime.now().strftime("%h"), datetime.now().strftime("%d"),
                                        datetime.now().strftime("%H"), datetime.now().strftime("%M"))
    log_dir = "./trained/{}/log".format(saved_file)

    train_writer = tf.summary.create_file_writer(logdir=log_dir)
    n_test = 1000

    test_count = 0
    for lamda_val in lamda_scheme:
        p_vals_z = []
        p_vals_z1 = []
        for z_dim in z_dims_scheme:
            p_values = []
            for _ in range(n_test):
                start_time = datetime.now()
                p_value = utils.gcit(n=sample_size, z_dim=z_dim, simulation='type1error', statistic="rdc",
                                     batch_size=batch_size, lamda=lamda_val, n_iter=1000, train_writer=train_writer)
                p_values.append(p_value)
                test_count += 1
                print("--- The %d'th iteration take %s seconds ---" % (test_count, (datetime.now() - start_time)))

                fp = [pval < alpha / 2.0 for pval in p_values]
                final_result = tf.reduce_sum(tf.cast(fp, tf.float32)) / len(fp)
                print('Type 1 error: {} for lambda {} and z dimension {} with '
                      'significance level {}'.format(final_result, lamda_val, z_dim, alpha))
                fp1 = [pval < alpha1 / 2.0 for pval in p_values]
                final_result1 = tf.reduce_sum(tf.cast(fp1, tf.float32)) / len(fp1)
                print('Type 1 error: {} for lambda {} and z dimension {} with '
                      'significance level {}'.format(final_result1, lamda_val, z_dim, alpha1))

            # print(np.asarray(p_values))
            base_path = './trained/{}/'.format(saved_file)
            filename = 'pvals_lam{}_z_dims{}.npz'.format(lamda_val, z_dim)
            np.savez(os.path.join(base_path, filename), np.asarray(p_values))
            p_vals_z.append(final_result)
            p_vals_z1.append(final_result1)
        p_vals_lamda.append(np.asarray(p_vals_z))
    print(p_vals_lamda)
    utils.plot_results(z_dims_scheme, p_vals_lamda, lamda_scheme)


if __name__ == '__main__':
    main()