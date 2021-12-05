import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import logging
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
from collections import defaultdict
import warnings
import cit_gan
from scipy.stats import rankdata
import xlwt
from tempfile import TemporaryFile
import scipy
import gan_utils
tf.random.set_seed(42)
np.random.seed(42)

'''
Inputs:
 - z: Confounder variables, this is the conditioning set
 - x: First target variable
 - y: Second target variable
'''


def same(x):
    return x


def cube(x):
    return np.power(x, 3)


def negexp(x):
    return np.exp(-np.abs(x))


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
                            debug=False, normalize=True, standardise=False, seed=None, dist_z='gaussian'):
    '''
    Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        f1, f2 to be within {x,x^2,x^3,tanh x, e^{-|x|}, cos x}
    Output:
        Samples X, Y, Z
    '''
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if fixed_function == 'linear':
        f1 = same
        f2 = same
    else:
        I1 = random.randint(2, 6)
        I2 = random.randint(2, 6)

        if I1 == 2:
            f1 = np.square
        elif I1 == 3:
            f1 = cube
        elif I1 == 4:
            f1 = np.tanh
        elif I2 == 5:
            f1 = negexp
        else:
            f1 = np.cos

        if I2 == 2:
            f2 = np.square
        elif I2 == 3:
            f2 = cube
        elif I2 == 4:
            f2 = np.tanh
        elif I2 == 5:
            f2 = negexp
        else:
            f2 = np.cos
    if debug:
        print(f1, f2)

    num = size

    if dist_z == 'gaussian':
        cov = np.eye(dz)
        mu = np.ones(dz)
        Z = np.random.multivariate_normal(mu, cov, num)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z, (num, dz))

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)

    # Axy = np.random.rand(dx, dy)
    Axy = np.ones((dx, dy)) * 0.001
    for i in range(dy):
        Axy[:, i] = Axy[:, i] / np.linalg.norm(Axy[:, i], ord=1)

    temp = np.matmul(Z, Ax)
    m = np.mean(np.abs(temp))
    nstd = nstd * m

    if sType == 'CI':
        X = f1(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = f1(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = f2(2.0 * np.matmul(X, Axy) + np.matmul(Z, Ay))

    if normalize:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)


def pc_ks(pvals):
    """ Compute the area under power curve and the Kolmogorov-Smirnoff
    test statistic of the hypothesis that pvals come from the uniform
    distribution with support (0, 1).
    """
    if pvals.size == 0:
        return [-1, -1]
    if -1 in pvals or -2 in pvals:
        return [-1, -1]
    pvals = np.sort(pvals)
    cdf = ecdf(pvals)
    auc = 0
    for (pv1, pv2) in zip(pvals[:-1], pvals[1:]):
        auc += integrate.quad(cdf, pv1, pv2)[0]
    auc += integrate.quad(cdf, pvals[-1], 1)[0]
    _, ks = kstest(pvals, 'uniform')
    return auc, ks


def np2r(x):
    """ Convert a numpy array to an R matrix.
    Args:
        x (dim0, dim1): A 2d numpy array.
    Returns:
        x_r: An rpy2 object representing an R matrix isometric to x.
    """
    if 'rpy2' not in sys.modules:
        raise ImportError(("rpy2 is not installed.",
                           " Cannot convert a numpy array to an R vector."))
    try:
        dim0, dim1 = x.shape
    except IndexError:
        raise IndexError("Only 2d arrays are supported")
    return R.r.matrix(R.FloatVector(x.flatten()), nrow=dim0, ncol=dim1)


def fdr(truth, pred, axis=None):
    """ Computes False discovery rate
    """
    return ((pred == 1) & (truth == 0)).sum(axis=axis) / pred.sum(axis=axis).astype(float).clip(1, np.inf)


def tpr(truth, pred, axis=None):
    """ Computes true positive rate
    """
    return ((pred == 1) & (truth == 1)).sum(axis=axis) / truth.sum(axis=axis).astype(float).clip(1, np.inf)


def true_positives(truth, pred, axis=None):
    """ Computes number of true positive
    """
    return ((pred == 1) & (truth == 1)).sum(axis=axis)


def false_positives(truth, pred, axis=None):
    """ Computes number of false positive
    """
    return ((pred == 1) & (truth == 0)).sum(axis=axis)


def bh(p, fdr):
    """ From vector of p-values and desired false positive rate,
    returns significant p-values with Benjamini-Hochberg correction
    """
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k + 1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries, dtype=int)


def mmd_squared(X, Y, gamma=1):
    X = tf.reshape(X, shape=[X.shape[0], 1])
    Y = tf.reshape(Y, shape=[Y.shape[0], 1])

    K_XX = rbf_kernel(X, gamma=gamma)
    K_YY = rbf_kernel(Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) + (np.sum(K_YY) - np.trace(K_YY)) / (
                m * (m - 1)) - 2 * np.sum(K_XY) / (m * n)

    return mmd_squared


def correlation(X, Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return np.abs(np.corrcoef(X, Y)[0, 1])


def kolmogorov(X, Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return ks_2samp(X, Y)[0]


def wilcox(X, Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return wilcoxon(X, Y)[0]


def rdc(x, y, f=np.sin, k=20, s=1 / 6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Source: https://github.com/garydoranjr/rdc
    """
    x = tf.reshape(x, shape=(x.shape[0], ))
    y = tf.reshape(y, shape=(y.shape[0], ))

    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1: x = tf.reshape(x, shape=(-1, 1))
    if len(y.shape) == 1: y = tf.reshape(y, shape=(-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in np.transpose(x)]) / float(x.shape[0])
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in np.transpose(y)]) / float(y.shape[0])

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))


def write_to_excel(p, z_dim, lamda, save_file):
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1')
    for i, e in enumerate(p):
        sheet1.write(i, 0, e)
    book.save('./trained/{}/p_values_z_dim={}_lambda={}.xls'.format(save_file, z_dim, lamda))
    book.save(TemporaryFile())


def plot_results(x, y, lamda, y_axis='Type 1 error'):
    n_seq = len(y)
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
    for i in range(n_seq):
        plt.plot(x, y[i], marker='', markerfacecolor='blue', markersize=12, color=flatui[i % len(flatui)], linewidth=4,
                 label=r'$\lambda$={}'.format(lamda[i]))
    plt.legend()
    plt.xlabel('Dimension of z')
    plt.ylabel(y_axis)
    # plt.title(test)
    plt.show()


def t_and_sigma(psy_x_i, psy_y_i, phi_x_i, phi_y_i):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    t_b = tf.reduce_sum(matrix, axis=1) / tf.cast(n, tf.float64)
    t_b = tf.expand_dims(t_b, axis=1)

    crit_matrix = matrix - t_b
    std_b = tf.sqrt(tf.reduce_sum(crit_matrix**2, axis=1) / tf.cast(n-1, tf.float64))
    return t_b, std_b


def test_statistics(psy_x_i, psy_y_i, phi_x_i, phi_y_i, t_b, std_b, k):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    crit_matrix = matrix - t_b
    test_stat = tf.reduce_max(tf.abs(tf.sqrt(tf.cast(n*2, tf.float64)) * tf.squeeze(t_b) / std_b))

    sig = tf.reduce_sum(crit_matrix[None, :, :] * crit_matrix[:, None, :], axis=2)
    coef = std_b[None, :] * std_b[:, None] * tf.cast(n-1, tf.float64)
    sig_xy = sig / coef

    eigenvalues, eigenvectors = tf.linalg.eigh(sig_xy)
    base = tf.zeros_like(eigenvectors)
    eig_vals = tf.sqrt(eigenvalues + 1e-9)
    lamda = tf.linalg.set_diag(base, eig_vals)
    sig_sqrt = tf.matmul(tf.matmul(eigenvectors, lamda), tf.linalg.inv(eigenvectors))
    # sig_sqrt = tf.matmul(tf.matmul(tf.transpose(eigenvectors), lamda), eigenvectors)
    # sig_inv = tf.linalg.inv(sig_sqrt)

    z_dist = tfp.distributions.Normal(0.0, scale=1.0)
    z_samples = z_dist.sample([b*b, k])
    z_samples = tf.cast(z_samples, tf.float64)
    vals = tf.matmul(sig_sqrt, z_samples)
    t_j = tf.reduce_max(vals, axis=0)
    return test_stat, t_j


def gcit_new(n=500, z_dim=100, simulation='type1error', batch_size=64, n_iter=1000, train_writer=None, current_iters=0):
    # generate samples x, y, z
    # arguments: size, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
    # debug=False, normalize=True, seed=None, dist_z='gaussian'
    x_dims = 1
    y_dims = 1
    if simulation == 'type1error':
        # generate samples x, y, z under null hypothesis - x and y are conditional independent
        x, y, z = generate_samples_random(size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim,
                                          fixed_function='nonlinear', dist_z='gaussian')

    elif simulation == 'power':
        # generate samples x, y, z under alternative hypothesis - x and y are dependent
        x, y, z = generate_samples_random(size=n, dz=z_dim, sType='dependent')
    else:
        raise ValueError('Test does not exist.')

    # define training and testing subsets, training for learning the sampler and
    # testing for computing test statistic. Set 2/3 and 1/3 as default
    # x_train, y_train, z_train = x[:int(2 * n / 3), ], y[:int(2 * n / 3), ], z[:int(2 * n / 3), ]
    k = 2
    x_train, y_train, z_train = x[:int(n / 2), ], y[:int(n / 2), ], z[:int(n / 2), ]
    # build data pipline for test set
    x_test, y_test, z_test = x[int(n / 2):, ], y[int(n / 2):, ], z[int(n / 2):, ]
    # build data pipline for training set
    dataset1 = tf.data.Dataset.from_tensor_slices((x_train, y_train, z_train))
    testset1 = tf.data.Dataset.from_tensor_slices((x_test, y_test, z_test))
    dataset2 = tf.data.Dataset.from_tensor_slices((x_test, y_test, z_test))
    testset2 = tf.data.Dataset.from_tensor_slices((x_train, y_train, z_train))

    # Repeat n epochs
    epochs = int(n_iter)
    dataset1 = dataset1.repeat(epochs)
    batched_train1 = dataset1.shuffle(300).batch(batch_size)
    batched_training_set1 = dataset1.shuffle(300).batch(batch_size)
    batched_test1 = testset1.batch(1)

    dataset2 = dataset2.repeat(epochs)
    batched_train2 = dataset2.shuffle(300).batch(batch_size)
    batched_test2 = testset2.batch(1)
    data_k = [[batched_train1, batched_test1], [batched_train2, batched_test2]]

    # no. of random and hidden dimensions
    if z_dim <= 20:
        v_dims = int(3)
        h_dims = int(3)

    else:
        v_dims = int(z_dim / 10)
        h_dims = int(z_dim / 10)
        # h_dims = 50

    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0 / 3.0))

    # create instance of G & D
    lr = 1e-4
    # input_dims = x_train.shape[1]
    generator_x = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, batch_size)
    generator_y = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, batch_size)
    discriminator_x = cit_gan.WGanDiscriminator(n, z_dim, h_dims, x_dims, batch_size)
    discriminator_y = cit_gan.WGanDiscriminator(n, z_dim, h_dims, y_dims, batch_size)

    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0
    w_clipping_val = 0.5
    w_clipping_norm = 1.0
    scaling_coef = 1000.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30

    gx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
    gy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    @tf.function(experimental_relax_shapes=True)
    def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_x.call(gen_inputs)
        fake_x_p = generator_x.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = \
                gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)[0]
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_x.trainable_variables)
        dx_optimiser.apply_gradients(zip(d_grads, discriminator_x.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_x.call(gen_inputs)
            fake_x_p = generator_x.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss, loss_xy, loss_xx, loss_yy = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                                           sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_x.trainable_variables)
        gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))
        return gen_loss, loss_xy, loss_xx, loss_yy

    @tf.function(experimental_relax_shapes=True)
    def y_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_y.call(gen_inputs)
        fake_x_p = generator_y.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_y.call(d_real)
            f_fake = discriminator_y.call(d_fake)
            f_real_p = discriminator_y.call(d_real_p)
            f_fake_p = discriminator_y.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = \
                gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)[0]
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_y.trainable_variables)
        dy_optimiser.apply_gradients(zip(d_grads, discriminator_y.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def y_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_y.call(gen_inputs)
            fake_x_p = generator_y.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_y.call(d_real)
            f_fake = discriminator_y.call(d_fake)
            f_real_p = discriminator_y.call(d_real_p)
            f_fake_p = discriminator_y.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss, loss_xy, loss_xx, loss_yy = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                                           sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_y.trainable_variables)
        gy_optimiser.apply_gradients(zip(generator_grads, generator_y.trainable_variables))
        return gen_loss, loss_xy, loss_xx, loss_yy

    psy_x_all = []
    phi_x_all = []
    psy_y_all = []
    phi_y_all = []
    test_samples = 30
    test_size = z_test.shape[0]
    M = 40
    count=0
    for batched_trainingset, batched_testset in data_k:
        for x_batch1, y_batch1, z_batch1 in batched_trainingset.take(n_iter):
            if count%20==0:
                print(count)
            count=count+1
            for x_batch2, y_batch2, z_batch2 in batched_training_set1.take(1):
                if x_batch1.shape[0] != batch_size:
                    continue
                # sample noise v
                noise_v = v_dist.sample([batch_size, v_dims])
                noise_v = tf.cast(noise_v, tf.float64)
                noise_v_p = v_dist.sample([batch_size, v_dims])
                noise_v_p = tf.cast(noise_v_p, tf.float64)
                x_update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                loss_x, a, b, c = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                y_update_d(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                loss_y, a, b, c = y_update_g(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)

                with train_writer.as_default():
                    # tf.summary.scalar('Wasserstein X Discriminator Loss', x_disc_loss, step=current_iters)
                    tf.summary.scalar('Wasserstein X GEN Loss', loss_x, step=current_iters)
                    # tf.summary.scalar('Wasserstein Y Discriminator Loss', y_disc_loss, step=current_iters)
                    tf.summary.scalar('Wasserstein Y GEN Loss', loss_y, step=current_iters)
                    train_writer.flush()

                # if tf.abs(x_disc_loss) < 0.1:
                #     break
                current_iters += 1

        psy_x_b = []
        phi_x_b = []
        psy_y_b = []
        phi_y_b = []

        x_samples = []
        y_samples = []
        z = []
        x = []
        y = []

        # the following code generate x_1, ..., x_400 for all B and it takes 61 secs for one test
        for test_x, test_y, test_z in batched_testset:
            tiled_z = tf.tile(test_z, [M, 1])
            noise_v = v_dist.sample([M, v_dims])
            noise_v = tf.cast(noise_v, tf.float64)
            g_inputs = tf.concat([tiled_z, noise_v], axis=1)
            # generator samples from G and evaluate from D
            fake_x = generator_x.call(g_inputs, training=False)
            fake_y = generator_y.call(g_inputs, training=False)
            x_samples.append(fake_x)
            y_samples.append(fake_y)
            z.append(test_z)
            x.append(test_x)
            y.append(test_y)

        standardise = True

        if standardise:
            x_samples = (x_samples - tf.reduce_mean(x_samples)) / tf.math.reduce_std(x_samples)
            y_samples = (y_samples - tf.reduce_mean(y_samples)) / tf.math.reduce_std(y_samples)
            x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
            y = (y - tf.reduce_mean(y)) / tf.math.reduce_std(y)
            z = (z - tf.reduce_mean(z)) / tf.math.reduce_std(z)

        for i in range(test_samples):
            f1 = cit_gan.CharacteristicFunction(M, x_dims, z_dim, test_size)
            f2 = cit_gan.CharacteristicFunction(M, y_dims, z_dim, test_size)

            phi_x = tf.reduce_mean(f1.call(x_samples, z), axis=1)
            phi_y = tf.reduce_mean(f2.call(y_samples, z), axis=1)
            psy_x = tf.squeeze(f1.call(x, z))
            psy_y = tf.squeeze(f2.call(y, z))

            psy_x_b.append(psy_x)
            phi_x_b.append(phi_x)
            psy_y_b.append(psy_y)
            phi_y_b.append(phi_y)
        psy_x_all.append(psy_x_b)
        phi_x_all.append(phi_x_b)
        psy_y_all.append(psy_y_b)
        phi_y_all.append(phi_y_b)

    # reshape
    psy_x_all = tf.reshape(psy_x_all, [k, test_samples, test_size])
    psy_y_all = tf.reshape(psy_y_all, [k, test_samples, test_size])
    phi_x_all = tf.reshape(phi_x_all, [k, test_samples, test_size])
    phi_y_all = tf.reshape(phi_y_all, [k, test_samples, test_size])

    j = 1000
    comparison = []

    t_b = 0.0
    std_b = 0.0
    for n in range(k):
        t, std = t_and_sigma(psy_x_all[n], psy_y_all[n], phi_x_all[n], phi_y_all[n])
        t_b += t
        std_b += std
    t_b = t_b / tf.cast(k, tf.float64)
    std_b = std_b / tf.cast(k, tf.float64)

    for n in range(k):
        stat, critical_vals = test_statistics(psy_x_all[n], psy_y_all[n], phi_x_all[n], phi_y_all[n], t_b, std_b, j)
        check = [c > stat or c == stat for c in critical_vals]
        comparison.append(check)
    # p-value computation as a two-sided test
    p_value = tf.reduce_sum(tf.cast(comparison, tf.float32)) / tf.cast(j*k, tf.float32)

    return p_value






