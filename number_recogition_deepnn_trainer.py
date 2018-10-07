"""
A FEED-FORWARD DEEP NEURAL NETWORK
"""

import pickle
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.linalg as la
import seaborn as sns

np.set_printoptions(formatter={'float': '{: 0.1f}'.format})


# Batch Normalization
def batch_norm_ff(modes, v, gamma_bn, beta_bn, i, bnorm):
    if bnorm:
        eps = 1.0e-1
        momenti = 0.9

        global running_mean, running_variance

        gamma = gamma_bn + 0
        beta = beta_bn + 0

        v_in = v + 0

        m_dim, n_dim = np.shape(v_in)

        if modes == 'train':
            means = np.mean(v_in, axis=0)
            variances = np.var(v_in, axis=0)
            va = v_in - means
            vx = np.sqrt((variances) + eps) + eps
            v_norm = (v_in - means) / (np.sqrt(variances + eps) + eps)

            v_out_bn = (gamma * v_norm) + beta

            # estimate running averages for test and validation
            running_mean[i] = (momenti * running_mean[i]) + (1 - momenti) * means
            running_variance[i] = (momenti * running_variance[i]) + (1 - momenti) * variances

            cache = [v_norm, v_in, means, variances, m_dim, gamma, beta]

            return [v_out_bn, cache]

        if modes == 'test' or modes == 'validate':
            v_norm = (v_in - running_mean[i]) / (np.sqrt(running_variance[i]) + eps)
            v_out_bn = (gamma_bn * v_norm) + beta_bn

            return v_out_bn

    if not bnorm and modes == 'test':
        return v

    return [v, 0]


def batch_norm_bp(delta, store, bnorm):
    if bnorm:
        v_norm, v_in, means, variance, m_dim, gamma, beta = store
        eps = 1.0e-8

        delta_in = delta + 0

        dgamma = np.sum((delta_in * v_norm), axis=0)
        dbeta = np.sum(delta_in, axis=0)

        inv_std = 1. / (np.sqrt(variance) + eps)

        dv_norm = delta_in * gamma
        dvar =  -0.5 * (inv_std ** 3) * np.sum(dv_norm *(v_in - means), axis = 0)
        dmean = -1 * inv_std * np.sum(dv_norm, axis=0) + dvar * -2.0  * np.mean((v_in - means), axis=0)

        ddelta = (inv_std * dv_norm) + (2.0 / m_dim * (v_in - means) * dvar) + (dmean / m_dim)

        # dx1 = gamma * t / m_dim
        # dx2 = (m_dim * delta_in) - np.sum(delta_in, axis=0)
        # dx3 = np.square(t) * (v_in - means)
        # dx4 = np.sum(delta_in * (v_in - means), axis=0)
        #
        # ddelta = dx1 * (dx2 - (dx3 * dx4))

        return ddelta, dgamma, dbeta

    return [delta, 0, 0]


def bn_term_update(g, b, dg, db, momentsg, momentsb):
        eps = 1.0e-8
        dwg = alpha * dg
        dwb = alpha * db
        beta = 0.9

        momentsg = (beta * momentsg) + ((1 - beta) * np.square(dg))
        momentsb = (beta * momentsb) + ((1 - beta) * np.square(db))
        rms_momentg= np.sqrt(momentsg) + eps
        rms_momentb = np.sqrt(momentsb) + eps

        g += dwg / rms_momentg
        b += dwb / rms_momentb

        return g, b


# Weighted sum of input nodes and weights
def weight_sum(x_data, weights):
    v = x_data.dot(weights)
    return v


# Activation functions
def activation(v, mode):
    y_io = 0
    if mode == 'reLU':
        y_io = v + 0
        np.putmask(y_io, y_io < 0, [0])
        # y = y * (y > 0)np.maximum(y, 0, y)

    if mode == 'leaky_reLU':
        y_io = v + 0
        np.putmask(y_io, y_io < 0, y_io * 0.01)

    if mode == 'sigmoid':
        y_io = 1 / (1 + np.exp(-v))

    if mode == 'softmax':
        ex = np.exp(v)
        sum_exp = ex.sum(axis=1)[:, np.newaxis]
        # out2 = (ex.T / sum_exp).T
        # out3 = (np.exp(v).T / np.sum(np.exp(v), axis=1)).T
        # ex / sum_exp[:, np.newaxis]  # or [:,None]
        # y = np.exp(v) / (np.sum(np.exp(v), axis=1)[:, np.newaxis]))
        y_io = ex / sum_exp
    return y_io


# Delta GRADIENT Rule
def delta_grad(y_in, e, mode):
    d_in = 0

    if mode == 'sigmoid':
        d_in = (y_in * (1 - y_in))

    if mode == 'reLU':
        d_in = y_in + 0
        # d = 1 * (d > 0)
        np.putmask(d_in, d_in > 0, [1])
        np.putmask(d_in, d_in < 0, [0])

    if mode == 'leaky_reLU':
        d_in = y_in + 0
        np.putmask(d_in, d_in > 0, [1])
        np.putmask(d_in, d_in < 0, [0.01])

    return d_in * e


# Backward Error calculation for Hidden layers
def error_h(delta, w):
    e_h = delta.dot(w.T)
    return e_h


def regularization(weights, opt=1):
    if opt != 0:
        lda = 0.001
        return lda * weights  # regularization improves learning accuracy
    else:
        return 0


# Weight Update Optimization Techniques against Vanishing Gradients -
# Advanced Gradient Descents for stability and better performance
def weight_update(x_data, weights_in, it, delta, momentums, moments, mode="SGD"):
    beta = 0.9

    g = x_data.T.dot(delta)
    dw = alpha * (g + regularization(weights_in))

    dW = dw

    if mode == 'Momentum':
        momentums = dw + (momentums * beta)
        dW = momentums

    if mode == 'NAG':  # Nesterov Accelerated Gradient
        momentums_old = momentums
        momentums = (momentums * beta) + dw
        dW = (momentums_old * beta) + ((1 + beta) * momentums)

    if mode == 'AdaGrad':
        eps = 1.0e-8
        dw = alpha * g

        moments += np.square(g)
        rms_moment = np.sqrt(moments) + eps

        dW = dw / rms_moment

    if mode == 'AdaDelta':
        eps = 1.0e-8

        moments = (beta * moments) + ((1 - beta) * np.square(g))
        rms_moment = np.sqrt(moments) + eps

        momentums = (beta * momentums) + ((1 - beta) * np.square(dw))
        rms_momentum = np.sqrt(momentums) + eps

        dW = rms_momentum * g / rms_moment

    if mode == 'RMSProp':
        eps = 1.0e-8
        dw = alpha * g
        beta = 0.9

        moments = (beta * moments) + ((1 - beta) * np.square(g))
        rms_moment = np.sqrt(moments) + eps

        dW = dw / rms_moment

    if mode == 'Adam':
        eps = 1.0e-8
        beta_1 = 0.9
        beta_2 = 0.99

        ts = it + 1

        momentums = (beta_1 * momentums) + (1 - beta_1) * g
        moments = (beta_2 * moments) + (1 - beta_2) * np.square(g)

        momentums_norm = momentums / (1 - np.power(beta_1, ts))
        moments_norm = moments / (1 - np.power(beta_2, ts))

        rms_moment = np.sqrt(moments_norm) + eps

        dW = (alpha * momentums_norm) / rms_moment

    if mode == 'AdaMax':
        eps = 1.0e-8
        beta_1 = 0.9
        beta_2 = 0.99

        ts = it + 1

        momentums = (beta_1 * momentums) + (1 - beta_1) * g
        m_norm = (beta_2 * moments) + eps
        moments = np.maximum(m_norm, np.abs(g))

        momentums_norm = momentums / (1 - np.power(beta_1, ts))

        dW = (alpha / (moments + eps)) * momentums_norm

    if mode == 'NAdam':
        eps = 1.0e-8
        beta_1 = 0.9
        beta_2 = 0.99

        ts = it + 1

        momentums = (beta_1 * momentums) + (1 - beta_1) * g

        moments = (beta_2 * moments) + (1 - beta_2) * np.square(g)

        momentums_norm = momentums / (1 - np.power(beta_1, ts))
        moments_norm = moments / (1 - np.power(beta_2, ts))

        rms_moment = np.sqrt(moments_norm) + eps

        nestrov_param = ((beta_1 * momentums_norm) + (1 - beta_1) * g) / (1 - np.power(beta_1, ts))
        dW = (alpha * nestrov_param) / rms_moment

    if mode == 'NAdaMax':
        eps = 1.0e-8
        beta_1 = 0.9
        beta_2 = 0.99

        ts = it + 1

        momentums = (beta_1 * momentums) + (1 - beta_1) * g
        m_norm = (beta_2 * moments) + eps
        moments = np.maximum(m_norm, np.abs(g))

        momentums_norm = momentums / (1 - np.power(beta_1, ts))

        nestrov_param = ((beta_1 * momentums_norm) + (1 - beta_1) * g) / (1 - np.power(beta_1, ts))
        dW = (alpha * nestrov_param) / (moments + eps)

    if mode == 'AdaDeltaMax':
        a = 0  # do nothing

    return weights_in + dW


# Drop-out : To prevent Over-fitting # TODO: Dropout causes an unstable learning model in nature
def drop_out(y_in, drops, drop_ratio=0.2):
    # drop-ratio or drop-percent: drops out this percentage from the hidden nodes,
    # by setting its output to zero
    v_in = 1
    if drops:
        # drop_ratio = 1 - drop_ratio
        # size = y_in.shape
        # v_in = np.random.binomial(1, drop_ratio, size=y_in.shape) / drop_ratio
        p = drop_ratio / (1 - drop_ratio)
        p = np.sqrt(p)

        # elements = np.size(y)
        my, ny = np.shape(y_in)
        v_in = np.zeros([my, ny])

        num_of_elem_nodrop = np.round(ny * (1 - drop_ratio))
        elem_index = np.random.choice(ny, int(num_of_elem_nodrop), replace=False)

        for i in range(my):
            # elem_index = np.random.choice(ny, int(num_of_elem_nodrop), replace=False)
            np.put(v_in[i, :], elem_index, [p], mode='raise')

    return v_in


# Back propagation, cross-entropy driven learning algorithm
def back_prop_ce_multi_class(modus, x, d=None, weights=None, ls=None, it=0, bn_terms=None):
    set_activation_modes()

    global bnorm

    if ls is None:
        ls = layer_space

    h = ls - 1
    # n = len(x)

    # output
    u = x
    y = np.zeros(ls, dtype=object) # or np.array or np.asarray([None] * ls)
    cache = np.zeros(ls, object)
    drop_cache = np.zeros(h, object)
    loss = 0

    # TEST MODE
    if modus == 'test':
        global bn_term
        gamma_bn, beta_bn = bn_term
        global running_weight
        print('...TEST MODE...\n')
        for i in range(ls):
            # v = weight_sum(u, weights[i])
            # y = sigmoid(v)
            if i == h:
                v = weight_sum(u, running_weight[i])
                y = activation(v, acto_mode)
            else:
                v = weight_sum(u, running_weight[i])
                v = batch_norm_ff('test', v, gamma_bn[i], beta_bn[i], i, bnorm)
                y = activation(v, acth_mode)
                u = y

        return y

    gamma_bn, beta_bn = bn_terms

    # TRAIN MODE
    for i in range(ls):
        # v = weight_sum(u, weights[i])
        # y = sigmoid(v)
        if i == h:
            v = weight_sum(u, weights[i])
            y[i] = activation(v, acto_mode)
        else:
            v = weight_sum(u, weights[i])
            v, cache[i] = batch_norm_ff('train', v, gamma_bn[i], beta_bn[i], i, bnorm)
            y[i] = activation(v, acth_mode)
            # drop_cache[i] = drop_out(y[i], drop, 0.2)
            y[i] *= drop_out(y[i], drop, 0.2)  # drop_cache[i]
            u = y[i]

    e = d - y[h]  # output error
    # drop_cache = drop_cache[::-1]
    ex, ey = np.shape(e)
    # loss = np.square(e)
    # avg_loss = np.sum(loss, axis=1) / ey
    # total_avg_loss = np.sum(avg_loss, axis=0) / ex

    # QUICK CALCULATION OF AVERAGE TRAINING ACCURACY AND LOSS
    ya = y[h] + 0
    dmax = np.argmax(d, axis=1)

    loss = np.asarray([np.square(d[i, dmax[i]] - ya[i, dmax[i]]) for i in range(ex)])
    total_avg_loss = np.sum(loss, axis=0) / ex

    accuracy = np.asarray([ya[i, dmax[i]] / d[i, dmax[i]] * 100 for i in range(ex)])
    total_avg_accuracy = np.sum(accuracy, axis=0) / ex

    # VALIDATION MODE
    if modus == 'validate':
        return [y[h], total_avg_loss, total_avg_accuracy]

    # TRAIN MODE: BACK-PROPAGATION
    deltas_r = np.zeros(ls, object)
    errors_r = np.zeros(ls, object)
    dgamma = np.zeros(ls, object)
    dbeta = np.zeros(ls, object)

    for i in range(h, -1, -1):
        if i == h:
            delta = e + 0
        else:
            delta = delta_grad(y[i], e, acth_mode)
            delta *= drop_out(delta, drop, 0.2)  # drop_cache[i]
            delta, dgamma[h - i], dbeta[h - i] = batch_norm_bp(delta, cache[i], bnorm)

        deltas_r[h - i] = delta
        errors_r[h - i] = e

        if i == 0:
            break

        e = error_h(delta, weights[i])

    deltas = deltas_r[::-1]
    # errors_r = errors_r[::-1]

    dgamma = dgamma[::-1]
    dbeta = dbeta[::-1]

    # WEIGHTS and BATCH TERMS ADJUSTMENTS
    uy = x
    for i in range(ls):
        weights[i] = weight_update(uy, weights[i], it,
                                   deltas[i], momentum[i], moment[i], w_mode)

        if i < ls - 1 and bnorm:
            gamma_bn[i], beta_bn[i] = bn_term_update(gamma_bn[i], beta_bn[i], dgamma[i],
                                                     dbeta[i], momentsg[i], momentsb[i])

        uy = y[i]

    return [weights, y[h], total_avg_loss, total_avg_accuracy, gamma_bn, beta_bn]


# CREATING THE VALIDATION PROCESS
def validate_trains_deepnn(x, d, ls, t_weight, bn_terms):
    nv = len(x)
    ncv = int(np.size(x) / nv)
    yv, valid_loss, valid_acc = back_prop_ce_multi_class('validate', x.reshape([nv, ncv]), d, t_weight, ls, it,
                                                         bn_terms)
    print('Epochs: {0:4d} / {1:4d} {4:^5s} '
          'Average Validation Loss: {2:2.4e} {5:^5s} '
          'Average Validation Accuracy: {3:3.4f}'.format(it, epoch, valid_loss, valid_acc, '|', '|'), end='\n')

    return yv, valid_loss, valid_acc


# CREATING THE TRAINED MULTI-CLASS NEURAL NETWORK
def neural_net(x, d, ls, nodes):
    # Weights Initialization
    weights = np.zeros(ls, object)

    for i in range(ls):
        r = 1 * np.sqrt(6 / (nodes[i] + nodes[i + 1]))
        #  weights[i] = 2 * np.random.uniform(0, 1, [nodes[i], nodes[i + 1]]) - 1
        weights[i] = np.random.uniform(-r, r, [nodes[i], nodes[i + 1]])

    gamma_bn = np.ones(ls-1, object)
    beta_bn = np.zeros(ls-1, object)

    for i in range(ls - 1):
        gamma_bn[i] = np.ones((nodes[i + 1]))
        beta_bn[i] = np.zeros((nodes[i + 1]))

    bn_terms = [gamma_bn, beta_bn]

    global momentum, moment, rms
    momentum = np.zeros_like(weights)
    moment = np.zeros_like(weights)

    global momentsg, momentsb
    momentsg = np.ones_like(gamma_bn)
    momentsb = np.zeros_like(beta_bn)

    global alpha
    # 0.01 is a stable learning rate. You can change between 0.1, 0.001 and 0.01, then visualize the learn curve
    alpha = 0.01
    print('Learning Rate: ', alpha)

    epochs = set_weight_optimizer()
    # set_activation_modes()

    weights_out, y, train_loss, train_acc, yv = 0, 0, 0, 0, 0

    tk, rk = 0, 0

    global it
    it_min_lim, it_max_lim = 0, (epochs-1+0)

    loss_cache = np.zeros(epochs, object)
    acc_cache = np.zeros(epochs, object)
    lossv_cache = np.zeros(epochs, object)
    accv_cache = np.zeros(epochs, object)

    for it in range(epochs):  # it === current epoch
        weights_out, y, train_loss, train_acc, gamma_bn, beta_bn = back_prop_ce_multi_class('train', x.reshape([n, nc]),
                                                                                            d, weights, ls,
                                                                                            it, bn_terms)
        print('Epochs: {0:4d} / {1:4d} {4:^5s} '
              'Average Training Loss: {2:2.4e} {5:^10s} '
              'Average Training Accuracy: {3:<3.4f}'.format(it, epochs, float(train_loss), float(train_acc), '|', '|'),
              end='\n')

        yv, valid_loss, valid_acc = validate_trains_deepnn(VX, VD, layer_space, weights_out, bn_terms)

        loss_cache[it] = train_loss
        acc_cache[it] = train_acc
        lossv_cache[it] = valid_loss
        accv_cache[it] = valid_acc

        if train_loss < 1.0e-8 and valid_loss < 1.0e-3:
            tk += 1
            print('Patience Limit: ', tk)
            if tk > (np.sqrt(epochs) * 0.5 + 0):
                print('Total Epochs:', it)
                it_max_lim = it + 0
                break

        if not drop and it > (np.sqrt(epochs) * 0.5 + 0):
            sub_loss_check = lossv_cache[it-1] - lossv_cache[it]
            if 1.0e-6 >= sub_loss_check <= 0:
                rk +=1
                print('Dull Limit: ', rk)
                if rk > 64:
                    print('Total Epochs:', it)
                    it_max_lim = it + 0
                    break

    visualize(it_max_lim, loss_cache[:it_max_lim], acc_cache[:it_max_lim], loss_text='Average Training Loss',
              acc_text='Average Training Accuracy', im_name='deep_test_viz.png')

    visualize(it_max_lim, lossv_cache[:it_max_lim], accv_cache[:it_max_lim], loss_text='Average Validation Loss',
              acc_text='Average Validation Accuracy', im_name='deep_validation_viz.png')

    print(y, yv, sep='\n\n', end='\n')

    validation_loss = lossv_cache[it_max_lim] + 0
    print(validation_loss)

    return [weights_out, validation_loss, it_max_lim, gamma_bn, beta_bn]


# Plot Graphical Visualization
# See what is going on
def visualize(it_max_lim, loss_cache, acc_cache, loss_text, acc_text, im_name):

    mean_acc = np.mean(acc_cache)
    std_acc = np.std(loss_cache)

    plt.close('all')

    sns.set_style('ticks')
    sns.set_context('paper')
    sns.despine()

    # print(len(loss_cache))
    pt_gd = w_mode
    pt_alpha = r'$\mathtt{\alpha}$'
    pt_actv_symbol = r'$\mathsf{\phi(\upsilon)}$'
    pt_drop = str(drop)
    pt_actvfunh = acth_mode
    pt_actvfuno = acto_mode
    pt_bnmode = str(bnorm)
    pt_title = 'Learning Rate, {5}: {6} | Epochs: {9}/{8}\nGradient Descent Optimization: {0}\n' \
               'Activation Functions {7}: Hidden Layers [{1}] = {2} | ' \
               'Output Layer = {3}\nDrop-Out: {4} | Batch-Normalization: {10}\n '\
               'Mean Average Accuracy: {11:.2f}% ... Average Standard Deviation : {12:.4f}' \
        .format(pt_gd, H, pt_actvfunh, pt_actvfuno, pt_drop, pt_alpha,
                alpha, pt_actv_symbol, epoch, it_max_lim + 1, pt_bnmode, mean_acc, std_acc)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    fig.suptitle(pt_title, fontweight='bold', fontsize='11', fontname='Romana BT')
    plt.grid(True)
    plt.subplots_adjust(top=0.8, wspace=0.2, hspace=0.4)
    # fig.set_size_inches(16,14)

    ax = plt.subplot2grid((2, 1), (0, 0))
    ax.plot(loss_cache, 'r:', linewidth=2)
    ax.grid(True)
    ax.set_xlim([0, it_max_lim + np.sqrt(it_max_lim)/2])
    # ax1.set_title(pt_title, fontweight='bold', fontsize='11', fontname='Bell MT')
    ax.set_xlabel('Epochs', fontsize='12', fontname='Romana BT')
    ax.set_ylabel(loss_text, fontsize='12', fontname='Romana BT')

    ax = plt.subplot2grid((2, 1), (1, 0))
    ax.plot(acc_cache, 'b-.', linewidth=2)
    ax.grid(True)
    ax.set_xlim([0, it_max_lim + np.sqrt(it_max_lim)/2])
    ax.set_xlabel('Epochs', fontsize='12', fontname='Romana BT')
    ax.set_ylabel(acc_text, fontsize='12', fontname='Romana BT')

    plt.savefig(im_name)
    # plt.show()

    rt_visual(acc_cache, acc_text, it_max_lim, loss_cache, loss_text, pt_title)


def rt_visual(acc_cache, acc_text, it_max_lim, loss_cache, loss_text, pt_title):
    # MONKEY PATCH!!
    def _blit_draw(self, artists, bg_cache):
        # Handles blitted drawing, which renders only the artists given instead
        # of the entire figure.
        updated_ax = []
        for a in artists:
            # If we haven't cached the background for this axes object, do
            # so now. This might not always be reliable, but it's an attempt
            # to automate the process.
            if a.axes not in bg_cache:
                # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
                # change here
                bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
            a.axes.draw_artist(a)
            updated_ax.append(a.axes)

        # After rendering all the needed artists, blit each axes individually.
        for ax in set(updated_ax):
            # and here
            # ax.figure.canvas.blit(ax.bbox)
            ax.figure.canvas.blit(ax.figure.bbox)

    animation.Animation._blit_draw = _blit_draw

    fig = plt.figure(2, figsize=(8, 6), dpi=150)
    fig.suptitle(pt_title, fontweight='bold', fontsize='11', fontname='Romana BT')
    plt.grid(True)
    plt.subplots_adjust(top=0.8, wspace=0.2, hspace=0.4)
    ax = plt.subplot2grid((2, 1), (0, 0))
    line, = ax.plot([], [], 'r:', lw=1)
    ax.grid(True)
    ax.set_xlabel('Epochs', fontsize='12', fontname='Romana BT')
    ax.set_ylabel(loss_text, fontsize='12', fontname='Romana BT')
    xmax = int(it_max_lim / 4)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.01, np.max(loss_cache) + 0.01)
    axi = plt.subplot2grid((2, 1), (1, 0))
    linei, = axi.plot([], [], 'b-.', lw=1)
    axi.grid(True)
    axi.set_xlabel('Epochs', fontsize='12', fontname='Romana BT')
    axi.set_ylabel(acc_text, fontsize='12', fontname='Romana BT')
    xmax = int(it_max_lim / 4)
    axi.set_xlim(0, xmax)
    axi.set_ylim(-0.01, np.max(acc_cache) + 0.01)
    x, y, yi = [], [], []

    def data_gen(t=-1):
        global loss_cache, acc_cache
        while t <= it_max_lim:
            t += 1
            yield t, loss_cache[t], acc_cache[t]

    def init():
        line.set_data(x, y)
        linei.set_data(x, yi)

        return line, linei, ax.xaxis, axi.xaxis,

    def run(data):
        global xmax

        a, b, c = data
        x.append(a)
        y.append(b)
        yi.append(c)

        xmin, xmax = ax.get_xlim()
        zoom_factor = 0.1
        p = 0.5 * xmax * (1 + zoom_factor)
        #

        if a >= xmax:
            if it_max_lim - a <= it_max_lim / 2:
                ax.set_xlim(a - xmax, it_max_lim)
            else:
                ax.set_xlim(a - xmax, a + p)
        else:
            # makes it look ok when the animation loops
            ax.set_xlim(0, xmax)

        line.set_data(x, y)
        linei.set_data(x, yi)

        return line, linei, ax.xaxis, axi.xaxis,

        # ax.set_xlim(x[0], (x[-1]*0.5)+p)
        # ax_ii.set_xlim(x[0], (x[-1]*0.5)+p)
        #
        # line.set_data(x, y)
        # line1.set_data(x, yi)
        #
        # return line, linei, ax.xaxis, axi.xaxis

    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=1,
                                  repeat=True, init_func=init)
    plt.show()


def set_activation_modes():
    global acth_mode
    global acto_mode
    global drop
    global bnorm

    activation_functions = {'1': 'sigmoid', '2': 'reLU', '3': 'leaky_reLU', '4': 'softmax'}

    acth_mode = activation_functions['3']
    acto_mode = activation_functions['4']

    bnorm = False
    drop = False
def set_weight_optimizer():
    weight_optimization = {'1': 'SGD', '2': 'Momentum', '3': 'NAG', '4': 'AdaGrad', '5': 'RMSProp',
                           '6': 'AdaDelta', '7': 'Adam', '8': 'AdaMax', '9': 'NAdam', '10': 'NAdaMax',
                           '11': 'AdaDeltaMax'}

    global w_mode
    w_mode = weight_optimization['5']
    global epoch
    epoch = 4096  # multiple of 4 - 128 * 8 int(1024 * 2)

    # if w_mode == 'RMSProp' or w_mode == 'AdaGrad':
    #     drops = True
    #     epoch = 4096  # inducing manual early stopping

    print(w_mode)

    return epoch


# Configure Training Data
N1 = np.array([
    [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]
])
N11 = np.array([
    [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]
])
N2 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [0, 1, 1, 1, 1]
])
N22 = np.array([
    [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]
])
N3 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]
])
N4 = np.array([
    [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]
])
N5 = np.array([
    [0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]
])
N51 = np.array([
    [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]
])

X = np.array([N1, N11, N2, N22, N3, N4, N5, N51])
D = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])

nodes_list = [25, 20, 20, 20, 5]  # format: input nodes - hidden layer(s) nodes - output nodes
n = len(X)
nc = int(np.size(X) / n)
H = 3

layer_space = H + 1

# Validation
XV1 = np.array([
    [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]
])
XV2 = np.array([
    [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]
])
XV3 = np.array([
    [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]
])
XV5 = np.array([
    [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]
])
XC1 = np.array([
    [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]
])
XC11 = np.array([
    [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 1]
])
XC2 = np.array([
    [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]
])
XC3 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]
])
XC5 = np.array([
    [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]
])

VX = np.array([XV1, XV2, XV3, XV5, XC1, XC11, XC2, XC3, XC5])
VD = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]
               ])

# validate_test_deepnn(VX, VD, layer_space, weight)

val_loss = 1
running_weight = 0
running_mean = np.zeros(layer_space, object)
running_variance = np.zeros(layer_space, object)
gamma_bn = 1
beta_bn = 0
try:
    r_file = open('weight.pki', 'rb')
    running_weight = pickle.load(r_file)
    # print(running_weight)
    r_file.close()

    rbn_file = open('bnmean.pki', 'rb')
    running_mean = pickle.load(rbn_file)
    # print(running_mean)
    rbn_file.close()

    rvbn_file = open('bnvariance.pki', 'rb')
    running_variance = pickle.load(rvbn_file)
    # print(running_variance)
    rvbn_file.close()

    rgbn_file = open('bn_gamma.pki', 'rb')
    gamma_bn = pickle.load(rgbn_file)
    rgbn_file.close()

    rbbn_file = open('bn_beta.pki', 'rb')
    beta_bn = pickle.load(rbbn_file)
    rbbn_file.close()

    bn_term = [gamma_bn, beta_bn]

except FileNotFoundError:
    running_weight = 0
    running_mean = np.zeros(layer_space, object)
    running_variance = np.zeros(layer_space, object)
    # print('first run')

if __name__ == '__main__':
    t0 = time.clock()
    weight, v_loss, iterations, gamma_bn, beta_bn = neural_net(X, D, layer_space, nodes_list)
    t1 = time.clock()
    print(f'Time: {t1-t0:.2f} seconds, Loops: {iterations}')

    # Save the optimal trained weights and its corresponding average validation error
    # if on next run, the new trained weights , corresponding error is less than the former,
    # then replace and make it the running weight for our test neural net. else leave it
    #

    try:
        lsr_file = open('loss.pki', 'rb')
        val_loss = pickle.load(lsr_file)
        lsr_file.close()
        print('Current Model Validation Error: ', v_loss, '\nSaved Model Validation Error: ', val_loss)
        print('Later Runs')
    except FileNotFoundError:
        print('curr', v_loss, 'saved', val_loss)
        ls_file = open('loss.pki', 'wb')
        pickle.dump(v_loss, ls_file)
        ls_file.close()

        w_file = open('weight.pki', 'wb')
        pickle.dump(weight, w_file)
        w_file.close()
        print('first run')

        rsbn_file = open('bnmean.pki', 'wb')
        pickle.dump(running_mean, rsbn_file)
        rsbn_file.close()

        rvbn_file = open('bnvariance.pki', 'wb')
        pickle.dump(running_variance, rvbn_file)
        rvbn_file.close()

        rgw_file = open('bn_gamma.pki', 'wb')
        pickle.dump(gamma_bn, rgw_file)
        rgw_file.close()

        rbw_file = open('bn_beta.pki', 'wb')
        pickle.dump(beta_bn, rbw_file)
        rbw_file.close()

    if v_loss < val_loss:
        print('Current Model Accuracy better than Saved Model Accuracy\n.......')
        time.sleep(2)
        print('Setting Current Model to Saved Optimal Model')
        ls_file = open('loss.pki', 'wb')
        pickle.dump(v_loss, ls_file)
        ls_file.close()

        w_file = open('weight.pki', 'wb')
        pickle.dump(weight, w_file)
        w_file.close()

        r_file = open('weight.pki', 'rb')
        running_weight = pickle.load(r_file)
        r_file.close()
        # print(running_weight)

        rsbn_file = open('bnmean.pki', 'wb')
        pickle.dump(running_mean, rsbn_file)
        rsbn_file.close()

        rvbn_file = open('bnvariance.pki', 'wb')
        pickle.dump(running_variance, rvbn_file)
        rvbn_file.close()

        rgw_file = open('bn_gamma.pki', 'wb')
        pickle.dump(gamma_bn, rgw_file)
        rgw_file.close()

        rbw_file = open('bn_beta.pki', 'wb')
        pickle.dump(beta_bn, rbw_file)
        rbw_file.close()

    if v_loss > val_loss:
        print('Current Model Accuracy less than Saved Model Accuracy\n.......')
        time.sleep(2)
        print('Defaulting to Saved Optimal Trained Model')
        r_file = open('weight.pki', 'rb')
        running_weight = pickle.load(r_file)
        r_file.close()
        # print(running_weight)
