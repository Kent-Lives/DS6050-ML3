from time import time
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import seaborn as sns

from time import time
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import seaborn as sns

############ Problem 1 Part C ############
## Implement binary and multi-class cross-entropy from scratch

def sigmoid(z: ndarray) -> ndarray:
    '''Compute batched p = sigmoid(z)

    Args:
        z: ndarray with shape (n,). Each z[i] is the ith sample of z.

    Returns:
        p: sigmoid(z) with shape (n,). Each p[i] is the sigmoid of z[i].
    '''
    assert z.ndim == 1, 'z must have shape (n,)'
    p = 1 / (1 + np.exp(-z))
    
    return p




def softmax(Z: ndarray) -> ndarray:
    '''Compute batched P = softmax(Z)

    Args:
        Z: ndarray with shape (n, k). Each Z[i] is the ith sample of Z

    Returns:
        P: softmax(Z) with shape (n, k)
    '''
    assert Z.ndim == 2, 'Z must have shape (n, k)'
    p = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    p /= np.sum(p, axis=1, keepdims=True)

    return p



def nll_binary(X: ndarray, w: ndarray, y: ndarray) -> float:
    '''Compute binary cross entropy from a batch of samples.

    This method should gather the bias into the weights.
    Thus, w[0] should be the bias term
    and w[1:] are the multiplicative terms.

    Args:
        X: batch of samples with shape (n, d)
        w: weights with shape (d+1,)
        y: labels with shape (n,). Each y[i] is either 0 or 1

    Returns:
        nll: negative log likelihood
    '''
    assert X.ndim == 2, 'X must have shape (n, d)'
    n, d = X.shape
    assert w.shape == (d+1,), 'w must have shape (d+1,)'
    assert y.shape == (n,), 'y must have shape (n,)'

    # Xaug[:, 0] is vector of 1s
    Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
    z = Xaug @ w
    nll = np.mean(np.logaddexp(0, -z) + (1 - y) * z)
    
    return nll  


def nll_multiclass(X: ndarray, W: ndarray, Y_onehot: ndarray) -> float:
    '''Compute multi-class cross entropy from a batch of samples

    This method should gather the bias into the weights.
    Thus, W[0, i] is the bias term for the i-th class
    and W[1:, i] are the multiplicative terms for the i-th class.

    Args:
        X: batch of samples with shape (n, d)
        W: weights with shape (d+1, k)
        Y_onehot: Onehot encoded class labels with shape (n, k)

    Returns:
        nll: negative log likelihood
    '''
    assert X.ndim == 2, 'X must have shape (n, d)'
    n, d = X.shape
    assert W.ndim == 2 and W.shape[0] == d+1, 'W must have shape (d+1, k)'
    k = W.shape[1]
    assert Y_onehot.shape == (n, k), 'Y_onehot must have shape (n, k)'

    # Xaug[:, 0] is vector of 1s
    Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
    z = Xaug @ W
    z = z - np.max(z, axis=1, keepdims=True)  
    logProbs = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    nll = -np.mean(np.sum(Y_onehot * logProbs, axis=1))
    return nll


############     Problem 2    ############
## Implement linear regression with normal equations and gradient descent

def linreg_ne(
    X: ndarray,
    Y: ndarray,
    lmbda: float | None
) -> tuple[ndarray, float]:
    '''Linear regression using normal equations.

    This method should gather the bias into the weights.
    Thus W[1:, :] are the multiplicative paramaters Theta
    and W[0, :] is the bias vector B in the equation
    X @ Theta + B = Y

    Only apply ridge regularization if lmbda is not None.

    Args:
        X: batch of samples with shape (n, d)
        Y: batch of targets with shape (n, m)
        lmbda: regularization scale factor

    Returns:
        W: batch of weights with shape (d+1, m)
        runtime: in seconds
    '''
    assert X.ndim == 2, 'X must have shape (n, d)'
    n, d = X.shape
    assert Y.shape[0] == n and Y.ndim <= 2, 'Y must have shape (n, m)'
    if Y.ndim == 1:
        Y = Y[:, None]  ## convert (n,) to (n, 1)

    # Xaug[:, 0] is vector of 1s
    Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
    t_start = time()
    A = Xaug.T @ Xaug
    if lmbda is not None:
        reg = n * lmbda * np.eye(d + 1)
        reg[0, 0] = 0
        A = A + reg
    XtY = Xaug.T @ Y
    W = np.linalg.solve(A, XtY)
    
    t_end = time()
    return W, t_end - t_start
    


def linreg_gd(
    X: ndarray,
    Y: ndarray,
    n_iters: int,
    lr: float
) -> tuple[ndarray, float]:
    '''Linear regression using gradient descent.

    Let W = Ws[i] hold the parameters after the i-th gradient step (0-indexed).
    For example, Ws[0] would hold the parameters after the first gradient step
    and Ws[10] would hold the parameters after the eleventh gradient step.

    Beware indexing errors from 0/1-indexing and the fencepost error!

    This method should gather the bias into the weights.
    Thus W[1:, :] are the multiplicative paramaters Theta
    and W[0, :] is the bias vector B in the equation
    X @ Theta + B = Y

    Args:
        X: batch of samples with shape (n, d)
        Y: batch of targets with shape (n, m)
        n_iters: number of gradient steps
        lr: learning rate

    Returns:
        Ws: batch of weights with shape (n_iters, d+1, m)
        runtime: in seconds
    '''
    assert X.ndim == 2, 'X must have shape (n, d)'
    n, d = X.shape
    assert Y.shape[0] == n and Y.ndim <= 2, 'Y must have shape (n, m)'
    if Y.ndim == 1:
        Y = Y[:, None]  # convert (n,) to (n, 1)
    m = Y.shape[1]

    # Xaug[:, 0] is vector of 1s
    Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
    Ws = np.zeros((n_iters, d+1, m))  # fixed init for reproducability
    W = np.zeros((d+1, m))
    t_start = time()

    for i in range(n_iters):
        grad = 2 * (Xaug.T @ (Xaug @ W - Y)) / n
        W = W - lr * grad 
        Ws[i] = W

    t_end = time()
    return Ws,  t_end - t_start
    

def MSE(Y: ndarray, Yhat: ndarray) -> float:
    '''Compute Mean Squared Error between Y and Yhat

    Args:
        Y: shape (n, m)
        Yhat: shape (n, m)

    Returns:
        mse: mean squared error
    '''
    assert Y.ndim == 2, 'Y must have shape (n, m)'
    assert Y.shape == Yhat.shape, 'Y and Yhave must have same shape'
    mse = float(np.mean((Y - Yhat) ** 2))
    return mse


def plot_runtime_v_feature_dim(
    ds: list[int],
    runtimes_ne: ndarray,
    runtimes_gd: ndarray,
    title: str,
    plotname: str
) -> None:
    '''Plot runtime v. feature dims for NE and GD

    Num. features is on the x-axis. Runtimes (in seconds) is on the y-axis.

    Make sure to label each curve NE or GD.

    Args:
        ds: list of num features. len == k
        runtimes_ne: normal equation runtimes of shape (k,)
        runtimes_gd: gradient descent runtimes of shape (k,)
        title: figure title
    plotname: plot saved to f'{plotname}.png'
    '''
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title(title)
    ax.set_xlabel('Num. Features')
    ax.set_ylabel('Runtime (seconds)')
    ax.grid(visible=True, alpha=0.3)
    ax.plot(ds, runtimes_ne, c='tab:green', label='NE')
    ax.plot(ds, runtimes_gd, c='tab:blue', label='GD')
    ax.legend(loc='upper left')
    fig.tight_layout()
    print(f'Saving {plotname}.png')
    fig.savefig(f'{plotname}.png')
    plt.close(fig)


def plot_gd_iters_v_mse(
    ds: list[int],
    mses_ne: ndarray,
    mses_ne_ridge: ndarray,
    mses_gd: ndarray,
    title: str,
    plotname: str
) -> None:
    '''Plot error v. iterations for GD

    Structure figure as a 3x3 grid of subplots, one for each d in ds.
    Iterations is on the x-axis. MSE is on the y-axis.

    For each d, this method must:
        1) Plot the MSE from the NE as a dashed horizontal line to
           indicate the true optimal solution.
        2) Plot the MSE from the GD over training iterations.
        3) Label both the MSE_NE and MSE_GD lines
        4) Title each plot with the number of features

    This method must create 2 plots.
        1) Full scale showing horizontal lines for NE loss and NE + ridge loss
        2) Zoomed scale showing horizontal line for only NE + ridge loss

    Make sure to label each curve.

    Args:
        ds: list of num features. len == k
        mses_ne: normal equation mse of shape (k,)
        mses_ne_ridge: normal equation + ridge mse of shape (k,)
        mses_gd: gradient descent mse of shape (k, n_iters)
        title: figure title
        plotname: plot saved to f'{plotname}_{plotname_suffix}.png'
    '''
    ncols = 3
    nrows, rem = divmod(len(ds), ncols)
    if rem > 0:
        nrows += 1
    suffs = [('Full Scale', 'fullscale'), ('Zoomed', 'zoomed')]
    for j, (title_suffix, plotname_suffix) in enumerate(suffs):
        # Create 3x3 grid of subplots
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6*ncols, 4*nrows),
            squeeze=False,
            sharex=True
        )
        fig.suptitle(f'{title} ({title_suffix})')
        fig.supxlabel('Iterations')
        fig.supylabel('MSE')
        for i, d in enumerate(ds):
            # select active axes
            ax = axs[*divmod(i, ncols)]
            ax.set_title(rf'$d = {d}$')
            ax.grid(visible=True, alpha=0.3)
            
            mses_ne_ridge_i = mses_ne_ridge[i]
            mses_gd_i = mses_gd[i]
            if j == 0:  # if fullscale
                mses_ne_i = mses_ne[i]
                ax.axhline(mses_ne_i, linestyle='-.', c='tab:green', label='NE')
            ax.axhline(mses_ne_ridge_i, linestyle='-.', c='tab:orange', label='NE Ridge')
            ax.plot(mses_gd_i, c='tab:blue', label='GD')
            ax.legend(loc='upper left')
        fig.tight_layout()
        print(f'Saving {plotname}_{plotname_suffix}.png')
        fig.savefig(f'{plotname}_{plotname_suffix}.png')
        plt.close(fig)



############     Problem 3    ############
## Relevant loss, gradient, and sgd functions for convenience
## taken directly from the linked notebook.

def check_escaped(losses: ndarray, thresh: float) -> ndarray:
    '''Returns 1. if escaped, 0. otherwise.

    Escaped if loss < thresh.

    Args:
        losses: ndarray of arbitrary shape
        thresh: loss threshold

    Returns:
        escaped: boolean ndarray of same shape as losses
    '''
    return losses < thresh


def loss_function(w: ndarray) -> float:
    r'''Two hole loss landscape function from linked notebook.

    Loss landscape with respect to the weights(!) w = [w1, w2].

    This specific function encodes the functions
    \begin{align*}
        L_\text{local}(w) &= -2 \exp \left\{ \frac{- \| w - [1.5, 1.5] \|^2}{0.2} \right\} \\
        L_\text{global}(w) &= -3.5 \exp \left\{ \frac{- \| w + [1.5, 1.5] \|^2}{1.5} \right\}
    \end{align*}
    where
    \[
        L(w) = L_\text{local}(w) + L_\text{global}(w)
    \]
    which has a hole of depth 2 at [1.5, 1.5] and another hole of depth 3.5 at [-1.5, -1.5].

    Args:
        w: ndarray of shape (2,) of floats for the current parameters

    Returns:
        L: total loss = loss_local + loss_global
    '''
    local_min = -2 * np.exp(-np.square(w - 1.5).sum() / 0.2)
    global_min = -3.5 * np.exp(-np.square(w + 1.5).sum() / 1.5)
    return local_min + global_min


def get_gradient_components(w: ndarray) -> tuple[ndarray, ndarray]:
    r'''Two hole loss landscape gradient with respect to w from linked notebook.

    Loss landspace gradient with respect to the weights(!) w = [w1, w2].

    This specific gradient encodes the functions
    \begin{align*}
        L_\text{local}^\prime (w) &= L_\text{local}(w) * (-2) * \frac{w - [1.5, 1.5]}{0.2} \\
        L_\text{global}^\prime (w) &= L_\text{global}(w) * (-2) * \frac{w + [1.5, 1.5]}{1.5}
    \end{align*}
    where
    \[
        L^\prime (w) = L_\text{local}^\prime (w) + L_\text{global}^\prime (w).
    \]

    This function returns the local and global gradient terms separately.

    Args:
        w: ndarray of shape (2,) of floats for the current parameters

    Returns:
        grad_local: ndarray of shape (2,) of floats for $L_\text{local}^\prime$
        grad_global: ndarray of shape (2,) of floats for $L_\text{global}^\prime$
    '''
    local_min = -2 * np.exp(-np.square(w - 1.5).sum() / 0.2)
    global_min = -3.5 * np.exp(-np.square(w + 1.5).sum() / 1.5)

    grad_local = local_min * (-2.) * (w - 1.5) / 0.2
    grad_global = global_min * (-2.) * (w + 1.5) / 1.5

    return grad_local, grad_global


## Part B: Systematic Hyperparameter Study
def run_sgd_improved_analysis(
    start_point: ndarray,
    grad_fn: Callable[[ndarray], tuple[ndarray, ndarray]],
    lr: float,
    max_iterations: int,
    initial_noise: float,
    batch_size: int,
    noise_decay: float,
    escape_chance: float,
    atol: float,
    prng: np.random.Generator
) -> tuple[ndarray, float, int]:
    '''SGD with adaptive noise and better convergence.

    This function extends run_sgd_improved() from the two_hole notebook.

    See hw1_script.problem_3_part_b() to see how this function is called.

    This function should implement how batch size affects sgd noise_scale
    (i.e. simulation stochasticity from batch sampling).

    Keep in mind that this function operates on the loss landscape in parameter space!

    Args:
        start_point: ndarray of shape (2,) of floats for initial weights
        grad_fn: computes current loss gradient based on current w
        lr: sgd learning rate
        max_iterations: max sgd steps. Exits training even if not converged.
        initial_noise: initial stochasticity at the start of training
        batch_size: batch size of sgd steps
        noise_decay: decay rate of noise scale
        escape_chance: percent chance of using grad_global instead of true_grad
                       which is most likely dominated by the local minima
        atol: if ( || true_grad || < atol ) then assume converged
        prng: pseudorandom number generator for escape chance and stochastic grad

    Returns:
        w: ndarray of shape (2,) of floats for final learned parameters after sgd
        runtime: total runtime of sgd in seconds
        iteration: number of iterations until convergence, upper bounded by max_iterations
    '''
    w = start_point.copy()

    #: simulate noise_scale based on batch size
    noise_scale = initial_noise * noise_decay * 1/np.sqrt(batch_size)
#    raise NotImplementedError

    converged = False
    t_start = time()
    # shift range by 1 for 1-based counting of total iterations
    for iteration in range(1, max_iterations+1):
        grad_local, grad_global = grad_fn(w)
        true_grad = grad_local + grad_global

        # Adaptive noise - decreases over time for better fine-tuning
        if prng.random() < escape_chance:
            stochastic_grad = grad_global + prng.standard_normal(size=2) * noise_scale
        else:
            stochastic_grad = true_grad + prng.standard_normal(size=2) * noise_scale

        # Check for convergence (but be more lenient due to noise)
        if np.linalg.norm(true_grad) < atol and noise_scale < 0.1:
            print(f"  SGD converged after {iteration} iterations")
            t_end = time()
            converged = True
            break

        w -= lr * stochastic_grad

        # Decay noise over time (helps with fine convergence)
        noise_scale *= noise_decay

    if not converged:
        t_end = time()

    runtime = t_end - t_start  # type: ignore

    return w, runtime, iteration  # type: ignore


def plot_heatmaps(
    lossname: Literal['two_hole'] | Literal['multi_modal'],
    learning_rates: list[float],
    batch_sizes: list[int],
    noise_scales: list[float],
    losses: ndarray,
    runtimes: ndarray,
    conv_iters: ndarray,
    escaped: ndarray
) -> None:
    '''Plots optimization heatmaps showing metrics per hyperparameter combination.

    n_trials = losses.shape[-1] is an integer representing the number of trials
    sgd was run for each hyperparameter combination.

    This function plots heatmaps where the rows and columns are separate
    hyperparameters (e.g. learning_rates v batch_sizes) and color represents the
    metric of interest (e.g. mean loss over n_trials for that hyperparam combination).

    Metrics of interest:
        1) Mean loss over n_trials
        2) Best loss over n_trials
        3) Escape probability over n_trials
        4) Mean runtime (seconds) over n_trials. Note: runtimes should already by in seconds.
        5) Mean convergence iters over n_trials
        6) Best convergence iters over n_trials

    This implementation creates a hierarchy of plots.
        1) Create subfigures for each metric (we used 3 rows, 2 cols)
        2) Create subplots for each noise scale (we used 2 rows, 2 cols)
        3) Subplot itself is a heatmap with:
            i)   batch size on rows
            ii)  learning rate on cols
            iii) numerical value of metric for ease of reading

    Heatmaps can be easily plotted using the seaborn's sns.heatmap().

    Args:
        lossname: name of current loss landscape
        learning_rates: list of size LR of learning rates
        batch_sizes: list of size B of batch sizes to simulate
        noise_scales: list of size NS initial noise scales
        losses: ndarray of shape (LR, B, NS, n_trials) of floats of losses
        runtimes: ndarray of shape (LR, B, NS, n_trials) of floats of runtimes (in seconds)
        conv_iters: ndarray of shape (LR, B, NS, n_trials) of ints of sgd iterations
        escaped: ndarray of shape (LR, B, NS, n_trials) of bools
    '''
    n_trials = losses.shape[-1]
    # : Collect metrics over n_trials
    # resulting ndarrays will have shape (LR, B, NS)
    #raise NotImplementedError
    losses_mean = np.mean(losses, axis=-1)
    losses_best = np.min(losses, axis=-1)
    runtimes_mean = np.mean(runtimes, axis=-1)
    escaped_prob =  np.mean(escaped, axis=-1)
    conv_iters_mean = np.mean(conv_iters, axis=-1)
    conv_iters_best = np.min(conv_iters, axis=-1)

    learning_rates_xticks = [str(x) for x in learning_rates]
    batch_sizes_yticks = [str(y) for y in batch_sizes]

    print('Plotting Heatmaps')
    # Plot heatmaps
    fig = plt.figure(figsize=(16, 24))
    subfigs = fig.subfigures(nrows=3, ncols=2)

    # Mean losses
    subfig = subfigs[0, 0]
    axs = subfig.subplots(nrows=2, ncols=2)
    subfig.suptitle(f'Mean Loss Over {n_trials} Trials')
    subfig.supxlabel('Learning Rates')
    subfig.supylabel('Batch Sizes')
    for i, sigma in enumerate(noise_scales):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        ax.set_title(f'Noise Level = {sigma:.2f}')
        sns.heatmap(
            losses_mean[:, :, i], annot=True, cbar=True,
            xticklabels=learning_rates_xticks, yticklabels=batch_sizes_yticks, ax=ax
        )

    # Best loss
    subfig = subfigs[0, 1]
    axs = subfig.subplots(nrows=2, ncols=2)
    subfig.suptitle(f'Best Loss Over {n_trials} Trials')
    subfig.supxlabel('Learning Rates')
    subfig.supylabel('Batch Sizes')
    for i, sigma in enumerate(noise_scales):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        ax.set_title(f'Noise Level = {sigma:.2f}')
        sns.heatmap(
            losses_best[:, :, i], annot=True, cbar=True,
            xticklabels=learning_rates_xticks, yticklabels=batch_sizes_yticks, ax=ax
        )

    # Escape prob
    subfig = subfigs[1, 0]
    axs = subfig.subplots(nrows=2, ncols=2)
    subfig.suptitle(f'Escape Probability Over {n_trials} Trials')
    subfig.supxlabel('Learning Rates')
    subfig.supylabel('Batch Sizes')
    for i, sigma in enumerate(noise_scales):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        ax.set_title(f'Noise Level = {sigma:.2f}')
        sns.heatmap(
            escaped_prob[:, :, i], annot=True, cbar=True,
            xticklabels=learning_rates_xticks, yticklabels=batch_sizes_yticks, ax=ax
        )

    # Runtimes
    subfig = subfigs[1, 1]
    axs = subfig.subplots(nrows=2, ncols=2)
    subfig.suptitle(f'Mean Runtime (seconds) Over {n_trials} Trials')
    subfig.supxlabel('Learning Rates')
    subfig.supylabel('Batch Sizes')
    for i, sigma in enumerate(noise_scales):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        ax.set_title(f'Noise Level = {sigma:.2f}')
        sns.heatmap(
            runtimes_mean[:, :, i], annot=True, cbar=True,
            xticklabels=learning_rates_xticks, yticklabels=batch_sizes_yticks, ax=ax
        )

    # Mean Convergence Iteration
    subfig = subfigs[2, 0]
    axs = subfig.subplots(nrows=2, ncols=2)
    subfig.suptitle(f'Mean Convergence Iters Over {n_trials} Trials')
    subfig.supxlabel('Learning Rates')
    subfig.supylabel('Batch Sizes')
    for i, sigma in enumerate(noise_scales):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        ax.set_title(f'Noise Level = {sigma:.2f}')
        sns.heatmap(
            conv_iters_mean[:, :, i], annot=True, cbar=True,
            xticklabels=learning_rates_xticks, yticklabels=batch_sizes_yticks, ax=ax
        )

    # Mean Convergence Iteration
    subfig = subfigs[2, 1]
    axs = subfig.subplots(nrows=2, ncols=2)
    subfig.suptitle(f'Best Convergence Iters Over {n_trials} Trials')
    subfig.supxlabel('Learning Rates')
    subfig.supylabel('Batch Sizes')
    for i, sigma in enumerate(noise_scales):
        r, c = divmod(i, 2)
        ax = axs[r, c]
        ax.set_title(f'Noise Level = {sigma:.2f}')
        sns.heatmap(
            conv_iters_best[:, :, i], annot=True, cbar=True,
            xticklabels=learning_rates_xticks, yticklabels=batch_sizes_yticks, ax=ax
        )

    print(f'Saving {lossname}_heatmaps.png')
    fig.savefig(f'{lossname}_heatmaps.png')


## Part C: Design Your Own Loss Landscape
## Create a more complex loss function with 3+ minima:
def multi_modal_loss(w: ndarray) -> float:
    r'''Custom loss landscape with multiple local minima of different qualities.

    Include saddle points for extra challenge!

    Loss landscape with respect to the weights(!) w = [w1, w2].

    Implement the following:
    \begin{align*}
        L_\text{local1}(w) &= -2 \exp \left\{ \frac{- \| w - [1.5, 1.5] \|^2}{0.2} \right\} \\
        L_\text{local2}(w) &= -1 \exp \left\{ \frac{- \| w - [1.5, -1.5] \|^2}{0.2} \right\} \\
        L_\text{local3}(w) &= -2 \exp \left\{ \frac{- \| w - [2.0, -2.0] \|^2}{0.2} \right\} \\
        L_\text{global}(w) &= -3.5 \exp \left\{ \frac{- \| w + [1.5, 1.5] \|^2}{1.5} \right\} \\
        L_\text{saddle}(w) &= 0.9 * (w_1^2 - w_2^2) * \exp \left\{ \frac{- \| w \|^2}{0.6} \right\}
    \end{align*}
    where
    \[
        L(w) = L_\text{global}(w) + L_\text{saddle}(w) + \sum_{i=1}^3 L_\text{local i}(w)
    \]

    Args:
        w: ndarray of shape (2,) of floats for the current parameters

    Returns:
        L: total loss
    '''
    # 
    L_local1 = -2 * np.exp(-np.linalg.norm(w - np.array([1.5, 1.5]))**2 / 0.2)
    L_local2 = -1 * np.exp(-np.linalg.norm(w - np.array([1.5, -1.5]))**2 / 0.2)
    L_local3 = -2 * np.exp(-np.linalg.norm(w - np.array([2.0, -2.0]))**2 / 0.2)
    L_global = -3.5 * np.exp(-np.linalg.norm(w + np.array([1.5, 1.5]))**2 / 1.5)
    L_saddle = 0.9 * (w[0]**2 - w[1]**2) * np.exp(-np.linalg.norm(w)**2 / 0.6)
    return L_local1 + L_local2 + L_local3 + L_global + L_saddle
    #raise NotImplementedError



def multi_modal_grad_components(w: ndarray) -> tuple[ndarray, ndarray]:
    '''Computes gradient of custom loss landscape.

    Instructor implementation computes grad of loss given in
    instructor version of multi_modal_loss().

    Args:
        w: ndarray of shape (2,) of floats for the current parameters

    Returns:
        grad_local: ndarray of shape (2,) of floats for local grad
        grad_global: ndarray of shape (2,) of floats for global grad
    '''
    # TODO
    c1 = np.array([1.5, 1.5])
    c2 = np.array([1.5, -1.5])
    c3 = np.array([2.0, -2.0])
    cGlobal = np.array([-1.5, -1.5])

    L_local1 = -2 * np.exp(-np.linalg.norm(w - c1)**2 / 0.2)
    L_local2 = -1 * np.exp(-np.linalg.norm(w - c2)**2 /0.2)
    L_local3 = -2 * np.exp(-np.linalg.norm(w - c3)**2 / 0.2)
    L_global = -3.5 * np.exp(-np.linalg.norm(w - cGlobal)**2 / 1.5)
    L_saddle =  0.9 * (w[0]**2 - w[1]**2) * np.exp(-np.linalg.norm(w)**2 / 0.6)
    globalMin = -3.5 * np.exp(-np.linalg.norm(w + np.array([1.5, 1.5]))**2 / 1.5)

    gradLocal1 = L_local1 * (-2) * (w - c1) / 0.2
    gradLocal2 = L_local2 * (-2) * (w - c2) / 0.2
    gradLocal3 = L_local3 * (-2) * (w - c3) / 0.2
    gradGlobal = L_global * (-2) * (w - cGlobal) / 1.5

    gradSaddle = 0.9 * np.array([2 * w[0], -2 * w[1]]) * np.exp(-np.linalg.norm(w)**2 / 0.6) + \
                 0.9 * (w[0]**2 - w[1]**2) * (-2 * w / 0.6) * np.exp(-np.linalg.norm(w)**2 / 0.6) 

    gradLocal = gradLocal1 + gradLocal2 + gradLocal3

    grad = gradLocal + gradGlobal + gradSaddle + globalMin
    return gradLocal, gradGlobal



############     Problem 4    ############
## Part 3: Implementing a Perceptron from Scratch

class SimplePerceptron:
    '''
    A simple perceptron implementation to demonstrate the linear threshold mechanism.
    This follows the classic perceptron learning algorithm from our lectures.
    '''
    def __init__(
        self,
        learning_rate: float,
        max_epochs: int,
        prng: np.random.Generator
    ) -> None:
        '''Initiailize perceptron with learning rate and max epochs.

        Args:
            learning_rate: (float) weight update learning rate
            max_epochs: (int) max number of training epochs
            prng: numpy prng Generator to initialize weights
        '''
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.prng = prng
        self.weights = None
        self.bias = None
        self.training_errors = []

    def _activation_function(self, z: ndarray) -> ndarray:
        '''Step function: returns 1 if z >= 0, else 0.

        Args:
            z: ndarray of shape (d,) of ints

        Returns:
            step_z: ndarray of shape(d,) of ints
        '''
        return np.where(z >= 0, 1, 0)

    def fit(self, X: ndarray, y: ndarray) -> None:
        '''Train the perceptron using the classic perceptron learning rule.

        Rule:
        w = w + eta * (target - prediction) * input
        for learning rate eta.

        Appends error at each training step to list of training errors.
        Early stops training if no errors found during current training epoch.

        Replace any None values specified by TODO comments

        Args:
            X: training samples. ndarray of shape (n, d)
            y: training labels. ndarray of shape (n,)
        '''
        n_samples, n_features = X.shape
        #raise NotImplementedError

        # : Initialize weights from a standard normal distribution and bias to zero
        self.weights = self.prng.standard_normal(n_features)
        self.bias = 0

        # Training loop - implement the perceptron learning algorithm
        for epoch in range(self.max_epochs):
            # Number of wrong predictions in current epoch
            errors = 0

            for i in range(n_samples):
                # : Compute the linear combination (net input)
                linear_output = X[i] @ self.weights + self.bias

                # Apply step function to get prediction
                prediction = self._activation_function(linear_output)

                # : Calculate the error and update rule
                error = y[i] - prediction

                # Only update weights if there's an error (classic perceptron rule)
                if error != 0:
                    errors += 1

                    # : Apply perceptron update rule
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error

            self.training_errors.append(errors)

            # If no errors in this epoch, we've converged (for linearly separable data)
            if errors == 0:
                print(f'Converged after {epoch + 1} epochs!')
                break
        else:
            print(f'Did not converge after {self.max_epochs} epochs - likely not linearly separable!')

    def predict(self, X: ndarray) -> ndarray:
        '''Make predictions using the learned decision boundary

        Prediction rule is linear combination

        y = activation(X . w + b)
        where . is the dot product symbol.

        Note that self.weights is None if fit() not called, so
        this function would raise an error when computing linear_output.

        Args:
            X: ndarray of testing samples of shape (n, d)

        Returns:
            y: ndarray of predicted labels of shape (n,)
        '''
        #
        return self._activation_function(X @ self.weights + self.bias)  
        

    def get_decision_boundary_params(self) -> dict[str, Any] | None:
        '''
        Get parameters for plotting the decision boundary line.
        Decision boundary: w1*x1 + w2*x2 + b = 0

        This method handles an edge case where the slope is a vertical line.

        Returns:
            dict with keys:
            - 'type': 'slope_intercept' or 'vertical'
            - 'slope', 'intercept': if type is 'slope_intercept'
            - 'x': if type is 'vertical' (the x-coordinate of vertical line)
        '''
        if len(self.weights) != 2:  # type: ignore
            return None

        w1, w2 = self.weights  # type: ignore

        if w2 == 0:
            # Vertical line: x1 = -b/w1
            if w1 != 0:
                return {'type': 'vertical', 'x': -self.bias / w1}  # type: ignore
            else:
                return None  # degenerate case: no valid boundary
        else:
            # Standard form: x2 = -(w1*x1 + b) / w2
            slope = -w1 / w2
            intercept = -self.bias / w2  # type: ignore
            return {'type': 'slope_intercept', 'slope': slope, 'intercept': intercept}


## Perceptron XOR Dataset Functions
def create_xor_dataset() -> tuple[ndarray, ndarray]:
    '''
    Create the classic XOR dataset that demonstrates linear non-separability.
    This is the problem that historically showed perceptron limitations.

    Returns:
        X: ndarray of shape (4, 2) of ints
        y: ndarray of shape (4,) of ints
    '''
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])

    return X, y


def create_nonlinear_features(X: ndarray) -> ndarray:
    '''Enhance X with non-linear features.

    Transforms the XOR problem into a linearly separable one by adding
    non-linear features.

    This is they key insight: we need to transform the problem space!

    One approach is to add x1 XOR x2 directly (but that's cheating!).
    Another approach is to add the product x1 * x2 as a new feature (aka x1 AND x2).

    This function should add the product feature:
    X_enhanced = [x1, x2, x1 * x2]

    Args:
        X: XOR dataset of shape (n, d)

    Returns:
        X_enhanced: Augmented XOR dataset of shape (n, d+1)
    '''
    #
    return np.concatenate((X, X[:, [0]] * X[:, [1]]), axis=1)
    


def plot_xor_data(X: ndarray, y: ndarray) -> None:
    '''
    Visualize the XOR problem to show why it's non-linearly separable.
    This plot clearly shows the geometric impossibility for a perceptron.

    Args:
        X: ndarray of shape (4, 2) of ints
        y: ndarray of shape (4,) of ints
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    class_0_mask = (y == 0)
    class_1_mask = (y == 1)
    ax.scatter(
        X[class_0_mask, 0],
        X[class_0_mask, 1],
        c='red',
        marker='o',
        s=200,
        label='XOR = 0',
        edgecolor='black',
        linewidth=2
    )
    ax.scatter(
        X[class_1_mask, 0],
        X[class_1_mask, 1],
        c='blue',
        marker='s',
        s=200,
        label='XOR = 1',
        edgecolor='black',
        linewidth=2
    )
    ax.set_xlabel(r'Input $x_1$', fontsize=14)
    ax.set_ylabel(r'Input $x_2$', fontsize=14)
    ax.set_title('XOR Problem: Why Perceptrons Fail\nCan you draw a single straight line to separate red and blue points?', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((-0.5, 1.5))

    # Add text annotations for each point showing the XOR computation
    for i in range(len(X)):
        ax.annotate(
            rf'$({X[i,0]}, {X[i,1]}) \rightarrow {y[i]}$',
            (X[i,0], X[i,1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    fig.tight_layout()
    print('Saving xor_plot.png')
    fig.savefig('xor_plot.png')


def visualize_decision_boundary(
    X: ndarray,
    y: ndarray,
    perceptron: SimplePerceptron
) -> None:
    '''Create a plot visualizing the learned decision boundary.

    Args:
        X: data samples of shape (n, d)
        y: data labels of shape (n,)
        perceptron: fitted simple perceptron
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()

    class_0_mask = (y == 0)
    class_1_mask = (y == 1)
    ax.scatter(X[class_0_mask, 0], X[class_0_mask, 1], c='red', marker='o', s=200,
                label='XOR = 0', edgecolor='black', linewidth=2)
    ax.scatter(X[class_1_mask, 0], X[class_1_mask, 1], c='blue', marker='s', s=200,
                label='XOR = 1', edgecolor='black', linewidth=2)

    params = perceptron.get_decision_boundary_params()
    if params is None:
        print('No valid boundary')
    elif params['type'] == 'vertical':
        ax.axvline(x=params['x'], color='k', linestyle='--', label='Decision Boundary', alpha=0.8)
        ax.fill_betweenx([-0.5, 1.5], -0.5, params['x'], alpha=0.1, color='blue', label='Predicted: Class 1')
        ax.fill_betweenx([-0.5, 1.5], params['x'], 1.5, alpha=0.1, color='red', label='Predicted: Class 0')
    elif params['type'] == 'slope_intercept':
        x_line = np.linspace(-0.5, 1.5, 100)
        y_line = params['slope'] * x_line + params['intercept']
        ax.plot(x_line, y_line, 'k--', linewidth=3, label='Decision Boundary', alpha=0.8)
        slope_pos = params['slope'] > 0
        if slope_pos:
            ax.fill_between(x_line, y_line, 2, alpha=0.1, color='blue', label='Predicted: Class 1')
            ax.fill_between(x_line, y_line, -1, alpha=0.1, color='red', label='Predicted: Class 0')
        else:
            ax.fill_between(x_line, y_line, -1, alpha=0.1, color='blue', label='Predicted: Class 1')
            ax.fill_between(x_line, y_line, 2, alpha=0.1, color='red', label='Predicted: Class 0')

    # Show predictions as text annotations
    predictions = perceptron.predict(X)
    for i in range(len(X)):
        color = 'green' if predictions[i] == y[i] else 'red'
        marker = 'o' if predictions[i] == y[i] else 'x'
        plt.annotate(f'Pred: {predictions[i]} {marker}',
                    (X[i,0], X[i,1]),
                    xytext=(5, -25), textcoords='offset points',
                    color=color, fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Input $x_1$', fontsize=14)
    ax.set_ylabel(r'Input $x_2$', fontsize=14)
    title = 'Perceptron Failure on XOR Problem'
    ax.set_title(title + '\nNotice: The straight line cannot separate the classes correctly!', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    fig.tight_layout()
    print('Saving decision_boundary.png')
    fig.savefig('decision_boundary.png')