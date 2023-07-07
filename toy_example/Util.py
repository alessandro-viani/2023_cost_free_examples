import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


def log_normal(x, mean, std):
    return -np.log(np.sqrt(2 * np.pi) * std) - 0.5 * np.square((x - mean) / std)


def sequence_of_exponents(n_iter, max_exp):
    return np.concatenate((np.power(np.linspace(0, 1, n_iter), 4), [max_exp + 0.1]))


def creation_data(n_data, theta):
    sourcespace = np.linspace(-5, 5, n_data)
    data = np.zeros(int(n_data))
    for i in range(0, n_data):
        data[i] = 1 * stats.norm.pdf(sourcespace[i], 0, 1) + np.random.normal(0, theta)
    return sourcespace, data


def eval_kl_div(post_clas, post_prop):
    post_clas = post_clas / np.sum(post_clas)
    post_prop = post_prop / np.sum(post_prop)
    return np.sum(post_clas * np.log(np.divide(post_clas, post_prop)))


def plot_confront(post_pm, post_fb, post_eb, theta_true=1, mean_min=-1, mean_max=1, theta_min=-0.1, theta_max=0.1, fontsize=10, linewidth=2, savefig=True):
    x_mean = np.linspace(-5, 5, 10000)
    
    post_pm.theta_posterior /= np.sum(post_pm.theta_posterior)
    fb_theta_posterior = stats.gaussian_kde(post_fb.vector_theta, weights=post_fb.vector_weight).pdf(post_pm.grid_theta)
    fb_theta_posterior /= np.sum(fb_theta_posterior)
    
    fig, ax = plt.subplots(3,1, figsize=(16,9))
    
    m = np.max([np.max(post_pm.theta_posterior), np.max(fb_theta_posterior)])
    
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0].plot(post_pm.grid_theta, post_pm.theta_posterior, color='#1f77b4', linewidth=linewidth, label='PropFB')
    ax[0].fill_between(post_pm.grid_theta, post_pm.theta_posterior, color='#1f77b4', alpha=0.125)
    ax[0].plot(post_pm.grid_theta, fb_theta_posterior, color='red', linewidth=linewidth, label='FB')
    ax[0].fill_between(post_pm.grid_theta, fb_theta_posterior, color='red', alpha=0.125)
    ax[0].vlines(theta_true, ymin=0, ymax=1.1, linestyles=':', linewidth=linewidth, colors='green')
    ax[0].set_title(r'$p(\theta\mid y)$', size=fontsize)
    ax[0].set_xlim([theta_min, theta_max])
    ax[0].set_ylim([0, 1.1*m])
    ax[0].legend(fontsize=fontsize)

    m = np.max([np.max(post_pm.mean_posterior), np.max(post_fb.mean_posterior)])

    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].plot(x_mean, post_pm.mean_posterior, color='#1f77b4', linewidth=linewidth, label='PropFB')
    ax[1].fill_between(x_mean, post_pm.mean_posterior, color='#1f77b4', alpha=0.125)
    ax[1].plot(x_mean, post_fb.mean_posterior, color='red', linewidth=linewidth, label='FB')
    ax[1].fill_between(x_mean, post_fb.mean_posterior, color='red', alpha=0.125)
    ax[1].vlines(0, ymin=0, ymax=1, linestyles=':', linewidth=linewidth, colors='green')
    ax[1].set_title(r'$p(\mu\mid y)$', size=fontsize)
    ax[1].set_xlim([mean_min, mean_max])
    ax[1].set_ylim([0, 1.1*m])
    ax[1].legend(fontsize=fontsize)

    m = np.max([np.max(post_pm.mean_eb_posterior), np.max(post_eb.mean_posterior)])

    ax[2].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[2].plot(x_mean, post_pm.mean_eb_posterior, color='green', linewidth=linewidth, label='PropEB')
    ax[2].fill_between(x_mean, post_pm.mean_eb_posterior, color='green', alpha=0.125)
    ax[2].plot(x_mean, post_eb.mean_posterior, color='darkorange', linewidth=linewidth, label='EB')
    ax[2].fill_between(x_mean, post_eb.mean_posterior, color='darkorange', alpha=0.125)
    ax[2].vlines(0, ymin=0, ymax=1, linestyles=':', linewidth=linewidth, colors='green')
    ax[2].set_title(r'$p^{\hat{\theta}_{MAP}}(\mu\mid y)$', size=fontsize)
    ax[2].set_xlim([mean_min, mean_max])
    ax[2].set_ylim([0, 1.1*m])
    ax[2].legend(fontsize=fontsize)



def plot_confront2(post_pm, post_fb, post_em, savefig=True):
    post_pm.vector_mean = np.append(post_pm.vector_mean, -5)
    post_pm.vector_mean = np.append(post_pm.vector_mean, 5)
    post_pm.vector_weight = np.append(post_pm.vector_weight, 0)
    post_pm.vector_weight = np.append(post_pm.vector_weight, 0)

    post_fb.vector_mean = np.append(post_fb.vector_mean, -5)
    post_fb.vector_mean = np.append(post_fb.vector_mean, 5)
    post_fb.vector_theta = np.append(post_fb.vector_theta, np.min(post_fb.vector_theta))
    post_fb.vector_theta = np.append(post_fb.vector_theta, np.min(post_fb.vector_theta))
    post_fb.vector_weight = np.append(post_fb.vector_weight, 0)
    post_fb.vector_weight = np.append(post_fb.vector_weight, 0)

    post_em.vector_mean = np.append(post_em.vector_mean, -5)
    post_em.vector_mean = np.append(post_em.vector_mean, 5)
    post_em.vector_weight = np.append(post_em.vector_weight, 0)
    post_em.vector_weight = np.append(post_em.vector_weight, 0)

    alpha = 0.5
    sns.set_style('darkgrid')
    color_map = ['#1f77b4', 'darkorange', 'forestgreen', 'red']

    min_theta = np.min([np.min(post_pm.vector_theta), np.min(post_fb.vector_theta), np.min(post_em.vector_theta)])
    max_theta = np.max([np.max(post_pm.vector_theta), np.max(post_fb.vector_theta), np.max(post_em.vector_theta)])

    post_fb.grid_theta = post_pm.grid_theta
    post_fb.theta_posterior = stats.gaussian_kde(post_fb.vector_theta, weights=post_fb.vector_weight).pdf(
        post_fb.grid_theta)
    integral = 0.5 * np.sum((post_fb.theta_posterior[:-1] + post_fb.theta_posterior[1:]) * np.abs(
        post_fb.grid_theta[:-1] - post_fb.grid_theta[1:]))
    post_fb.theta_posterior /= integral

    fig, ax = plt.subplots(3, 2, figsize=(15, 5))

    plt.sca(ax[0, 0])
    plt.xlim([-5, 5])
    plt.title(r'$p(\mu\mid y)$')
    plt.ylabel('Proposed Method', fontsize=10, rotation=90, labelpad=20)
    sns.histplot(x=post_pm.vector_mean, stat='probability', weights=post_pm.vector_weight, bins=post_pm.n_bins, color=color_map[0], alpha=alpha)

    plt.sca(ax[0, 1])
    plt.title(r'$p(\theta\mid y)$')
    plt.plot(post_pm.grid_theta, post_pm.theta_posterior, color=color_map[0], alpha=alpha)
    plt.xlim([min_theta, max_theta])
    plt.fill_between(post_pm.grid_theta, post_pm.theta_posterior, color=color_map[0], alpha=alpha * 0.25)

    plt.sca(ax[1, 0])
    plt.xlim([-5, 5])
    plt.ylabel('Fully Bayesian', fontsize=10, rotation=90, labelpad=20)
    sns.histplot(x=post_fb.vector_mean, stat='probability', weights=post_fb.vector_weight, bins=post_fb.n_bins, color=color_map[1], alpha=alpha)

    plt.sca(ax[1, 1])
    sns.histplot(x=post_fb.vector_theta, stat='density', weights=post_fb.vector_weight, bins=post_fb.n_bins, color=color_map[1], alpha=alpha)
    plt.plot(post_fb.grid_theta, post_fb.theta_posterior, color=color_map[1], alpha=alpha)
    plt.xlim([min_theta, max_theta])
    plt.fill_between(post_fb.grid_theta, post_fb.theta_posterior, color=color_map[1], alpha=alpha * 0.25)

    plt.sca(ax[2, 0])
    plt.xlim([-5, 5])
    plt.ylabel('EM', fontsize=10, rotation=90, labelpad=20)
    sns.histplot(x=post_em.vector_mean, stat='probability', weights=post_em.vector_weight, bins=post_em.n_bins, color=color_map[2], alpha=alpha)

    plt.sca(ax[2, 1])
    plt.plot(post_em.grid_theta, post_em.theta_posterior, color=color_map[2], alpha=alpha)
    plt.fill_between(post_em.grid_theta, post_em.theta_posterior, color=color_map[2], alpha=alpha * 0.25)
    plt.xlim([min_theta, max_theta])
    fig.tight_layout()
    if savefig:
        plt.savefig('fig/plot_confront.png', dpi=1000)
