import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from decimal import Decimal


# Load experimental results from HDF5 files
def get_results(dataset="FashionMNIST", algorithm="FedAvg", epsilon=0.1, power=2e-3, num_runs=10, r=0.9):
    base_path="./results_fig3/"
    base_filename = f"{dataset}_{algorithm}_epsilon_{epsilon}_Power_{str(Decimal(str(power)))}_r_{str(r)}_test"

    all_accuracies = []
    all_losses = []
    for i in range(num_runs):

        file_path = os.path.join(base_path, f"{base_filename}_{i}.h5")

        with h5py.File(file_path, 'r') as f:
            test_acc = f['rs_test_acc'][:]
            train_loss = f['rs_train_loss'][:]

            all_accuracies.append(test_acc)
            all_losses.append(train_loss)

    return all_accuracies, all_losses


# Load results with online channel estimation
def get_online_results(dataset="FashionMNIST", algorithm="FedAvg", epsilon=0.1, power=2e-3, num_runs=30, r=0.9, sigma=0.1):
    base_path = "./results_fig3/"
    base_filename = f"{dataset}_{algorithm}_epsilon_{epsilon}_Power_{str(Decimal(str(power)))}_r_{str(r)}_sigma_{str(sigma)}_test"

    all_accuracies = []
    all_losses = []
    for i in range(num_runs):

        file_path = os.path.join(base_path, f"{base_filename}_{i}.h5")

        with h5py.File(file_path, 'r') as f:
            test_acc = f['rs_test_acc'][:]
            train_loss = f['rs_train_loss'][:]

            all_accuracies.append(test_acc)
            all_losses.append(train_loss)

    return all_accuracies, all_losses



if __name__ == '__main__':

    # Font settings
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    font = {'family': 'Times New Roman',
            'size': 16}
    mpl.rc('font', **font)

    fig, ax = plt.subplots()
    rounds = np.arange(1, 51)

    # Load baseline results
    FedAvg_acc, FedAvg_loss = get_results(algorithm="FedAvg")
    clipFedAvg_acc, clipFedAvg_loss = get_results(algorithm="FedAvg_Clip")
    ZF_acc, ZF_loss = get_results(algorithm="AirFL_ZF")

    epsilon_list = [0.04, 0.08, 0.15]
    linestyle_list = ['-.', (0, (10, 5)), (0, (5, 5))]

    # Plot average training losses
    ax.plot(rounds, np.mean(FedAvg_loss, axis=0), '-', color='k', linewidth=1.5)
    ax.plot(rounds, np.mean(clipFedAvg_loss, axis=0), linestyle='none', color='k', markevery=3, marker='x', markerfacecolor='w', linewidth=1.5)
    ax.plot(rounds, np.mean(ZF_loss, axis=0), '--', color='k', markevery=3, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)

    # Plot AirFL-DP results under different privacy budgets
    for i in range(len(epsilon_list)):
        DP_acc_i, DP_loss_i = get_results(algorithm="AirFL_DP", epsilon=epsilon_list[i])
        ax.plot(rounds, np.mean(DP_loss_i, axis=0), linestyle=linestyle_list[i], color='k', linewidth=1.5)

    # Plot AirFL-DP with online channel estimation
    DP_acc_online, DP_loss_online = get_online_results(algorithm="AirFL_DP_online", num_runs=30, epsilon=0.08, sigma=0.5)
    ax.plot(rounds, np.mean(DP_loss_online, axis=0), linestyle=(0, (1, 1)), color='k', linewidth=1.5)

    # Axis settings
    ax.set_xlabel('Communication round $T$')
    ax.set_ylabel('Training loss')
    ax.grid(True)


    # Zoomed-in inset
    axins = ax.inset_axes([0.55, 0.52, 0.4, 0.3])
    axins.plot(rounds, np.mean(FedAvg_loss, axis=0), '-', color='k', linewidth=1.5)
    axins.plot(rounds, np.mean(clipFedAvg_loss, axis=0), linestyle='none', color='k', markevery=3, marker='x', markerfacecolor='w', linewidth=1.5)
    axins.plot(rounds, np.mean(ZF_loss, axis=0), '--', color='k', markevery=3, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)

    for i in range(len(epsilon_list)):
        DP_acc_i, DP_loss_i = get_results(algorithm="AirFL_DP", epsilon=epsilon_list[i])
        axins.plot(rounds, np.mean(DP_loss_i, axis=0), linestyle=linestyle_list[i], color='k', linewidth=1.5)

    axins.plot(rounds, np.mean(DP_loss_online, axis=0), linestyle=(0, (1, 1)), color='k', linewidth=1.5)

    # Inset axis limits
    x1, x2 = 23, 37
    y1, y2 = 0.26, 0.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_xticks([])
    axins.set_yticks([])
    # axins.grid(True, linestyle=':', alpha=0.6)

    ax.indicate_inset_zoom(axins, edgecolor="black", lw=1)

    # Curve annotations
    _ = ax.annotate(r'AirFL-DP ($\tilde{\epsilon} = 0.04$)',
                    xy=(36.5, 0.876), xycoords='data',
                    xytext=(-67, 26), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"))

    _ = ax.annotate(r'AirFL-MIMO',
                    xy=(31.3, 1.60), xycoords='data',
                    xytext=(-158, -25), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)
    _ = ax.annotate(r'FL w/ clipping',
                    xy=(36.43, 1.578), xycoords='data',
                    xytext=(-190, -63.5), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'AirFL-DP ($\tilde{\epsilon} = 0.08$)',
                    xy=(34, 1.915), xycoords='data',
                    xytext=(-195, 15), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'AirFL-DP ($\tilde{\epsilon} = 0.15$)',
                    xy=(34, 1.67), xycoords='data',
                    xytext=(-195, 6), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)
    _ = ax.annotate(r'AirFL-DP-online ($\tilde{\epsilon} = 0.08$)',
                    xy=(38, 1.84), xycoords='data',
                    xytext=(-90, 43), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'Vanilla FL',
                    xy=(5.01, 0.58), xycoords='data',
                    xytext=(-33.8, -38), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    plt.savefig('figure_3a.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

