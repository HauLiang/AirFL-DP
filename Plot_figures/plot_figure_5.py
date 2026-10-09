import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from decimal import Decimal
import re


# Load final-round accuracy and loss from summary files
def get_results(dataset="FashionMNIST", algorithm="FedAvg", epsilon=0.1, power=2e-3, r=0.9):

    base_path = "./results_fig5/"
    base_filename = f"summary_{dataset}_{algorithm}_epsilon_{epsilon}_Power_{str(Decimal(str(power)))}_r_{str(r)}"

    file_path = os.path.join(base_path, f"{base_filename}.txt")

    with open(file_path, 'r') as f:
        content = f.read()

        # Extract accuracy
        match = re.search(r'mean for final-round accuracy: ([\d.]+)', content)
        acc = float(match.group(1))

        # Extract loss
        match = re.search(r'mean for final-round loss: ([\d.]+)', content)
        loss = float(match.group(1))

    return acc, loss


if __name__ == '__main__':

    # Font settings
    # plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    font = {'family': 'Times New Roman',
            # 'weight': 'bold',
            'size': 16}

    mpl.rc('font', **font)

    fig, ax = plt.subplots()

    # Privacy budget settings
    # epsilon_list = [0.01, 0.0175, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25]
    epsilon_list = [0.01, 0.0175, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28]

    # Load baseline results
    FedAvg_acc, FedAvg_loss = get_results(algorithm="FedAvg")
    clipFedAvg_acc, clipFedAvg_loss = get_results(algorithm="FedAvg_Clip")
    ZF_acc, ZF_loss = get_results(algorithm="AirFL_ZF")

    # Plot baseline accuracies
    ax.plot(epsilon_list, np.full_like(epsilon_list, FedAvg_acc), '-', color='k', linewidth=1.5)
    ax.plot(epsilon_list, np.full_like(epsilon_list, clipFedAvg_acc), linestyle='none', color='k', markevery=1, marker='x', markerfacecolor='w', linewidth=1.5)
    ax.plot(epsilon_list, np.full_like(epsilon_list, ZF_acc), '--', color='k', markevery=1, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)

    # Load AirFL-DP results under different privacy budgets
    DP_acc = []
    DP_loss = []
    for i in range(len(epsilon_list)):
        DP_acc_i, DP_loss_i = get_results(algorithm="AirFL_DP", epsilon=epsilon_list[i])
        DP_acc.append(DP_acc_i)
        DP_loss.append(DP_loss_i)

    ax.plot(epsilon_list, DP_acc, linestyle='-.', color='k', linewidth=1.5)

    # Axis settings
    ax.set_xlabel(r'Privacy budget $\tilde{\epsilon}$')
    ax.set_ylabel('Test accuracy')
    ax.grid(True)
    ax.set_ylim(0.64, 0.91)
    fig.tight_layout()

    # Zoomed-in inset
    axins_2 = ax.inset_axes([0.54, 0.28, 0.43, 0.3])
    axins_2.plot(epsilon_list, np.full_like(epsilon_list, FedAvg_acc), '-', color='k', linewidth=1.5)
    axins_2.plot(epsilon_list, np.full_like(epsilon_list, clipFedAvg_acc), linestyle='none', color='k', markevery=1, marker='x', markerfacecolor='w', linewidth=1.5)
    axins_2.plot(epsilon_list, np.full_like(epsilon_list, ZF_acc), '--', color='k', markevery=1, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)
    axins_2.plot(epsilon_list, DP_acc, linestyle='-.', color='k', linewidth=1.5)

    # Inset axis limits
    x1, x2 = 0.15, 0.24
    y1, y2 = 0.87, 0.9
    axins_2.set_xlim(x1, x2)
    axins_2.set_ylim(y1, y2)
    axins_2.set_xticklabels('')
    axins_2.set_yticklabels('')
    axins_2.set_xticks([])
    axins_2.set_yticks([])
    # axins.grid(True, linestyle=':', alpha=0.6)

    ax.indicate_inset_zoom(axins_2, edgecolor="black", lw=1)

    # Curve annotations
    _ = ax.annotate(r'AirFL-MIMO',
                    xy=(0.215, 0.771), xycoords='data',
                    xytext=(-188, -50), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'FL w/ clipping',
                    xy=(0.1735, 0.772), xycoords='data',
                    xytext=(-135, 11), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'AirFL-DP',
                    xy=(0.22, 0.737), xycoords='data',
                    xytext=(-32.5, -60), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'Vanilla FL',
                    xy=(0.22, 0.782), xycoords='data',
                    xytext=(-34.5, 45), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    # Save and display the figure
    plt.savefig('figure_5.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()