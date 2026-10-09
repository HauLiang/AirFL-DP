import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import re


# Load final-round accuracy and loss from summary files
def get_results(dataset="FashionMNIST", algorithm="FedAvg", epsilon=0.1, power=2e-3, r=0.9):

    base_path = "./results_fig6/"
    base_filename = f"summary_{dataset}_{algorithm}_epsilon_{epsilon}_Power_{power}_r_{str(r)}"

    # File path
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

    # Transmit power settings
    power_list = [2e-11, 7e-11, 2e-10, 7e-10, 2e-9, 7e-9, 2e-8, 7e-8, 2e-7]

    # Calculate uplink SNR
    sigma2 = 10 ** (-100 / 10) * 1e-3
    c_light = 3e8
    f_c = 2.4e9
    r = 1000
    path_loss = (c_light / (4 * np.pi * f_c * r)) ** 2
    sigma_uplink2 = sigma2 / path_loss

    snr_list = 10 * np.log10(np.array(power_list) / sigma_uplink2)

    # Load baseline results
    FedAvg_acc, FedAvg_loss = get_results(algorithm="FedAvg", power=0.002)
    clipFedAvg_acc, clipFedAvg_loss = get_results(algorithm="FedAvg_Clip", power=0.002)

    # Load AirFL results under different transmit powers
    ZF_acc = []
    ZF_loss = []
    DP_acc = []
    DP_loss = []

    for i in range(len(power_list)):
        ZF_acc_i, ZF_loss_i = get_results(algorithm="AirFL_ZF", power=power_list[i])
        DP_acc_i, DP_loss_i = get_results(algorithm="AirFL_DP", power=power_list[i])
        ZF_acc.append(ZF_acc_i)
        ZF_loss.append(ZF_loss_i)
        DP_acc.append(DP_acc_i)
        DP_loss.append(DP_loss_i)

    # Plot test accuracies
    ax.plot(snr_list, np.full_like(power_list, FedAvg_acc), linestyle='-', color='k', linewidth=1.5)
    ax.plot(snr_list, np.full_like(power_list, clipFedAvg_acc), linestyle='--', color='k', markevery=1, marker='x', markerfacecolor='w', linewidth=1.5)

    ax.plot(snr_list, ZF_acc, '--', color='k', markevery=1, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)
    ax.plot(snr_list, DP_acc, linestyle='-.', color='k', linewidth=1.5)

    # Axis settings
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('Test accuracy')
    ax.grid(True)
    fig.tight_layout()

    # Zoomed-in inset
    axins = ax.inset_axes([0.2, 0.27, 0.5, 0.35])
    axins.plot(snr_list, np.full_like(power_list, FedAvg_acc), linestyle='-', color='k', linewidth=1.5)
    axins.plot(snr_list, np.full_like(power_list, clipFedAvg_acc), linestyle='--', color='k', markevery=1, marker='x', markerfacecolor='w', linewidth=1.5)
    axins.plot(snr_list, ZF_acc, '--', color='k', markevery=1, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)
    axins.plot(snr_list, DP_acc, linestyle='-.', color='k', linewidth=1.5)

    # Inset axis limits
    x1, x2 = -65, -55
    y1, y2 = 0.84, 0.905
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_xticks([])
    axins.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="black", lw=1)

    # Curve annotations
    _ = ax.annotate(r'AirFL-MIMO',
                    xy=(-53, 0.732), xycoords='data',
                    xytext=(53, -50), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'FL w/ clipping',
                    xy=(-53, 0.741), xycoords='data',
                    xytext=(40, 51), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'AirFL-DP',
                    xy=(-59, 0.673), xycoords='data',
                    xytext=(-32.5, -68), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'Vanilla FL',
                    xy=(-59, 0.748), xycoords='data',
                    xytext=(-34.5, 46), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    # Save and display the figure
    plt.savefig('figure_6.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()