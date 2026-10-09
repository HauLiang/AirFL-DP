import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import re


# Load final-round accuracy and loss from summary files
def get_results(dataset="FashionMNIST", algorithm="FedAvg", epsilon=0.1, c=12, r=0.9):

    base_path = "./results_fig7/"
    base_filename = f"summary_{dataset}_{algorithm}_epsilon_{epsilon}_clip_{c}_r_{str(r)}"

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

    # Clipping factor settings
    c_list = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Load baseline results
    FedAvg_acc, FedAvg_loss = get_results(algorithm="FedAvg")

    ax.plot(np.array(c_list) * 1e-3, np.full_like(c_list, FedAvg_acc), linestyle='-', color='k', linewidth=1.5)

    # Load results under different clipping factors
    DP_acc = []
    DP_loss = []
    ZF_acc = []
    ZF_loss = []
    Fed_Clip_acc = []
    Fed_Clip_loss = []

    for i in range(len(c_list)):
        DP_acc_i, DP_loss_i = get_results(algorithm="AirFL_DP", c=c_list[i])
        DP_acc.append(DP_acc_i)
        DP_loss.append(DP_loss_i)

        ZF_acc_i, ZF_loss_i = get_results(algorithm="AirFL_MIMO", c=c_list[i])
        ZF_acc.append(ZF_acc_i)
        ZF_loss.append(ZF_loss_i)

        Fed_Clip_acc_i, Fed_Clip_loss_i = get_results(algorithm="FedAvg_Clip", c=c_list[i])
        Fed_Clip_acc.append(Fed_Clip_acc_i)
        Fed_Clip_loss.append(Fed_Clip_loss_i)

    # Plot test accuracies
    ax.plot(np.array(c_list) * 1e-3, DP_acc, linestyle='-.', color='k', linewidth=1.5)
    ax.plot(np.array(c_list) * 1e-3, ZF_acc, '--', color='k', markevery=1, marker='o', markerfacecolor='w', zorder=-1, linewidth=1.5)
    ax.plot(np.array(c_list) * 1e-3, Fed_Clip_acc, linestyle='--', color='k', markevery=1, marker='x', markerfacecolor='w', linewidth=1.5)

    # Axis settings
    ax.set_xlabel(r'Clipping factor $\tilde{c}$')
    ax.set_ylabel('Test accuracy')
    ax.grid(True)
    fig.tight_layout()

    # Curve annotations
    _ = ax.annotate(r'AirFL-MIMO',
                    xy=(0.007, 0.8739), xycoords='data',
                    xytext=(-44.5, -46), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'FL w/ clipping',
                    xy=(0.005, 0.8775), xycoords='data',
                    xytext=(-48, 38), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'AirFL-DP',
                    xy=(0.005, 0.8648), xycoords='data',
                    xytext=(-32.5, -49), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    _ = ax.annotate(r'Vanilla FL',
                    xy=(0.005, 0.895), xycoords='data',
                    xytext=(-34, -45), textcoords='offset points', fontsize=16,
                    arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.5, connectionstyle="arc3,rad=0"), zorder=10)

    # Save and display the figure
    plt.savefig('figure_7.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()