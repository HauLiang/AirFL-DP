import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import re


# Load final-round accuracy and loss from summary files
def get_results(dataset="FashionMNIST", algorithm="FedAvg", epsilon=20.0, D=10, r=0.9):

    base_path = "./results_fig8/"
    base_filename = f"summary_{dataset}_{algorithm}_epsilon_{epsilon}_D_{D}_r_{str(r)}"

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

    # Bounded domain diameter settings
    D_list = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]

    # Load AirFL-DP results under different domain diameters
    DP_acc = []
    DP_loss = []
    ZF_acc = []
    ZF_loss = []

    for i in range(len(D_list)):
        DP_acc_i, DP_loss_i = get_results(algorithm="AirFL_DP_with_D", D=D_list[i])
        DP_acc.append(DP_acc_i)
        DP_loss.append(DP_loss_i)

    # Plot test accuracy
    ax.plot(D_list, DP_acc, linestyle='-.', color='k', linewidth=1.5)

    # Axis settings
    ax.set_xlabel(r'Bounded domain diameter $D$')
    ax.set_ylabel('Test accuracy')
    ax.grid(True)

    fig.tight_layout()

    # Transition point between the two regimes
    D_split = 40

    # Shade the (P3b)-active and (P3a)-active regions
    ax.axvspan(5, D_split, color='#DCEEFF', alpha=0.7, zorder=0)  # Light blue
    ax.axvspan(D_split, 80, color='#FFF2CC', alpha=0.7, zorder=0)  # Light yellow

    # Mark the transition point
    ax.axvline(D_split, color='darkred', linestyle='--', linewidth=1.5)

    # Region annotations
    ax.text(22.43, 0.828, '(P3b)-active region', fontsize=16, color='darkblue', ha='center')
    ax.text(62.3, 0.886, '(P3a)-active region', fontsize=16, color='#7A5200', ha='center')
    ax.text(D_split + 0.8, 0.845, 'transition', fontsize=15, color='darkred', rotation=90, va='bottom')

    # Save and display the figure
    plt.savefig('figure_8.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.show()