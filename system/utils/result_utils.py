import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, dp_parameter=1, P = 2e-3, r=1):
    test_acc, train_loss = get_all_results_for_one_algo(algorithm, dataset, goal, times, dp_parameter, P, r)

    max_accuracy = []
    
    final_round_acc = []
    final_round_loss = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

        final_round_acc.append(test_acc[i][-1])
        final_round_loss.append(train_loss[i][-1])

    acc_conf_int = np.percentile(final_round_acc, [2.5, 97.5]) 
    acc_lb = acc_conf_int[0]
    acc_ub = acc_conf_int[1]

    loss_conf_int = np.percentile(final_round_loss, [2.5, 97.5]) 
    loss_lb = loss_conf_int[0]
    loss_ub = loss_conf_int[1]

    results_dir = "../results_for_monte"
    os.makedirs(results_dir, exist_ok=True)
    summary_filename = f"summary_{dataset}_{algorithm}_epsilon_{dp_parameter}_power_{P}_r_{r}.txt"
    full_path = os.path.join(results_dir, summary_filename)

    with open(full_path, 'a') as f:
        f.write(f"--- Results for {algorithm} on {dataset} (epsilon={dp_parameter}, power={P}) ---\n")
        f.write("\n") 
        f.write(f"mean for final-round accuracy: {np.mean(final_round_acc)}\n")
        f.write(f"Acc: 95% interval (lower bound): {acc_lb}\n")
        f.write(f"Acc: 95% interval (upper bound): {acc_ub}\n")
        f.write("\n") 
        f.write(f"mean for final-round loss: {np.mean(final_round_loss)}\n")
        f.write(f"Loss: 95% interval (lower bound): {loss_lb}\n")
        f.write(f"Loss: 95% interval (upper bound): {loss_ub}\n")

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, dp_parameter=1, P = 2e-3, r=1):
    test_acc = []
    train_loss = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + "epsilon_" + str(dp_parameter) + "_" + "Power_" + str(P) + "_" + "r_" + str(r)  + "_" + goal + "_" + str(i)
        train_loss.append(np.array(read_data_then_delete(file_name, delete=False, flag="loss")))
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False, flag="acc")))

    return test_acc, train_loss


def read_data_then_delete(file_name, delete=False, flag = "acc"):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
        rs_train_loss = np.array(hf.get('rs_train_loss'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    if flag == "acc":
        return rs_test_acc
    elif flag == "loss":
        return rs_train_loss