import time
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data


class AirFL_DP_online(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):

        # For New-clients Accuracy figure
        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)

        # optimization
        sigma_est = 0.5
        print("Start optimization!")
        w_list, pi_list, sigma_uplink2 = self.solve_dp_online(sigma_est=sigma_est)
        print("Finish optimization!")

        print(f"pi_list: {pi_list}")
        print(f"w_list: {w_list}")
        print(f"ratio: {np.array(w_list)/np.array(pi_list)}")

        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                # Default
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_diffs_over_the_air()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # DP optimization
            w_norm = w_list[i]
            pi_norm = pi_list[i]
            self.aggregate_diffs_over_the_air_dp(w_norm, pi_norm, sigma_uplink2)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Default
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_results_online(sigma=sigma_est)
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    # # For trials of changing number of training clients / or for test accuracy experiments
    # def set_clients(self, clientObj):
    #     num_ids = [i for i in range(20)]
    #     self.chosen_ids = np.random.choice(num_ids, size=self.num_clients, replace=False)
    #     self.chosen_ids.sort()
    #     for i, train_slow, send_slow in zip(self.chosen_ids, self.train_slow_clients, self.send_slow_clients):
    #         train_data = read_client_data(self.dataset, i, is_train=True)
    #         test_data = read_client_data(self.dataset, i, is_train=False)
    #         client = clientObj(self.args,
    #                         id=i,
    #                         train_samples=len(train_data),
    #                         test_samples=len(test_data),
    #                         train_slow=train_slow,
    #                         send_slow=send_slow)
    #         self.clients.append(client)
    #
    # # For trials of changing number of training clients / or for test accuracy experiments
    # def set_new_clients(self, clientObj):
    #     num_ids = [i for i in range(20)]
    #     rest_ids = np.setdiff1d(num_ids, self.chosen_ids)
    #     rest_ids.sort()
    #     for i in rest_ids:
    #         train_data = read_client_data(self.dataset, i, is_train=True)
    #         test_data = read_client_data(self.dataset, i, is_train=False)
    #         client = clientObj(self.args,
    #                         id=i,
    #                         train_samples=len(train_data),
    #                         test_samples=len(test_data),
    #                         train_slow=False,
    #                         send_slow=False)
    #         self.new_clients.append(client)
