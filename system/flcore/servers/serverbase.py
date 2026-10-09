import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
import cvxpy as cp


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.glr = args.global_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        # over-the-air para.
        self.d = len(torch.cat([param.flatten() for param in self.global_model.parameters()]))
        self.power = args.power
        self.ratio = args.join_ratio
        self.clipping_threshold = np.sqrt(2 * self.d * 2e-3 * 3)
        self.m_antenna = args.server_antenna_number
        self.c_light = 3e8
        self.f_c = 2.4e9

        # dp para.
        self.epsilon = args.dp_epsilon
        self.delta = args.dp_delta
        self.cvxpy_flag = args.cvxpy_flag
        self.dBm = args.dBm
        self.r = 1000 * np.sqrt(np.random.rand())

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # receive model difference
    def receive_diffs(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_diffs = []
        tot_samples = 0

        # Simulate commun.
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.uploaded_models.append(client.model)

                # calculate the model difference
                diff = self.calculate_model_difference(self.global_model, client.model)

                # Divided by learning rate
                for param in diff.parameters():
                    param.data = param.data / self.learning_rate

                self.uploaded_diffs.append(diff)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # get model difference
    def calculate_model_difference(self, global_model, client_model):
        diff_model = copy.deepcopy(global_model)
        for diff_param, global_param, client_param in zip(diff_model.parameters(), global_model.parameters(),
                                                          client_model.parameters()):
            diff_param.data = client_param.data - global_param.data
        return diff_model
    
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    # global update by using model difference
    def aggregate_diffs(self):
        assert (len(self.uploaded_diffs) > 0)
       
        for w, diff in zip(self.uploaded_weights, self.uploaded_diffs):
            self.update_global_model(w, diff)

    # update global model by using model difference
    def update_global_model(self, w, diff):
        # For model update using model difference
        for global_param, diff_param in zip(self.global_model.parameters(), diff.parameters()):
            global_param.data += diff_param.data * w * self.glr * self.learning_rate

    def add_parameters(self, w, client_model):
        # for upload model para.
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            print(self.ratio)
            algo = algo + "_" + "epsilon_" + str(self.epsilon) + "_" +  "Power_" + str(self.power) + "_" + "r_" + str(self.ratio)  + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_results_online(self, sigma):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            print(self.ratio)
            algo = algo + "_" + "epsilon_" + str(self.epsilon) + "_" +  "Power_" + str(self.power) + "_" + "r_" + str(self.ratio)  + "_" + "sigma_" + str(sigma)  + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                
    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    # gradient clipping
    def receive_diffs_over_the_air(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_diffs = []
        tot_samples = 0

        # Comm. channel
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost<= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)

                # calculate the model difference
                diff = self.calculate_model_difference(self.global_model, client.model)

                # Divided by learning rate
                for param in diff.parameters():
                    param.data = param.data / self.learning_rate

                self.uploaded_diffs.append(diff)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        # scaling + clipping
        # Transmit a scaled and clipped version of model diffs  
        for client_id, diff in zip(self.uploaded_ids, self.uploaded_diffs):
            
            # sample-wise clipping! 
            # calculate clipping factor
            client_norm = torch.norm(torch.cat([param.flatten() for param in diff.parameters()]))
            # print(f"client norm: {client_norm}")
            # print(f"clipping threshold: {self.clipping_threshold}")

            if client_norm > self.clipping_threshold:
                clipping_factor = self.clipping_threshold / client_norm
                print(f"clipping_fatctor: {clipping_factor}")
            else:
                clipping_factor = 1.0

            # gradient clipping (tbc.)
            for param in diff.parameters():
                param.data = clipping_factor * param.data


    def aggregate_diffs_over_the_air(self):
        assert (len(self.uploaded_diffs) > 0)

        # Sum of clipped model difference
        sum_diffs = copy.deepcopy(self.uploaded_diffs[0])

        for param in sum_diffs.parameters():
            param.data.zero_()

        # aggregation
        for diff in self.uploaded_diffs:
            for sum_param, diff_param in zip(sum_diffs.parameters(), diff.parameters()):
                sum_param.data += diff_param.data

        # flattened diff vector
        sum_diffs_flattened = torch.cat([param.flatten() for param in sum_diffs.parameters()])
        x = sum_diffs_flattened.detach().cpu().numpy()

        # channel noise
        # print(f"sdadsasdasds d value:{self.d}")
        self.sigma2 = 10 ** (self.dBm / 10) * 1e-3   
        path_loss = (self.c_light / (4 * np.pi * self.f_c * self.r) ) ** 2   
        sigma_uplink2 = self.sigma2 / path_loss  

        # ZF eqv. noise
        gamma = self.clipping_threshold/np.sqrt(self.d * self.power)
        w_norm = self.zero_forcing_beamforming(self.m_antenna, self.num_clients, gamma)
        N0 = sigma_uplink2 * (w_norm ** 2)
   
        # SNR
        # SNR_origin = 10*np.log10(self.num_clients*self.power/sigma_uplink2)
        # SNR_ZF = 10*np.log10(self.num_clients*self.power/N0)

        # # print(f"SNR (origin): {SNR_origin}")
        # # print(f"SNR (after ZF): {SNR_ZF}")

        # noise scale: eqv. channel noiseS
        noise_scale = 1 / 2 * N0
        noise = np.sqrt(noise_scale) * np.random.randn(self.d)
        
        # channel noise
        x_hat = x + noise

        SNR = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(noise) )
        print(f"SNR (practical): {SNR}")
        
        sum_diffs_hat_flattened = torch.from_numpy(x_hat).to(self.device)
        # Reshape the flattened tensor back into the model parameters
        start = 0
        for param in sum_diffs.parameters():
            end = start + param.numel()
            param.data = sum_diffs_hat_flattened[start:end].reshape(param.shape)
            start = end

        # Update global model
        for global_param, diff_param in zip(self.global_model.parameters(), sum_diffs.parameters()):
            global_param.data += self.glr * self.learning_rate / len(self.uploaded_diffs) * diff_param.data

    # zero forcing method
    def zero_forcing_beamforming(self, m, n, gamma):

        assert (m >= n)

        # simulate Rayleigh channel
        real_part = np.random.randn(m, n) / np.sqrt(2)
        imag_part = np.random.randn(m, n) / np.sqrt(2)

        # preliminary
        H = real_part + 1j * imag_part
        HTH = H.conj().T @ H
        ones_vector = np.ones((n, 1))

        try:
            inv_HTH = np.linalg.inv(HTH)
        except np.linalg.LinAlgError:
            print("Warning: Matrix HTH is a singular matrix")
            return None

        w = gamma * H @ inv_HTH @ ones_vector
        w_norm = np.linalg.norm(w)

        return w_norm
    

    def aggregate_diffs_over_the_air_dp(self, w_norm, pi_norm, sigma_uplink2):
        assert (len(self.uploaded_diffs) > 0)

        # Sum of clipped model difference
        sum_diffs = copy.deepcopy(self.uploaded_diffs[0])

        for param in sum_diffs.parameters():
            param.data.zero_()

        # Aggregation
        for diff in self.uploaded_diffs:
            for sum_param, diff_param in zip(sum_diffs.parameters(), diff.parameters()):
                sum_param.data += diff_param.data

        # flattened diff vector
        sum_diffs_flattened = torch.cat([param.flatten() for param in sum_diffs.parameters()])
        x = sum_diffs_flattened.detach().cpu().numpy()

        # channel noise
        N0 = sigma_uplink2 * (w_norm ** 2)

        # SNR
        # SNR_origin = 10*np.log10(self.num_clients*self.power/sigma_uplink2)
        # SNR_ZF = 10*np.log10(self.num_clients*self.power/(sigma_uplink2 * (pi_norm ** 2)))
        # SNR_DP = 10*np.log10(self.num_clients*self.power/N0)

        # print(f"SNR (origin): {SNR_origin}")
        # print(f"SNR (after ZF): {SNR_ZF}")
        # print(f"SNR (after DP): {SNR_DP}")

        # noise scale: eqv. channel noise
        noise_scale = 1 / 2 * N0
        print("noise: ", noise_scale)
        noise = np.sqrt(noise_scale) * np.random.randn(self.d)

        x_hat = x + noise
        SNR = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(noise) )
        print(f"SNR (practical): {SNR}")
       
        sum_diffs_hat_flattened = torch.from_numpy(x_hat).to(self.device)
        # Reshape the flattened tensor back into the model parameters
        start = 0
        for param in sum_diffs.parameters():
            end = start + param.numel()
            param.data = sum_diffs_hat_flattened[start:end].reshape(param.shape)
            start = end

        # Update global model
        for global_param, diff_param in zip(self.global_model.parameters(), sum_diffs.parameters()):
            global_param.data += self.glr * self.learning_rate / len(self.uploaded_diffs) * diff_param.data


    # AirFL-DP
    def solve_dp_optimization(self):

        # solve DP optimization
        # noise design
        self.sigma2 = 10 ** (self.dBm / 10) * 1e-3
        path_loss = (self.c_light / (4 * np.pi * self.f_c * self.r) ) ** 2   
        sigma_uplink2 = self.sigma2 / path_loss  

        gamma = self.clipping_threshold/np.sqrt(self.d * self.power)
        c_delta = 2 * self.epsilon * np.sqrt(self.d) /np.log(1/self.delta)
        Kappa = ((self.epsilon ** 2) * self.d * sigma_uplink2) / (2*c_delta+8) / np.log(1/self.delta) / (self.clipping_threshold ** 2) / self.join_ratio

        pi_list = list()
        # solve pi_t
        for i in range(self.global_rounds):
            pi_t = self.zero_forcing_beamforming(self.m_antenna, self.num_clients, gamma)
            pi_list.append(pi_t)

        # privacy constraint
        w_list = list()
        summation_result = sum([1 / (pi**2) for pi in pi_list])
        if summation_result <= Kappa:
            print("Do not need to tune")
            w_list = list(pi_list)
        else:
            if self.cvxpy_flag:
                # do optimization
                print("Using CVXPY to solve the optimization problem...")
                optimal_values = self.solve_dp_optimal_cvxpy(pi_list, Kappa)
            else:
                print("Using bisection method to solve the optimization problem...")
                optimal_mu = self.solve_opt_mu(pi_list, Kappa)
                optimal_values = []
                for pi in pi_list:
                    q_t = max(pi, optimal_mu**0.25)
                    optimal_values.append(q_t)

            w_list = list(optimal_values)

        return w_list, pi_list, sigma_uplink2


    def solve_dp_optimal_cvxpy(self, pi_list, Kappa):

        # 1. Define optimization variables
        q = cp.Variable(self.global_rounds)
        
        # 2. Define the objective function
        objective = cp.Minimize(cp.sum_squares(q))
        
        # 3. Define constraints
        constraint1 = cp.sum(cp.square(cp.inv_pos(q))) <= Kappa
        constraint2 = q >= pi_list
        constraints = [constraint1, constraint2]
        
        # 4. Formulate and solve the optimization problem
        problem = cp.Problem(objective, constraints)
        
        # Set verbose=True to display solver details
        problem.solve(verbose=False)
        
        # 5. Check solver status and return results
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return q.value
        else:
            print(f"Warning The solver status: {problem.status}")
            return None
        

    # find optimal μ by using bisection search
    def solve_opt_mu(self, pi_list, Kappa, tol=1e-12, max_iter=1000):

        T = len(pi_list)
        
        # Determine the search bounds
        mu_low = 0.0
        mu_high = max((T / Kappa)**2, max(pi**4 for pi in pi_list)) * 1.1
        
        # Define the function h(μ)
        def h_func(mu):
            """Compute ∑[max(p_t, μ^{1/4})]^{-2}"""
            h_value = 0.0
            mu_14 = mu**0.25  # μ^{1/4}
            for pi in pi_list:
                q_t = max(pi, mu_14)
                h_value  += 1 / (q_t**2)
            return h_value 
        
        # bisection method
        for _ in range(max_iter):
            mu_mid = (mu_low + mu_high) / 2
            h_mid = h_func(mu_mid)
            
            # Check convergence
            if abs(h_mid - Kappa) < tol:
                return mu_mid
            
            # Update search bounds
            if h_mid > Kappa:
                mu_low = mu_mid
            else:
                mu_high = mu_mid
        
        # Return the final estimate
        return (mu_low + mu_high) /2
    

    def solve_dp_online(self, sigma_est):
        """
        Online DP optimization over global_rounds.
        Design w_t using the current channel information and estimated future channels,
        while tracking the cumulative privacy cost.
        """

        # Noise and privacy budget settings
        self.sigma2 = 10 ** (self.dBm / 10) * 1e-3
        path_loss = (self.c_light / (4 * np.pi * self.f_c * self.r)) ** 2
        sigma_uplink2 = self.sigma2 / path_loss

        gamma = self.clipping_threshold / np.sqrt(self.d * self.power)
        c_delta = 2 * self.epsilon * np.sqrt(self.d) / np.log(1 / self.delta)
        Kappa_total = ((self.epsilon ** 2) * self.d * sigma_uplink2) / (2 * c_delta + 8) / np.log(1 / self.delta) / (self.clipping_threshold ** 2) / self.join_ratio

        # Initialize online variables
        w_list_online = []
        pi_list_online = []
        privacy_used = 0.0

        # Generate Rayleigh fading channels
        real_part = np.random.randn(self.global_rounds, self.m_antenna, self.num_clients) / np.sqrt(2)
        imag_part = np.random.randn(self.global_rounds, self.m_antenna, self.num_clients) / np.sqrt(2)
        H_true = real_part + 1j * imag_part

        # Online optimization for each round
        for t_now in range(self.global_rounds):

            # Compute pi_t using current and estimated future channels
            pi_list = list()
            for t in range(t_now, self.global_rounds):

                if t == t_now:
                    H_t = H_true[t]
                else:
                    real_noise = np.random.randn(self.m_antenna, self.num_clients) / np.sqrt(2)
                    imag_noise = np.random.randn(self.m_antenna, self.num_clients) / np.sqrt(2)
                    noise = real_noise + 1j * imag_noise

                    # Estimate future channels
                    H_t = np.sqrt(1 - sigma_est**2) * H_true[t] + sigma_est * noise

                pi_t = self.zero_forcing_online(self.m_antenna, self.num_clients, gamma, H_t)
                pi_list.append(pi_t)

            # Compute the remaining privacy budget
            Kappa_remain = Kappa_total - privacy_used
            if Kappa_remain <= 0:
                print(f"[WARN] Privacy budget exhausted at round {t_now}.")
                break

            # Optimize the noise scaling factors
            summation_result = sum([1 / (pi**2) for pi in pi_list])
            if summation_result <= Kappa_remain:
                print("Do not need to tune")
                w_list = list(pi_list)
            else:
                if self.cvxpy_flag:
                    # Solve using CVXPY
                    print("Using CVXPY to solve the optimization problem...")
                    optimal_values = self.solve_dp_optimal_cvxpy(pi_list, Kappa_remain)
                else:
                    # Solve using the bisection method
                    print("Using bisection method to solve the optimization problem...")
                    optimal_mu = self.solve_opt_mu(pi_list, Kappa_remain)
                    optimal_values = []
                    for pi in pi_list:
                        q_t = max(pi, optimal_mu**0.25)
                        optimal_values.append(q_t)

                w_list = list(optimal_values)

            # Store the current-round results
            w_t_now = w_list[0]
            pi_t_now = pi_list[0]
            w_list_online.append(w_t_now)
            pi_list_online.append(pi_t_now)

            # Update the cumulative privacy cost
            privacy_used += 1.0 / (w_t_now ** 2)

        return w_list_online, pi_list_online, sigma_uplink2


    def zero_forcing_online(self, m, n, gamma, H_t):

        assert (m >= n)
        
        HTH = H_t.conj().T @ H_t
        ones_vector = np.ones((n, 1))

        try:
            inv_HTH = np.linalg.inv(HTH)
        except np.linalg.LinAlgError:
            print("Warning: Matrix HTH is a singular matrix")
            return None

        w = gamma * H_t @ inv_HTH @ ones_vector
        w_norm = np.linalg.norm(w)

        return w_norm