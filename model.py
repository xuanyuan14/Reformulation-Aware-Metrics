'''
@ref: Incorporating Query Reformulating Behavior into Web Search Evaluation
@author: Jia Chen, Yiqun Liu, Jiaxin Mao, Fan Zhang, Tetsuya Sakai, Weizhi Ma, Min Zhang, Shaoping Ma
@desc: Implementation of each models
'''
# encoding: utf-8
from __future__ import division
import os
import json
import math
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind, ttest_rel


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=0)
    return a


class RAM:
    def __init__(self, args):
        self.args = args
        self.k_num = self.args.k_num
        self.max_dnum = self.args.max_dnum
        self.max_usefulness = self.args.max_usefulness
        self.id = self.args.id
        self.metric_type = self.args.metric_type
        self.data = self.args.data
        self.click_model = self.args.click_model
        self.use_knowledge = self.args.use_knowledge

        self.iter_num = self.args.iter_num
        self.alpha = self.args.alpha
        self.lamda = self.args.lamda
        self.alpha_decay = self.args.alpha_decay
        self.patience = self.args.patience
        if self.k_num == 6:
            self.train_type = "fineinit"
        else:
            self.train_type = "wo_reform"

        # param initialization
        self.omegas = ["F", "A", "D", "K", "T", "O"]
        if self.use_knowledge:
            self.pi_omega_k = {"F": [100, 0, 0, 0, 0, 0],
                                "A": [0, 89, 1, 7, 3, 0],
                                "D": [0, 8, 76, 12, 2, 1],
                                "K": [0, 10, 1, 86, 2, 1],
                                "T": [0, 32, 7, 15, 43, 4],
                                "O": [0, 8, 4, 4, 34, 49]}
        else:
            self.pi_omega_k = {"F": [1, 0, 0, 0, 0, 0],
                               "A": [0, 1, 0, 0, 0, 0],
                               "D": [0, 0, 1, 0, 0, 0],
                               "K": [0, 0, 0, 1, 0, 0],
                               "T": [0, 0, 0, 0, 1, 0],
                               "O": [0, 0, 0, 0, 0, 1]}
        self.psi_k = [3] * self.k_num
        self.beta_k = [0.5] * self.k_num

        # add some disturbance
        for reform in self.omegas:
            self.pi_omega_k[reform] = self.pi_omega_k[reform][:self.k_num]
            for k in range(self.k_num):
                self.pi_omega_k[reform][k] += random.random() * 0.1
                self.pi_omega_k[reform][k] = math.log(self.pi_omega_k[reform][k])

        self.train_data = self.load_data(mode='train')
        self.test_data = self.load_data(mode='test')

    def load_data(self, mode='train'):
        dir = "./data/bootstrap_%s/%s/%s_file.txt" % (self.data, self.id, mode)
        f = open(dir, "r")
        lines = f.read().strip().split('\n')
        train_data = []
        for line in lines:
            es = line.strip().split("\t")
            train_data.append([es[0], json.loads(es[1]), json.loads(es[2]), es[3]])
        return train_data

    def train_model(self):
        """
            train the RAM with different click mdoels
        """
        pass

    def eval(self):
        """
            evaluate the various models
        """
        pass


class uDBN(RAM):

    def __init__(self, args):
        RAM.__init__(self, args)
        if self.click_model == 'DBN':
            self.gamma_k = [0.5] * self.k_num
        else:   # 'SDBN
            self.gamma_k = [1.0] * self.k_num
        self.sigma_u_k = {i: [0.5] * self.k_num for i in range(self.max_usefulness + 1)}

        # best params
        self.best_pi_omega_k,  self.best_i_omega_k  = {}, {}
        if self.click_model == 'DBN':
            self.best_gamma_k = [0.5] * self.k_num
        else:   # 'SDBN
            self.best_gamma_k = [1.0] * self.k_num
        self.best_sigma_u_k = {i: [0.5] * self.k_num for i in range(self.max_usefulness + 1)}
        self.best_psi_k = [3] * self.k_num
        self.best_beta_k = [0.5] * self.k_num

        # randomize
        for k in range(self.k_num):
            self.psi_k[k] += random.random() * 0.05
            self.beta_k[k] += random.random() * 0.05
            for usef in range(self.max_usefulness + 1):
                self.sigma_u_k[usef][k] += random.random() * 0.05

    def train_model(self):
        num_q = len(self.train_data)
        last_loss1, last_loss2, best_loss = 1e30, 1e30, 1e30
        for i in range(int(self.iter_num)):
            loss, loss1, loss2 = 0., 0., 0.

            # initialize the partials
            partial_pi_omega_k = {"F": np.zeros(self.k_num, dtype=float),
                                  "A": np.zeros(self.k_num, dtype=float),
                                  "D": np.zeros(self.k_num, dtype=float),
                                  "K": np.zeros(self.k_num, dtype=float),
                                  "T": np.zeros(self.k_num, dtype=float),
                                  "O": np.zeros(self.k_num, dtype=float)}
            partial_gamma_k = np.zeros(self.k_num, dtype=float)
            partial_sigma_u_k = np.zeros((self.max_usefulness + 1, self.k_num), dtype=float)
            partial_psi_k = np.zeros(self.k_num, dtype=float)
            partial_beta_k = np.zeros(self.k_num, dtype=float)

            # update i_omega_k according to pi
            i_omega_k = {}
            for key in self.pi_omega_k:
                intent_dist = np.array(self.pi_omega_k[key], dtype=float)
                softmax_dist = soft_max(intent_dist).tolist()
                if key not in i_omega_k:
                    i_omega_k[key] = softmax_dist

            # training
            iter_loss1 = 0
            for train_s in self.train_data:
                if self.k_num == 1:  # w/o reform inform
                    reform = 'F'
                else:
                    reform = train_s[0]
                clicks = train_s[1]
                usefs = train_s[2]
                sat = float(train_s[3])

                alphas = (np.exp2(usefs) - 1) / np.exp2(self.max_usefulness)
                alphas = alphas.tolist()

                # loss calculation
                epsilons, epsilons_1 = np.zeros((self.k_num, self.max_dnum), dtype=float), np.zeros((self.k_num, self.max_dnum), dtype=float)  # epsilon: conditional, epsilon_1: independent
                epsilons += 1e-30
                epsilons_1 += 1e-30
                epsilons[:, 0], epsilons_1[:, 0] = 1.0, 1.0

                for j in range(1, self.max_dnum):
                    for k in range(self.k_num):
                        # conditional without k, independent with k
                        epsilons[k][j] = self.gamma_k[k] * (
                        clicks[j - 1] * (1 - self.sigma_u_k[usefs[j - 1]][k]) + (1 - clicks[j - 1]) * (1 - alphas[j - 1]) *
                        epsilons[k][j - 1] / (1 - alphas[j - 1] * epsilons[k][j - 1]))
                        epsilons_1[k][j] = self.gamma_k[k] * epsilons_1[k][j - 1] * (
                        1 - alphas[j] * self.sigma_u_k[usefs[j - 1]][k])

                    sum_epsilons = np.dot(i_omega_k[reform], epsilons)  # 1 x k, k x N

                    try:
                        loss1 -= math.log(
                            (alphas[j] * sum_epsilons[j] + 1e-30) * clicks[j] + (1 - alphas[j] * sum_epsilons[j]) * (
                            1 - clicks[j]))
                    except:
                        print(
                            'math.log error with j = {}, alpha = {} epsilon = {}'.format(j, alphas[j], sum_epsilons[j]))
                        print(sum_epsilons)
                try:
                    loss1 -= math.log((alphas[self.max_dnum - 1] * sum_epsilons[self.max_dnum - 1] + 1e-30) * clicks[self.max_dnum - 1] + (1 - alphas[self.max_dnum - 1] * sum_epsilons[self.max_dnum - 1]) * (1 - clicks[self.max_dnum - 1]))
                except:
                    print('math.log error with alpha = {} epsilon = {}'.format(alphas[self.max_dnum - 1],
                                                                               sum_epsilons[self.max_dnum - 1]))
                    print(sum_epsilons)

                if loss1 - iter_loss1 >= 50:
                    loss1 = iter_loss1
                iter_loss1 = loss1

                pred_sat_total = 0.
                for k in range(self.k_num):
                    pred_sat = 0.
                    for j in range(self.max_dnum):
                        if self.metric_type == "effort":
                            this_sigma = np.mean(np.array(self.sigma_u_k[usefs[j]]))
                            pred_sat += this_sigma * alphas[j] * epsilons_1[k][j] / (j + 1)
                        elif self.metric_type == "expected_utility":
                            pred_sat += alphas[j] * usefs[j] * epsilons_1[k][j]
                    pred_sat_total = pred_sat_total + i_omega_k[reform][k] * self.beta_k[k] * (pred_sat + self.psi_k[k])
                loss2 += (sat - pred_sat_total) ** 2

                # update parameters, calculate all derivatives
                # click estimation, epsilon, * (-1)

                # sigma
                partial_epsilon_sigma = {}
                rel2min_rank = {}

                for j in range(self.max_dnum):
                    this_rel = usefs[j]
                    if this_rel not in rel2min_rank:
                        rel2min_rank[this_rel] = j

                for rel in rel2min_rank.keys():
                    if rel not in partial_epsilon_sigma:
                        partial_epsilon_sigma[rel] = np.zeros((self.k_num, self.max_dnum), dtype=float)  # p 1 1 = 0
                    for j in range(rel2min_rank[rel] + 1, self.max_dnum):
                        indicator = 1 if (usefs[j - 1] == rel) else 0
                        for k in range(self.k_num):
                            partial_epsilon_sigma[rel][k][j] = (self.gamma_k[k] * (
                            - indicator * clicks[j - 1] + (1 - clicks[j - 1]) * (1 - alphas[j - 1]) / (
                            (1 - alphas[j - 1] * epsilons[k][j - 1]) ** 2) * partial_epsilon_sigma[rel][k][j - 1]))

                    for j in range(self.max_dnum):
                        this_rel = usefs[j]
                        for k in range(self.k_num):
                            partial_sigma_u_k[this_rel][k] -= ((1 - self.lamda) * ((clicks[j] / sum_epsilons[j] - (1 - clicks[j]) * alphas[
                                j] / (1 - alphas[j] * sum_epsilons[j])) * i_omega_k[reform][k] *
                                                               partial_epsilon_sigma[rel][k][j]))

                # gamma
                partial_epsilon_gamma = np.zeros((self.k_num, self.max_dnum), dtype=float)
                for j in range(1, self.max_dnum):
                    for k in range(self.k_num):
                        partial_g_gamma = (epsilons[k][j - 1] - alphas[j - 1] * (epsilons[k][j - 1] ** 2) + self.gamma_k[k] *
                                           partial_epsilon_gamma[k][j - 1]) / (1 - alphas[j - 1] * epsilons[k][j - 1]) ** 2
                        partial_epsilon_gamma[k][j] = (clicks[j - 1] * (1 - self.sigma_u_k[usefs[j - 1]][k]) + (1 - clicks[j - 1]) * (1 - alphas[j - 1]) * partial_g_gamma)

                for k in range(self.k_num):
                    for j in range(self.max_dnum):
                        partial_gamma_k[k] -= ((1 - self.lamda) * ((clicks[j] / sum_epsilons[j] - (1 - clicks[j]) * alphas[j] / (
                        1 - alphas[j] * sum_epsilons[j])) * i_omega_k[reform][k] * partial_epsilon_gamma[k][j]))

                # pi
                for k in range(self.k_num):
                    partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (np.sum(np.array(self.pi_omega_k[reform]))) - math.exp(
                        2 * self.pi_omega_k[reform][k])) / ((np.sum(np.array(self.pi_omega_k[reform]))) ** 2)
                    for j in range(self.max_dnum):
                        partial_pi_omega_k[reform][k] -= ((1 - self.lamda) * ((clicks[j] / sum_epsilons[j] - (1 - clicks[j]) * alphas[j] / (
                        1 - alphas[j] * sum_epsilons[j])) * epsilons[k][j] * partial_i_pi))

                # satisfaction estimation
                partial_epsilon_1_sigma = {}
                partial_epsilon_1_gamma = np.zeros((self.k_num, self.max_dnum), dtype=float)
                # partial_epsilon_1_alpha = {}
                for rel in rel2min_rank.keys():
                    if rel not in partial_epsilon_1_sigma:
                        partial_epsilon_1_sigma[rel] = np.zeros((self.k_num, self.max_dnum), dtype=float)  # p 1 1 = 0
                    for j in range(rel2min_rank[rel] + 1, self.max_dnum):
                        indicator = 1 if (usefs[j - 1] == rel) else 0
                        for k in range(self.k_num):
                            partial_epsilon_1_sigma[rel][k][j] = (- indicator * self.gamma_k[k] * self.sigma_u_k[usefs[j - 1]][k] * epsilons_1[k][j - 1] + self.gamma_k[k] * (1 - alphas[j - 1] * self.sigma_u_k[usefs[j - 1]][k]) * partial_epsilon_1_sigma[rel][k][j - 1])

                for j in range(1, self.max_dnum):
                    for k in range(self.k_num):
                        partial_epsilon_1_gamma[k][j] = (1 - alphas[j - 1] * self.sigma_u_k[usefs[j - 1]][k]) * (
                        epsilons_1[k][j - 1] + self.gamma_k[k] * partial_epsilon_1_gamma[k][j - 1])

                if self.metric_type == "expected_utility":
                    for k in range(self.k_num):
                        partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (
                        np.sum(np.array(self.pi_omega_k[reform]))) - math.exp(2 * self.pi_omega_k[reform][k])) / (
                                       (np.sum(np.array(self.pi_omega_k[reform]))) ** 2)

                        partial_beta_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.psi_k[k])
                        partial_psi_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k])
                        partial_pi_omega_k[reform][k] += (
                        self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * self.psi_k[k])
                        for j in range(self.max_dnum):
                            partial_beta_k[k] += (
                            self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * alphas[j] * usefs[j] *
                            epsilons_1[k][j])
                            partial_pi_omega_k[reform][k] += (
                            self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * alphas[j] * usefs[j] *
                            epsilons_1[k][j])

                        for rel in rel2min_rank.keys():
                            for j in range(rel2min_rank[rel] + 1, self.max_dnum):
                                partial_sigma_u_k[rel][k] += (
                                self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k] * (
                                alphas[j] * usefs[j] * partial_epsilon_1_sigma[rel][k][j]))
                                partial_gamma_k[k] += (
                                self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k] * alphas[j] *
                                usefs[j] * partial_epsilon_1_gamma[k][j])

                elif self.metric_type == "effort":
                    for k in range(self.k_num):
                        partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (
                        np.sum(np.array(self.pi_omega_k[reform]))) - math.exp(2 * self.pi_omega_k[reform][k])) / (
                                       (np.sum(np.array(self.pi_omega_k[reform]))) ** 2)

                        partial_beta_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.psi_k[k])
                        partial_psi_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k])
                        partial_pi_omega_k[reform][k] += (
                        self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * self.psi_k[k])
                        for j in range(self.max_dnum):
                            partial_beta_k[k] += (
                            self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.sigma_u_k[usefs[j]][k] * alphas[
                                j] * epsilons_1[k][j] / (j + 1))
                            partial_pi_omega_k[reform][k] += (
                            self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * alphas[j] *
                            self.sigma_u_k[usefs[j]][k] * epsilons_1[k][j] / (j + 1))

                        for rel in rel2min_rank.keys():
                            for j in range(rel2min_rank[rel] + 1, self.max_dnum):
                                # indicator = 1 if (usefs[j] == rel) else 0
                                partial_sigma_u_k[rel][k] += (
                                self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k] * (
                                alphas[j] * self.sigma_u_k[usefs[j]][k] * partial_epsilon_1_sigma[rel][k][j]) / (j + 1))
                                partial_gamma_k[k] += (
                                self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k] * alphas[j] *
                                self.sigma_u_k[usefs[j]][k] * partial_epsilon_1_gamma[k][j] / (j + 1))

            for k in range(self.k_num):
                for reform in self.omegas:
                    if partial_pi_omega_k[reform][k] > 0:
                        self.pi_omega_k[reform][k] -= self.alpha
                    elif partial_pi_omega_k[reform][k] < 0:
                        self.pi_omega_k[reform][k] += self.alpha

                if self.click_model == 'DBN':
                    if partial_gamma_k[k] > 0:  # setting bound
                        self.gamma_k[k] = max(self.gamma_k[k] - self.alpha, 1e-4)
                    elif partial_gamma_k[k] < 0:
                        self.gamma_k[k] = min(self.gamma_k[k] + self.alpha, 1 - 1e-4)

                if partial_psi_k[k] > 0:
                    self.psi_k[k] -= self.alpha
                elif partial_psi_k[k] < 0:
                    self.psi_k[k] += self.alpha

                if partial_beta_k[k] > 0:
                    self.beta_k[k] -= self.alpha
                elif partial_beta_k[k] < 0:
                    self.beta_k[k] += self.alpha

            for rel in range(self.max_usefulness):  # setting bound
                for k in range(self.k_num):
                    if partial_sigma_u_k[rel][k] > 0:
                        self.sigma_u_k[rel][k] = max(self.sigma_u_k[rel][k] - self.alpha, 1e-4)
                    elif partial_sigma_u_k[rel][k] < 0:
                        self.sigma_u_k[rel][k] = min(self.sigma_u_k[rel][k] + self.alpha, 1 - 1e-4)

            loss1 /= num_q
            loss2 /= num_q
            loss = (1 - self.lamda) * loss1 + self.lamda * loss2
            print("Iter = {}\tLoss = {}\tLoss1 = {}\tLoss2={}".format(i + 1, loss, loss1, loss2))

            if loss > ((1 - self.lamda) * last_loss1 + self.lamda * last_loss2):
                print("Loss increasing...")
                self.alpha *= 0.5
                self.patience -= 1
                if self.patience == 0:
                    print("Training terminated with loss increasing...")
                    break
            elif (loss1 < last_loss1 and math.fabs(loss1 - last_loss1) <= 1e-9) or (
                    loss2 < last_loss2 and math.fabs(loss2 - last_loss2) <= 1e-9):
                print("Training done with iter_num = {}".format(i + 1))
                break
            else:
                if loss < best_loss:
                    best_loss = loss
                    self.best_pi_omega_k = self.pi_omega_k.copy()
                    self.best_gamma_k = self.gamma_k[:]
                    self.best_sigma_u_k = self.sigma_u_k.copy()
                    self.best_psi_k = self.psi_k[:]
                    self.best_beta_k = self.beta_k[:]

                self.patience = 5

            last_loss1 = loss1
            last_loss2 = loss2
            self.alpha *= self.alpha_decay

        for key in self.best_pi_omega_k:
            intent_dist = np.array(self.best_pi_omega_k[key], dtype=float)
            softmax_dist = soft_max(intent_dist).tolist()
            if key not in self.best_i_omega_k:
                self.best_i_omega_k[key] = softmax_dist

        for reform in self.omegas:
            for k in range(self.k_num):
                print('i {} {} = {}'.format(reform, k, self.best_i_omega_k[reform][k]))

        for k in range(self.k_num):
            print('psi {} = {}'.format(k, self.best_psi_k[k]))

        for k in range(self.k_num):
            print('beta {} = {}'.format(k, self.best_beta_k[k]))

        for k in range(self.k_num):
            print('gamma {} = {}'.format(k, self.best_gamma_k[k]))

        for rel in range(self.max_usefulness + 1):
            for k in range(self.k_num):
                print('sigma {} {} = {}'.format(rel, k, self.best_sigma_u_k[rel][k]))

        # save params
        param_dir = "./data/bootstrap_%s/%s/%s_%s_%s" % (self.data, self.id, self.click_model, self.metric_type, self.train_type)

        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

        np.save("%s/best_psi_k.npy" % param_dir, self.best_psi_k)
        np.save("%s/best_beta_k.npy" % param_dir, self.best_beta_k)
        np.save("%s/best_gamma_k.npy" % param_dir, self.best_gamma_k)
        with open("%s/best_pi_omega_k.json" % param_dir, "w") as fw1:
            json.dump(self.best_pi_omega_k, fw1)

        with open("%s/best_sigma_u_k.json" % param_dir, "w") as fw2:
            json.dump(self.best_sigma_u_k, fw2)

    def eval(self):
        param_dir = "./data/bootstrap_%s/%s/%s_%s_%s" % (self.data, self.id, self.click_model, self.metric_type, self.train_type)

        psi_k = np.load('%s/best_psi_k.npy' % param_dir)
        with open('%s/best_pi_omega_k.json' % param_dir, 'r') as f1:
            pi_omega_k = json.load(f1)
        beta_k = np.load("%s/best_beta_k.npy" % param_dir)
        gamma_k = np.load("%s/best_gamma_k.npy" % param_dir)
        with open("%s/best_sigma_u_k.json" % param_dir, 'r') as f2:
            sigma_u_k = json.load(f2)


        i_omega_k = {}
        for key in pi_omega_k:
            intent_dist = np.array(pi_omega_k[key], dtype=float)
            softmax_dist = soft_max(intent_dist).tolist()
            if key not in i_omega_k:
                i_omega_k[key] = softmax_dist

        sat_list, pred_sat_list = [], []
        clicks, pred_clicks = [], []

        for test_s in self.test_data:
            if self.k_num == 1:
                reform = 'F'
            else:
                reform = test_s[0]
            true_reform = test_s[0]
            this_clicks = test_s[1]
            usefs = test_s[2]
            sat = float(test_s[3])
            alphas = (np.exp2(usefs) - 1) / np.exp2(self.max_usefulness)
            alphas = alphas.tolist()

            max_clicked_pos = -1
            for j in range(self.max_dnum):
                if this_clicks[j] == 1:
                    max_clicked_pos = max(max_clicked_pos, j)
           
            # loss calculation
            epsilons, epsilons_1 = np.zeros((self.k_num, self.max_dnum), dtype=float), np.zeros((self.k_num, self.max_dnum), dtype=float)  # epsilon: conditional, epsilon_1: independent
            epsilons += 1e-30
            epsilons_1 += 1e-30
            epsilons[:, 0], epsilons_1[:, 0] = 1.0, 1.0

            # print(sigma_u_k)
            for j in range(1, self.max_dnum):
                for k in range(self.k_num):
                    # conditional without k, independent with k
                    epsilons[k][j] = gamma_k[k] * (this_clicks[j - 1] * (1 - sigma_u_k[str(usefs[j - 1])][k]) + (1 - this_clicks[j - 1]) * (1 - alphas[j - 1]) * epsilons[k][j - 1] / (1 - alphas[j - 1] * epsilons[k][j - 1]))
                    epsilons_1[k][j] = gamma_k[k] * epsilons_1[k][j - 1] * (1 - alphas[j] * sigma_u_k[str(usefs[j - 1])][k])

            sum_epsilons = np.dot(i_omega_k[reform], epsilons_1)

            pred_sat_total = 0.
            for k in range(self.k_num):
                pred_sat = 0.
                for j in range(self.max_dnum):
                    if self.metric_type == "effort":
                        this_sigma = np.mean(np.array(self.sigma_u_k[usefs[j]]))
                        pred_sat += this_sigma * alphas[j] * epsilons_1[k][j] / (j + 1)
                    elif self.metric_type == "expected_utility":
                        pred_sat += alphas[j] * usefs[j] * epsilons_1[k][j]
                pred_sat_total = pred_sat_total + i_omega_k[reform][k] * beta_k[k] * (pred_sat + psi_k[k])
            sat_list.append(sat)
            pred_sat_list.append(pred_sat_total)
            this_pred_clicks = []
            for j in range(self.max_dnum):
                this_pred_click = 0
                for k in range(self.k_num):
                    this_pred_click += i_omega_k[reform][k] * epsilons_1[k][j] * alphas[j]
                this_pred_clicks.append(this_pred_click)

            this_pred_clicks = np.array(this_pred_clicks)
            this_clicks = np.array(this_clicks)
            this_loss = -np.sum(
                np.log2((this_pred_clicks + 1e-30) * this_clicks + (1 - this_pred_clicks + 1e-30) * (1 - this_clicks)))
            if this_loss < 50:
                clicks.append(this_clicks.tolist())
                pred_clicks.append(this_pred_clicks.tolist())

        clicks = np.array(clicks)  # n * 10
        pred_clicks = np.array(pred_clicks)  # n * 10
        ppl_all = clicks * np.log2(pred_clicks + 1e-30) + (1 - clicks) * np.log2(1 - pred_clicks + 1e-30)  # n * 10
        ppl_rank = np.mean(ppl_all, axis=0)  # 1 * 10
        ppl_base = np.array([2.0] * self.max_dnum)
        ppl_rank = np.power(ppl_base, -ppl_rank)
        ppl_all = np.mean(ppl_rank)

        r1, p1 = pearsonr(sat_list, pred_sat_list)
        r2, p2 = spearmanr(sat_list, pred_sat_list)
        mse = np.mean(np.square(np.array(sat_list) - np.array(pred_sat_list)))
        print("Evaluation: pearson = {}, spearman = {}, mse = {}, PPL = {}".format(r1, r2, mse, ppl_all))
        np.save("%s/sat_list.npy" % param_dir, sat_list)
        np.save("%s/pred_sat_list.npy" % param_dir, pred_sat_list)


class uUBM(RAM):

    def __init__(self, args):
        RAM.__init__(self, args)
        # gamma_{r, r', k}, r: 1-10, r': 1-10
        self.gamma_k = np.zeros((self.max_dnum + 1, self.max_dnum + 1, self.k_num), dtype=float)
        self.gamma_k += 0.5
        self.alpha_R = {i: ((math.pow(2, i) - 1) / math.pow(2, self.max_usefulness)) + 1e-30 for i in range(self.max_usefulness + 1)}

        # best params
        self.best_pi_omega_k, self.best_i_omega_k = {}, {}
        self.best_gamma_k = np.zeros((self.max_dnum + 1, self.max_dnum + 1, self.k_num), dtype=float)
        self.best_psi_k = [3] * self.k_num
        self.best_beta_k = [0.5] * self.k_num
        self.best_alpha_R = {i: ((math.pow(2, i) - 1) / math.pow(2, self.max_usefulness)) for i in range(self.max_usefulness + 1)}

        # randomize
        for k in range(self.k_num):
            self.psi_k[k] += random.random() * 0.05
            self.beta_k[k] += random.random() * 0.05

    def train_model(self):
        last_loss1, last_loss2, best_loss = 1e30, 1e30, 1e30
        for i in range(int(self.iter_num)):
            loss, loss1, loss2 = 0., 0., 0.

            # initialize the partials
            partial_pi_omega_k = {"F": np.zeros(self.k_num, dtype=float),
                                  "A": np.zeros(self.k_num, dtype=float),
                                  "D": np.zeros(self.k_num, dtype=float),
                                  "K": np.zeros(self.k_num, dtype=float),
                                  "T": np.zeros(self.k_num, dtype=float),
                                  "O": np.zeros(self.k_num, dtype=float)}
            partial_gamma_k = np.zeros((self.max_dnum + 1, self.max_dnum + 1, self.k_num), dtype=float)
            partial_psi_k = np.zeros(self.k_num, dtype=float)
            partial_beta_k = np.zeros(self.k_num, dtype=float)

            # update i_omega_k according to pi
            i_omega_k = {}
            for key in self.pi_omega_k:
                intent_dist = np.array(self.pi_omega_k[key], dtype=float)
                softmax_dist = soft_max(intent_dist).tolist()
                if key not in i_omega_k:
                    i_omega_k[key] = softmax_dist

            # training
            num_q = len(self.train_data)
            iter_loss1 = 0
            for train_s in self.train_data:
                if self.k_num == 1:  # w/o reform inform
                    reform = 'F'
                else:
                    reform = train_s[0]
                clicks = train_s[1]
                usefs = train_s[2]
                sat = float(train_s[3])
                alphas = (np.exp2(usefs) - 1) / np.exp2(self.max_usefulness)

                # find previous click
                prev_click_pos = np.zeros(self.max_dnum, dtype=int)
                for j in range(self.max_dnum):
                    if clicks[j] == 1:
                        prev_click_pos[j + 1: ] = j + 1

                # 0, 1-10
                cond_clicks, ind_clicks = np.zeros((self.k_num, self.max_dnum + 1), dtype=float), np.zeros((self.k_num, self.max_dnum + 1), dtype=float)

                cond_clicks += 1e-30
                ind_clicks += 1e-30
                cond_clicks[:, 0], ind_clicks[: 0] = 1.0, 1.0   # C_0 = 1.0

                # gamma_{r, r', k}, r: 1-10, r': 0-9, j: 1-10
                for j in range(1, self.max_dnum + 1):
                    for k in range(self.k_num):
                        # conditional without k, independent with k
                        cond_clicks[k][j] += alphas[j - 1] * self.gamma_k[j][prev_click_pos[j - 1]][k]  # 0, 0, 1, 0
                        for l in range(j):
                            factor = 1.0
                            for m in range(l + 1, j):
                                factor = factor * (1 - alphas[m - 1] * self.gamma_k[m][l][k])
                            ind_clicks[k][j] += ind_clicks[k][l] * factor * alphas[j - 1] * self.gamma_k[j][l][k]

                    sum_clicks = np.dot(i_omega_k[reform], cond_clicks)  # 1 x k, k x (N + 1) ==> 1 x (N + 1)

                    try:
                        loss1 -= math.log((sum_clicks[j] + 1e-30) * clicks[j - 1] + (1 - sum_clicks[j]) * (1 - clicks[j - 1]))
                    except:
                        print('math.log error with j = {}, click = {}'.format(j, sum_clicks[j]))
                try:
                    loss1 -= math.log((sum_clicks[self.max_dnum] + 1e-30) * clicks[self.max_dnum - 1] + (1 - sum_clicks[self.max_dnum]) * (1 - clicks[self.max_dnum - 1]))
                except:
                    print('math.log error with j = {}, click = {}'.format(j, sum_clicks[j]))

                if loss1 - iter_loss1 >= 50:
                    loss1 = iter_loss1
                iter_loss1 = loss1

                pred_sat_total = 0.
                for k in range(self.k_num):
                    pred_sat = 0.
                    for j in range(self.max_dnum):
                        pred_sat += usefs[j] * cond_clicks[k][j + 1]
                    pred_sat_total = pred_sat_total + i_omega_k[reform][k] * self.beta_k[k] * (pred_sat + self.psi_k[k])
                loss2 += (sat - pred_sat_total) ** 2

                # update parameters, calculate all derivatives
                # click estimation, epsilon, * (-1)

                # gamma
                for j in range(1, self.max_dnum + 1):
                    for k in range(self.k_num):
                        prev_click = prev_click_pos[j - 1]
                        partial_gamma_k[j][prev_click][k] -= ((1 - self.lamda) * ((i_omega_k[reform][k] * alphas[j - 1] / sum_clicks[j]) * clicks[j - 1] + ((-alphas[j - 1] * i_omega_k[reform][k]) / (1 - sum_clicks[j])) * (1 - clicks[j - 1])))

                # pi
                for k in range(self.k_num):
                    partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (np.sum(np.array(self.pi_omega_k[reform]))) - math.exp(2 * self.pi_omega_k[reform][k])) / ((np.sum(np.array(self.pi_omega_k[reform]))) ** 2)
                    for j in range(1, self.max_dnum + 1):
                        prev_click = prev_click_pos[j - 1]
                        partial_pi_omega_k[reform][k] -= ((1 - self.lamda) * ((clicks[j - 1] * alphas[j - 1] * self.gamma_k[j][prev_click][k] / sum_clicks[j]
                                                           + (1 - clicks[j - 1] - 1) * alphas[j - 1] * (self.gamma_k[j][prev_click][k] ) / (1 - sum_clicks[j])) * partial_i_pi))

                if self.metric_type == "expected_utility":
                    for k in range(self.k_num):
                        partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (np.sum(np.array(self.pi_omega_k[reform])))
                                        - math.exp(2 * self.pi_omega_k[reform][k])) / ((np.sum(np.array(self.pi_omega_k[reform]))) ** 2)

                        partial_beta_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.psi_k[k])
                        partial_psi_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k])
                        partial_pi_omega_k[reform][k] += (self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * self.psi_k[k])
                        for j in range(self.max_dnum):
                            partial_beta_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * ind_clicks[k][j + 1] * usefs[j])
                            partial_pi_omega_k[reform][k] += (self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * ind_clicks[k][j + 1] * usefs[j])

                        for j in range(1, self.max_dnum + 1):
                            prev_click = prev_click_pos[j - 1]
                            for k in range(self.k_num):
                                partial_gamma_k[j][prev_click][k] += (self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * alphas[j - 1] * usefs[j - 1])

            for k in range(self.k_num):
                for reform in self.omegas:
                    if partial_pi_omega_k[reform][k] > 0:
                        self.pi_omega_k[reform][k] -= self.alpha
                    elif partial_pi_omega_k[reform][k] < 0:
                        self.pi_omega_k[reform][k] += self.alpha

                if partial_psi_k[k] > 0:
                    self.psi_k[k] -= self.alpha
                elif partial_psi_k[k] < 0:
                    self.psi_k[k] += self.alpha

                if partial_beta_k[k] > 0:
                    self.beta_k[k] -= self.alpha
                elif partial_beta_k[k] < 0:
                    self.beta_k[k] += self.alpha

                for m in range(self.max_dnum + 1):
                    for n in range(m + 1, self.max_dnum + 1):
                        if partial_gamma_k[n][m][k] > 0:  # setting bound
                            self.gamma_k[n][m][k] = max(self.gamma_k[n][m][k] - self.alpha, 1e-4)
                        elif partial_gamma_k[n][m][k] < 0:
                            self.gamma_k[n][m][k] = min(self.gamma_k[n][m][k] + self.alpha, 1 - 1e-4)

            loss1 /= num_q
            loss2 /= num_q
            loss = (1 - self.lamda) * loss1 + self.lamda * loss2
            print("Iter = {}\tLoss = {}\tLoss1 = {}\tLoss2={}".format(i + 1, loss, loss1, loss2))

            if loss > ((1 - self.lamda) * last_loss1 + self.lamda * last_loss2):
                print("Loss increasing...")
                self.alpha *= 0.5
                self.patience -= 1
                if self.patience == 0:
                    print("Training terminated with loss increasing...")
                    break
            elif (loss1 < last_loss1 and math.fabs(loss1 - last_loss1) <= 1e-9) or (loss2 < last_loss2 and math.fabs(loss2 - last_loss2) <= 1e-9):
                print("Training done with iter_num = {}".format(i + 1))
                break
            else:
                if loss < best_loss:
                    best_loss = loss
                    self.best_pi_omega_k = self.pi_omega_k.copy()
                    self.best_gamma_k = self.gamma_k[:, :, :]
                    self.best_psi_k = self.psi_k[:]
                    self.best_beta_k = self.beta_k[:]
                    self.best_alpha_R = self.alpha_R.copy()

                self.patience = 5

            last_loss1 = loss1
            last_loss2 = loss2
            self.alpha *= self.alpha_decay

        for key in self.best_pi_omega_k:
            intent_dist = np.array(self.best_pi_omega_k[key], dtype=float)
            softmax_dist = soft_max(intent_dist).tolist()
            if key not in self.best_i_omega_k:
                self.best_i_omega_k[key] = softmax_dist

        for reform in self.omegas:
            for k in range(self.k_num):
                print('i {} {} = {}'.format(reform, k, self.best_i_omega_k[reform][k]))

        for r in range(self.max_usefulness + 1):
            print('alpha {} = {}'.format(r, self.best_alpha_R[r]))

        for k in range(self.k_num):
            print('psi {} = {}'.format(k, self.best_psi_k[k]))

        for k in range(self.k_num):
            print('beta {} = {}'.format(k, self.best_beta_k[k]))

        for k in range(self.k_num):
            for m in range(self.max_dnum + 1):
                for n in range(self.max_dnum + 1):
                    print('gamma {} {} {} = {}'.format(n, m, k, self.best_gamma_k[n][m][k]))

        # save params
        param_dir = "./data/bootstrap_%s/%s/%s_%s_%s" % (self.data, self.id, self.click_model, self.metric_type, self.train_type)

        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

        np.save("%s/best_psi_k.npy" % param_dir, self.best_psi_k)
        np.save("%s/best_beta_k.npy" % param_dir, self.best_beta_k)
        np.save("%s/best_gamma_k.npy" % param_dir, self.best_gamma_k)
        with open("%s/best_pi_omega_k.json" % param_dir, "w") as fw1:
            json.dump(self.best_pi_omega_k, fw1)
        with open("%s/best_alpha_R.json" % param_dir, "w") as fw2:
            json.dump(self.best_alpha_R, fw2)

    def eval(self):
        param_dir = "./data/bootstrap_%s/%s/%s_%s_%s" % (
            'fsd', self.id, self.click_model, self.metric_type, self.train_type)

        best_psi_k = np.load("%s/best_psi_k.npy" % param_dir)
        best_beta_k = np.load("%s/best_beta_k.npy" % param_dir)
        best_gamma_k = np.load("%s/best_gamma_k.npy" % param_dir)
        with open("%s/best_pi_omega_k.json" % param_dir, "r") as f1:
            best_pi_omega_k = json.load(f1)

        i_omega_k = {}
        for key in best_pi_omega_k:
            intent_dist = np.array(best_pi_omega_k[key], dtype=float)
            softmax_dist = soft_max(intent_dist).tolist()
            if key not in i_omega_k:
                i_omega_k[key] = softmax_dist

        sat_list, pred_sat_list = [], []
        clicks, pred_clicks = [], []
        for test_s in self.test_data:
            if self.k_num == 1:  # w/o reform inform
                reform = 'F'
            else:
                reform = test_s[0]
            true_reform = test_s[0]
            this_clicks = test_s[1]
            usefs = test_s[2]
            sat = float(test_s[3])
            # alphas = [best_alpha_R[int(usef)] for usef in usefs]
            alphas = (np.exp2(usefs) - 1) / np.exp2(self.max_usefulness)
            # alphas = alphas.tolist()

            max_clicked_pos = -1
            for j in range(self.max_dnum):
                if this_clicks[j] == 1:
                    max_clicked_pos = max(max_clicked_pos, j)

            # find previous click, 0-9
            prev_click_pos = np.zeros(self.max_dnum, dtype=int)
            for j in range(self.max_dnum):
                if this_clicks[j] == 1:
                    prev_click_pos[j + 1:] = j + 1

            # 0, 1-10
            cond_clicks, ind_clicks = np.zeros((self.k_num, self.max_dnum + 1), dtype=float), np.zeros((self.k_num, self.max_dnum + 1), dtype=float)

            cond_clicks += 1e-30
            ind_clicks += 1e-30
            cond_clicks[:, 0], ind_clicks[: 0] = 1.0, 1.0  # C_0 = 1.0

            # gamma_{r, r', k}, r: 1-10, r': 0-9, j: 1-10
            sum_epsilons = np.zeros(self.max_dnum, dtype=float)
            for j in range(1, self.max_dnum + 1):
                for k in range(self.k_num):
                    # conditional without k, independent with k
                    sum_epsilons[j - 1] += i_omega_k[reform][k] * best_gamma_k[j][prev_click_pos[j - 1]][k]
                    cond_clicks[k][j] += alphas[j - 1] * best_gamma_k[j][prev_click_pos[j - 1]][k]  # 0, 0, 1, 0
                    for l in range(j):
                        factor = 1.0
                        for m in range(l + 1, j):
                            factor = factor * (1 - alphas[m - 1] * best_gamma_k[m][l][k])
                        # print(ind_clicks[k][l], factor, alphas[j - 1], best_gamma_k[j][l][k])
                        ind_clicks[k][j] += ind_clicks[k][l] * factor * alphas[j - 1] * best_gamma_k[j][l][k]

            pred_sat_total = 0.
            for k in range(self.k_num):
                pred_sat = 0.
                for j in range(self.max_dnum):
                    pred_sat += usefs[j] * cond_clicks[k][j + 1]
                pred_sat_total = pred_sat_total + i_omega_k[reform][k] * best_beta_k[k] * (pred_sat + best_psi_k[k])

            sat_list.append(sat)
            pred_sat_list.append(pred_sat_total)

            this_pred_clicks = np.dot(i_omega_k[reform], cond_clicks)[1:]
            this_pred_clicks = np.array(this_pred_clicks)
            this_clicks = np.array(this_clicks)
            this_loss = - np.sum(np.log2((this_pred_clicks + 1e-30) * this_clicks + (1 - this_pred_clicks + 1e-30) * (1 - this_clicks)))
            if this_loss < 50:
                # print(this_clicks, this_pred_clicks)
                clicks.append(this_clicks.tolist())
                pred_clicks.append(this_pred_clicks.tolist())

        clicks = np.array(clicks)  # n * 10
        pred_clicks = np.array(pred_clicks)  # n * 10
        ppl_all = clicks * np.log2(pred_clicks + 1e-30) + (1 - clicks) * np.log2(1 - pred_clicks + 1e-30)  # n * 10
        ppl_rank = np.mean(ppl_all, axis=0)  # 1 * 10
        ppl_base = np.array([2.0] * self.max_dnum)
        ppl_rank = np.power(ppl_base, -ppl_rank)
        ppl_all = np.mean(ppl_rank)

        r1, p1 = pearsonr(sat_list, pred_sat_list)
        r2, p2 = spearmanr(sat_list, pred_sat_list)
        mse = np.mean(np.square(np.array(sat_list) - np.array(pred_sat_list)))
        print("Evaluation: pearson = {}, spearman = {}, mse = {}, PPL = {}".format(r1, r2, mse, ppl_all))
        np.save("%s/sat_list.npy" % param_dir, sat_list)
        np.save("%s/pred_sat_list.npy" % param_dir, pred_sat_list)


class uPBM(RAM):
    def __init__(self, args):
        RAM.__init__(self, args)
        # gamma_{r, k}, r: 0-9
        self.gamma_k = np.zeros((self.max_dnum, self.k_num), dtype=float)
        self.gamma_k += 0.5

        # best params
        self.best_pi_omega_k, self.best_i_omega_k = {}, {}
        self.best_gamma_k = np.zeros((self.max_dnum, self.k_num), dtype=float)
        self.best_psi_k = [3] * self.k_num
        self.best_beta_k = [0.5] * self.k_num

        # randomize
        for k in range(self.k_num):
            self.psi_k[k] += random.random() * 0.05
            self.beta_k[k] += random.random() * 0.05

    def train_model(self):
        last_loss1, last_loss2, best_loss = 1e30, 1e30, 1e30
        for i in range(int(self.iter_num)):
            loss, loss1, loss2 = 0., 0., 0.

            # initialize the partials
            partial_pi_omega_k = {"F": np.zeros(self.k_num, dtype=float),
                                  "A": np.zeros(self.k_num, dtype=float),
                                  "D": np.zeros(self.k_num, dtype=float),
                                  "K": np.zeros(self.k_num, dtype=float),
                                  "T": np.zeros(self.k_num, dtype=float),
                                  "O": np.zeros(self.k_num, dtype=float)}
            partial_gamma_k = np.zeros((self.max_dnum, self.k_num), dtype=float)
            partial_psi_k = np.zeros(self.k_num, dtype=float)
            partial_beta_k = np.zeros(self.k_num, dtype=float)

            # update i_omega_k according to pi
            i_omega_k = {}
            for key in self.pi_omega_k:
                intent_dist = np.array(self.pi_omega_k[key], dtype=float)
                softmax_dist = soft_max(intent_dist).tolist()
                if key not in i_omega_k:
                    i_omega_k[key] = softmax_dist

            # training
            num_q = len(self.train_data)
            iter_loss1 = 0
            for train_s in self.train_data:
                if self.k_num == 1:  # w/o reform inform
                    reform = 'F'
                else:
                    reform = train_s[0]
                clicks = train_s[1]
                usefs = train_s[2]
                sat = float(train_s[3])
                usefs = [float(usef) for usef in usefs]
                alphas = (np.exp2(usefs) - 1) / np.exp2(self.max_usefulness)

                # 1-10
                cond_clicks, ind_clicks = np.zeros((self.k_num, self.max_dnum), dtype=float), np.zeros((self.k_num, self.max_dnum), dtype=float)
                cond_clicks += 1e-30
                ind_clicks += 1e-30

                for j in range(self.max_dnum):
                    for k in range(self.k_num):
                        # conditional without k, independent with k
                        cond_clicks[k][j] += alphas[j] * self.gamma_k[j][k]
                        ind_clicks[k][j] += alphas[j] * self.gamma_k[j][k]

                    sum_clicks = np.dot(i_omega_k[reform], cond_clicks)  # 1 x k, k x (N + 1) ==> 1 x (N + 1)

                    try:
                        loss1 -= math.log((sum_clicks[j] + 1e-30) * clicks[j] + (1 - sum_clicks[j]) * (1 - clicks[j]))
                    except:
                        print('math.log error with j = {}, click = {}'.format(j, sum_clicks[j]))
                try:
                    loss1 -= math.log((sum_clicks[self.max_dnum - 1] + 1e-30) * clicks[j] + (1 - sum_clicks[self.max_dnum - 1]) * (1 - clicks[j]))
                except:
                    print('math.log error with j = {}, click = {}'.format(j, sum_clicks[self.max_dnum - 1]))

                if loss1 - iter_loss1 >= 50:
                    loss1 = iter_loss1
                iter_loss1 = loss1

                pred_sat_total = 0.
                for k in range(self.k_num):
                    pred_sat = 0.
                    for j in range(self.max_dnum):
                        pred_sat += usefs[j] * ind_clicks[k][j]
                    pred_sat_total = pred_sat_total + i_omega_k[reform][k] * self.beta_k[k] * (pred_sat + self.psi_k[k])
                loss2 += (sat - pred_sat_total) ** 2

                # update parameters, calculate all derivatives
                # click estimation, epsilon, * (-1)

                # gamma
                for j in range(self.max_dnum):
                    for k in range(self.k_num):
                        partial_gamma_k[j][k] -= ((1 - self.lamda) * ((i_omega_k[reform][k] * alphas[j] / sum_clicks[j]) * clicks[j] + ((-alphas[j] * i_omega_k[reform][k]) / (1 - sum_clicks[j])) * (1 - clicks[j])))

                # pi
                for k in range(self.k_num):
                    partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (np.sum(np.array(self.pi_omega_k[reform]))) - math.exp(2 * self.pi_omega_k[reform][k])) / (
                                   (np.sum(np.array(self.pi_omega_k[reform]))) ** 2)
                    for j in range(self.max_dnum):
                        partial_pi_omega_k[reform][k] -= ((1 - self.lamda) * ((clicks[j] * alphas[j] * self.gamma_k[j][k] / sum_clicks[j]
                                                        + (1 - clicks[j] - 1) * alphas[j] * (self.gamma_k[j][k]) / (1 - alphas[j] * sum_clicks[j])) * partial_i_pi))

                if self.metric_type == "expected_utility":
                    for k in range(self.k_num):
                        partial_i_pi = (math.exp(self.pi_omega_k[reform][k]) * (
                        np.sum(np.array(self.pi_omega_k[reform])))
                                        - math.exp(2 * self.pi_omega_k[reform][k])) / (
                                       (np.sum(np.array(self.pi_omega_k[reform]))) ** 2)

                        partial_beta_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.psi_k[k])
                        partial_psi_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * self.beta_k[k])
                        partial_pi_omega_k[reform][k] += (self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * self.psi_k[k])
                        for j in range(self.max_dnum):
                            partial_beta_k[k] += (self.lamda * 2 * (pred_sat_total - sat) * i_omega_k[reform][k] * ind_clicks[k][j] * usefs[j])
                            partial_pi_omega_k[reform][k] += (self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * partial_i_pi * ind_clicks[k][j] * usefs[j])
                            partial_gamma_k[j][k] += (self.lamda * 2 * (pred_sat_total - sat) * self.beta_k[k] * alphas[j] * usefs[j])

            for k in range(self.k_num):
                for reform in self.omegas:
                    if partial_pi_omega_k[reform][k] > 0:
                        self.pi_omega_k[reform][k] -= self.alpha
                    elif partial_pi_omega_k[reform][k] < 0:
                        self.pi_omega_k[reform][k] += self.alpha

                if partial_psi_k[k] > 0:
                    self.psi_k[k] -= self.alpha
                elif partial_psi_k[k] < 0:
                    self.psi_k[k] += self.alpha

                if partial_beta_k[k] > 0:
                    self.beta_k[k] -= self.alpha
                elif partial_beta_k[k] < 0:
                    self.beta_k[k] += self.alpha

                for m in range(self.max_dnum):
                    if partial_gamma_k[m][k] > 0:  # setting bound
                        self.gamma_k[m][k] = max(self.gamma_k[m][k] - self.alpha, 1e-4)
                    elif partial_gamma_k[m][k] < 0:
                        self.gamma_k[m][k] = min(self.gamma_k[m][k] + self.alpha, 1 - 1e-4)

            loss1 /= num_q
            loss2 /= num_q
            loss = (1 - self.lamda) * loss1 + self.lamda * loss2
            print("Iter = {}\tLoss = {}\tLoss1 = {}\tLoss2={}".format(i + 1, loss, loss1, loss2))

            if loss > ((1 - self.lamda) * last_loss1 + self.lamda * last_loss2):
                print("Loss increasing...")
                self.alpha *= 0.5
                self.patience -= 1
                if self.patience == 0:
                    print("Training terminated with loss increasing...")
                    break
            elif (loss1 < last_loss1 and math.fabs(loss1 - last_loss1) <= 1e-9) or (
                            loss2 < last_loss2 and math.fabs(loss2 - last_loss2) <= 1e-9):
                print("Training done with iter_num = {}".format(i + 1))
                break
            else:
                if loss < best_loss:
                    best_loss = loss
                    self.best_pi_omega_k = self.pi_omega_k.copy()
                    self.best_gamma_k = self.gamma_k[:, :]
                    self.best_psi_k = self.psi_k[:]
                    self.best_beta_k = self.beta_k[:]

                self.patience = 5

            last_loss1 = loss1
            last_loss2 = loss2
            self.alpha *= self.alpha_decay

        for key in self.best_pi_omega_k:
            intent_dist = np.array(self.best_pi_omega_k[key], dtype=float)
            softmax_dist = soft_max(intent_dist).tolist()
            if key not in self.best_i_omega_k:
                self.best_i_omega_k[key] = softmax_dist

        for reform in self.omegas:
            for k in range(self.k_num):
                print('i {} {} = {}'.format(reform, k, self.best_i_omega_k[reform][k]))

        for k in range(self.k_num):
            print('psi {} = {}'.format(k, self.best_psi_k[k]))

        for k in range(self.k_num):
            print('beta {} = {}'.format(k, self.best_beta_k[k]))

        for k in range(self.k_num):
            for m in range(self.max_dnum):
                    print('gamma {} {} = {}'.format(m, k, self.best_gamma_k[m][k]))

        # save params
        param_dir = "./data/bootstrap_%s/%s/%s_%s_%s" % (
            self.data, self.id, self.click_model, self.metric_type, self.train_type)

        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

        np.save("%s/best_psi_k.npy" % param_dir, self.best_psi_k)
        np.save("%s/best_beta_k.npy" % param_dir, self.best_beta_k)
        np.save("%s/best_gamma_k.npy" % param_dir, self.best_gamma_k)
        with open("%s/best_pi_omega_k.json" % param_dir, "w") as fw1:
            json.dump(self.best_pi_omega_k, fw1)

    def eval(self):
        param_dir = "./data/bootstrap_%s/%s/%s_%s_%s" % (
            self.data, self.id, self.click_model, self.metric_type, self.train_type)

        best_psi_k = np.load("%s/best_psi_k.npy" % param_dir)
        best_beta_k = np.load("%s/best_beta_k.npy" % param_dir)
        best_gamma_k = np.load("%s/best_gamma_k.npy" % param_dir)
        with open("%s/best_pi_omega_k.json" % param_dir, "r") as f1:
            best_pi_omega_k = json.load(f1)

        i_omega_k = {}
        for key in best_pi_omega_k:
            intent_dist = np.array(best_pi_omega_k[key], dtype=float)
            softmax_dist = soft_max(intent_dist).tolist()
            if key not in i_omega_k:
                i_omega_k[key] = softmax_dist

        sat_list, pred_sat_list = [], []
        clicks, pred_clicks = [], []
        for test_s in self.test_data:
            if self.k_num == 1:  # w/o reform inform
                reform = 'F'
            else:
                reform = test_s[0]
            true_reform = test_s[0]
            this_clicks = test_s[1]
            usefs = test_s[2]
            sat = float(test_s[3])
            usefs = [float(usef) for usef in usefs]
            alphas = (np.exp2(usefs) - 1) / np.exp2(self.max_usefulness)

            max_clicked_pos = -1
            for j in range(self.max_dnum):
                if this_clicks[j] == 1:
                    max_clicked_pos = max(max_clicked_pos, j)

            # 1-10
            cond_clicks, ind_clicks = np.zeros((self.k_num, self.max_dnum), dtype=float), np.zeros(
                (self.k_num, self.max_dnum), dtype=float)

            cond_clicks += 1e-30
            ind_clicks += 1e-30

            sum_epsilons = np.zeros(self.max_dnum, dtype=float)
            for j in range(self.max_dnum):
                for k in range(self.k_num):
                    # conditional without k, independent with k
                    sum_epsilons[j] += i_omega_k[reform][k] * best_gamma_k[j][k]
                    cond_clicks[k][j] += alphas[j] * best_gamma_k[j][k]
                    ind_clicks[k][j] += alphas[j] * best_gamma_k[j][k]

            pred_sat_total = 0.
            for k in range(self.k_num):
                pred_sat = 0.
                for j in range(self.max_dnum):
                    pred_sat += usefs[j] * ind_clicks[k][j]
                pred_sat_total = pred_sat_total + i_omega_k[reform][k] * best_beta_k[k] * (pred_sat + best_psi_k[k])

            sat_list.append(sat)
            pred_sat_list.append(pred_sat_total)

            this_pred_clicks = np.dot(i_omega_k[reform], cond_clicks)
            this_pred_clicks = np.array(this_pred_clicks)
            this_clicks = np.array(this_clicks)
            this_loss = - np.sum(
                np.log2((this_pred_clicks + 1e-30) * this_clicks + (1 - this_pred_clicks + 1e-30) * (1 - this_clicks)))
            if this_loss < 50:
                clicks.append(this_clicks.tolist())
                pred_clicks.append(this_pred_clicks.tolist())

        clicks = np.array(clicks)  # n * 10
        pred_clicks = np.array(pred_clicks)  # n * 10
        ppl_all = clicks * np.log2(pred_clicks + 1e-30) + (1 - clicks) * np.log2(1 - pred_clicks + 1e-30)  # n * 10
        ppl_rank = np.mean(ppl_all, axis=0)  # 1 * 10
        ppl_base = np.array([2.0] * self.max_dnum)
        ppl_rank = np.power(ppl_base, -ppl_rank)
        ppl_all = np.mean(ppl_rank)

        r1, p1 = pearsonr(sat_list, pred_sat_list)
        r2, p2 = spearmanr(sat_list, pred_sat_list)
        mse = np.mean(np.square(np.array(sat_list) - np.array(pred_sat_list)))
        print("Evaluation: pearson = {}, spearman = {}, mse = {}, PPL = {}".format(r1, r2, mse, ppl_all))
        np.save("%s/sat_list.npy" % param_dir, sat_list)
        np.save("%s/pred_sat_list.npy" % param_dir, pred_sat_list)
        np.save("%s/pred_sat_list.npy" % param_dir, pred_sat_list)







