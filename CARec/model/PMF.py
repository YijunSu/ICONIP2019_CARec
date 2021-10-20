# -*- coding: utf-8 -*-
import time
import math
import numpy as np


class PMF(object):
    def __init__(self, K=30, alpha_U=0.2, alpha_L=0.2):
        self.K = K
        self.alpha_U = alpha_U
        self.alpha_L = alpha_L
        self.U, self.L = None, None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...", )
        np.save(path + "U", self.U)
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...", )
        self.U = np.load(path + "U.npy")
        self.L = np.load(path + "L.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def train(self, sparse_check_in_matrix, max_iters=50, learning_rate=1e-4):
        ctime = time.time()
        print("Training PMF...", )
        K = self.K
        lambda_l = 0.1 / self.alpha_U
        lambda_u = 0.1 / self.alpha_L

        F = sparse_check_in_matrix
        M, N = sparse_check_in_matrix.shape
        U = np.random.normal(0, self.alpha_U, (M, K))  # mean-Standard Deviation-shape
        L = np.random.normal(0, self.alpha_L, (N, K))

        F = F.tocoo()
        entry_index = list(zip(F.row, F.col))
        F = F.tocsr()
        F_dok = F.todok()

        def Indicator(i, j):
            if (sparse_check_in_matrix[i, j] == 1.0):
                return 1.0
            else:
                return 0.0

        tau = 10
        last_loss = float('Inf')
        for iters in range(max_iters):
            F_Y = F_dok.copy()
            for i, j in entry_index:
                F_Y[i, j] = F_dok[i, j] - U[i].dot(L[j])
            F_Y = F_Y.tocsr()

            learning_rate_k = learning_rate * tau / (tau + iters)
            U += learning_rate_k * (F_Y.dot(L) - lambda_l * U)
            L += learning_rate_k * ((F_Y.T).dot(U) - lambda_u * L)

            loss = 0.0
            for i, j in entry_index:
                loss += 0.5 * ((F_dok[i, j] - U[i].dot(L[j])) ** 2 + lambda_u * np.square(
                    U[i]).sum() + lambda_l * np.square(L[j]).sum())

            print('Iteration:', iters, 'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.U, self.L = U, L

    def predict(self, uid, lid, sigmoid=False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[uid].dot(self.L[lid])))
        return self.U[uid].dot(self.L[lid])
