# -*- coding: utf-8 -*-
import numpy as np

class PMF(object):
    """
    Probabilistic Matrix Factorization (PMF) implementation with stochastic gradient descent optimization.
    """

    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        """
        Initializes PMF parameters.

        Args:
            num_feat (int): Number of latent features.
            epsilon (float): Learning rate.
            _lambda (float): L2 regularization term.
            momentum (float): Momentum for gradient descent.
            maxepoch (int): Maximum number of epochs.
            num_batches (int): Number of batches per epoch.
            batch_size (int): Number of samples per batch.
        """
        self.num_feat = num_feat
        self.epsilon = epsilon
        self._lambda = _lambda
        self.momentum = momentum
        self.maxepoch = maxepoch
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.w_Item = None
        self.w_User = None
        self.rmse_train = []
        self.rmse_test = []

    def fit(self, train_vec, test_vec):
        """
        Trains the PMF model using stochastic gradient descent.

        Args:
            train_vec (np.ndarray): Training data as [user_id, item_id, rating].
            test_vec (np.ndarray): Test data as [user_id, item_id, rating].
        """
        self.mean_inv = np.mean(train_vec[:, 2])
        pairs_train = train_vec.shape[0]
        pairs_test = test_vec.shape[0]
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1

        if self.w_Item is None:
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)
            self.w_Item_inc = np.zeros((num_item, self.num_feat))
            self.w_User_inc = np.zeros((num_user, self.num_feat))

        while self.epoch < self.maxepoch:
            self.epoch += 1
            shuffled_order = np.random.permutation(train_vec.shape[0])

            for batch in range(self.num_batches):
                batch_idx = shuffled_order[batch * self.batch_size:(batch + 1) * self.batch_size]
                batch_UserID = train_vec[batch_idx, 0].astype('int32')
                batch_ItemID = train_vec[batch_idx, 1].astype('int32')

                pred_out = np.sum(self.w_User[batch_UserID] * self.w_Item[batch_ItemID], axis=1)
                rawErr = pred_out - train_vec[batch_idx, 2] + self.mean_inv

                Ix_User = 2 * rawErr[:, None] * self.w_Item[batch_ItemID] + self._lambda * self.w_User[batch_UserID]
                Ix_Item = 2 * rawErr[:, None] * self.w_User[batch_UserID] + self._lambda * self.w_Item[batch_ItemID]

                dw_User = np.zeros_like(self.w_User)
                dw_Item = np.zeros_like(self.w_Item)
                np.add.at(dw_User, batch_UserID, Ix_User)
                np.add.at(dw_Item, batch_ItemID, Ix_Item)

                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size
                self.w_Item -= self.w_Item_inc
                self.w_User -= self.w_User_inc

            pred_out = np.sum(self.w_User[train_vec[:, 0].astype('int32')] * self.w_Item[train_vec[:, 1].astype('int32')], axis=1)
            rawErr = pred_out - train_vec[:, 2] + self.mean_inv
            obj = np.linalg.norm(rawErr) ** 2 + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)
            self.rmse_train.append(np.sqrt(obj / pairs_train))

            pred_out = np.sum(self.w_User[test_vec[:, 0].astype('int32')] * self.w_Item[test_vec[:, 1].astype('int32')], axis=1)
            rawErr = pred_out - test_vec[:, 2] + self.mean_inv
            self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))
            print('Training RMSE: %f, Test RMSE: %f' % (self.rmse_train[-1], self.rmse_test[-1]))

    def predict(self, invID):
        """
        Predicts item ratings for a specific user.

        Args:
            invID (int): User ID.

        Returns:
            np.ndarray: Predicted ratings for all items.
        """
        return np.dot(self.w_Item, self.w_User[int(invID)]) + self.mean_inv

    def set_params(self, parameters):
        """
        Updates PMF parameters from a dictionary.

        Args:
            parameters (dict): Dictionary containing parameter values.
        """
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", self.num_feat)
            self.epsilon = parameters.get("epsilon", self.epsilon)
            self._lambda = parameters.get("_lambda", self._lambda)
            self.momentum = parameters.get("momentum", self.momentum)
            self.maxepoch = parameters.get("maxepoch", self.maxepoch)
            self.num_batches = parameters.get("num_batches", self.num_batches)
            self.batch_size = parameters.get("batch_size", self.batch_size)

    def topK(self, test_vec, k=10):
        """
        Computes top-K precision and recall for the test set.

        Args:
            test_vec (np.ndarray): Test data as [user_id, item_id, rating].
            k (int): Number of top recommendations to consider.

        Returns:
            tuple: Precision and recall averaged over users.
        """
        inv_lst = np.unique(test_vec[:, 0])
        pred = {inv: np.argsort(self.predict(inv))[-k:] for inv in inv_lst}
        intersection_cnt = {inv: np.sum(test_vec[test_vec[:, 0] == inv, 1] == pred[inv]) for inv in inv_lst}
        invPairs_cnt = np.bincount(test_vec[:, 0].astype('int32'))

        precision_acc = sum(intersection_cnt.get(inv, 0) / k for inv in inv_lst) / len(inv_lst)
        recall_acc = sum(intersection_cnt.get(inv, 0) / invPairs_cnt[int(inv)] for inv in inv_lst) / len(inv_lst)
        return precision_acc, recall_acc