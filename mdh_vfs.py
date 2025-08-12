import numpy as np
from collections import defaultdict
from scipy.stats import norm

class MDHistogramDyn:
    def __init__(self):
        self.referValue = {}
        self.referNum = None
        self.mDegree_each = {}
        self.mDegree_sum = {}
        self.mDegree = {}

    def getMDH(self, data, referNum, referValue):
        data = np.array(data)
        if np.isscalar(referNum):
            referNum = [referNum]

        self.referNum = referNum
        self.referValue = self._computeReferValue(data, referNum, referValue)
        self._computeCredibility(data, self.referValue, referNum)

    def _computeReferValue(self, data, referNum, method):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        _, attriNum = data.shape

        referValue = {}

        for i in range(attriNum):
            minVal = np.min(data[:, i])
            maxVal = np.max(data[:, i])
            num = referNum[i]

            if num <= 1:
                referValue[i] = np.array([minVal])
                continue

            if method == 'linear':
                referValue[i] = np.linspace(minVal, maxVal, num)
            elif method == 'log':
                mid = np.logspace(0.1, 1, num - 2, base=10)
                mid = (mid - mid.min()) / (mid.max() - mid.min())
                vals = minVal + mid * (maxVal - minVal)
                referValue[i] = np.concatenate(([minVal], vals, [maxVal]))
            elif method == 'center':
                x = np.linspace(0, 1, num - 2)
                mid = (1 - np.cos(np.pi * x)) / 2
                vals = minVal + mid * (maxVal - minVal)
                referValue[i] = np.concatenate(([minVal], vals, [maxVal]))

        return referValue

    def _computeCredibility(self, data, referValue, referNum):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        sampleNum, attriNum = data.shape

        self.mDegree_each = {}
        for i in range(attriNum):
            self.mDegree_each[i] = np.zeros((sampleNum, referNum[i]))

        for k in range(sampleNum):
            for i in range(attriNum):
                self.mDegree_each[i][k,:] = self._MDegree_single(data[k, i], referValue[i])

        self._MDegree()

    def _MDegree_single(self, currentVal, referVals):
        singleMDegree = np.zeros(len(referVals))

        if currentVal <= referVals[0]:
            singleMDegree[0] = 1.0
            return singleMDegree
        elif currentVal >= referVals[-1]:
            singleMDegree[-1] = 1.0
            return singleMDegree

        for j in range(len(referVals) - 1):
            if referVals[j] <= currentVal <= referVals[j + 1]:
                denominator = referVals[j + 1] - referVals[j]
                if denominator == 0:
                    if currentVal == referVals[j]:
                        singleMDegree[j] = 1.0
                else:
                    weight_j = (referVals[j + 1] - currentVal) / denominator
                    weight_j_plus_1 = 1 - weight_j
                    singleMDegree[j] = weight_j
                    singleMDegree[j + 1] = weight_j_plus_1
                break

        return singleMDegree

    def _MDegree(self):
        attriNum = len(self.referNum)

        for i in range(attriNum):
            self.mDegree_sum[i] = np.sum(self.mDegree_each[i], axis=0)

            total = np.sum(self.mDegree_sum[i])
            if total > 0:
                self.mDegree[i] = self.mDegree_sum[i] / total
            else:
                self.mDegree[i] = np.zeros(self.referNum[i])

    def getMDH_new_dyn(self, data, referNum, referValue):
        data = np.array(data)
        if np.isscalar(referNum):
            referNum = [referNum]

        self.referNum = referNum
        self.referValue = referValue

        if not self.mDegree_sum:
            self._computeCredibility(data, self.referValue, referNum)
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        new_data = data[-1]
        attriNum = len(self.referNum)

        for i in range(attriNum):
            self.mDegree_each[i] = self.mDegree_each[i][1:, :]

            new_contribution = self._MDegree_single(new_data[i], self.referValue[i])
            new_contribution = new_contribution.reshape(1, -1)
            self.mDegree_each[i] = np.vstack([self.mDegree_each[i], new_contribution])

        self._MDegree()

    def get_credibility_matrix(self):
        return self.mDegree

    def get_projection_matrix(self):
        return self.mDegree_sum

    def get_reference_values(self):
        return self.referValue

class MDHVFS:
    def __init__(self, hist_window_size, new_window_size, referValue,
                 epsilon, referNum, warn, update_thre):
        self.referValue = referValue
        self.hist_window_size = hist_window_size
        self.new_window_size = new_window_size
        self.hist_window = defaultdict(list)
        self.new_window = defaultdict(list)
        self.hist_window_mdh = defaultdict(list)
        self.new_window_mdh = defaultdict(list)
        self.referNum = referNum

        self.epsilon = {}
        self.default_epsilon = epsilon
        self.probabilitylist = defaultdict(list)

        self.uni_fea = []
        self.instances_seen = 0
        self.is_collecting_hist = {}
        self.default_is_collecting_hist = True

        self.warning_num = {}
        self.safe_num = {}
        self.warn = warn
        self.update_thre = update_thre

    def _initialize_feature(self, feature_id):
        self.epsilon[feature_id] = self.default_epsilon
        self.is_collecting_hist[feature_id] = self.default_is_collecting_hist

        self.hist_window_mdh[feature_id] = MDHistogramDyn()
        self.new_window_mdh[feature_id] = MDHistogramDyn()

        self.warning_num[feature_id] = 0
        self.safe_num[feature_id] = 0

    def detect(self, x):
        self.instances_seen += 1

        idx_t = np.where(~np.isnan(x))[0]
        x = x[idx_t]

        if self.instances_seen == 1:
            self.uni_fea = idx_t
        else:
            self.uni_fea = np.union1d(self.uni_fea, idx_t)

        drift_detected = False
        drift_features = []
        drift_position = []

        for i in range(len(x)):
            feature_id = idx_t[i]
            value = x[i]

            if feature_id not in self.epsilon:
                self._initialize_feature(feature_id)

            if self.is_collecting_hist[feature_id]:
                self.hist_window[feature_id].append(value)
                if len(self.hist_window[feature_id]) == self.hist_window_size:
                    self.is_collecting_hist[feature_id] = False
                    self.hist_window_mdh[feature_id].getMDH(self.hist_window[feature_id], self.referNum, self.referValue)
            else:
                if len(self.new_window[feature_id]) == self.new_window_size:
                    self.new_window[feature_id].pop(0)
                self.new_window[feature_id].append(value)

                if len(self.new_window[feature_id]) == self.new_window_size:
                    self.new_window_mdh[feature_id].getMDH_new_dyn(self.new_window[feature_id],
                                                               self.hist_window_mdh[feature_id].referNum,
                                                               self.hist_window_mdh[feature_id].referValue)
                    hellinger = self._calculate_hellinger(feature_id)

                    self.probabilitylist[feature_id].append(hellinger)
                    if len(self.probabilitylist[feature_id]) > 100:
                        self.probabilitylist[feature_id] = self.probabilitylist[feature_id][-100:]

                    self._update_epsilon(feature_id)

                    if hellinger > self.epsilon[feature_id]:
                        self.warning_num[feature_id] += 1
                        self.safe_num[feature_id] = 0
                    else:
                        self.safe_num[feature_id] += 1
                        if self.safe_num[feature_id] > self.warn and self.warning_num[feature_id] > 0:
                            self.warning_num[feature_id] = 0
                            self.safe_num[feature_id] = 0

                    if self.warning_num[feature_id] >= self.warn:
                        self._handle_drift(feature_id)
                        drift_detected = True
                        drift_features.append(feature_id)
                        drift_position = self.instances_seen

        return drift_detected, drift_features, drift_position

    def _calculate_hellinger(self, feature_id):
        new_data = self.new_window_mdh[feature_id].mDegree[0]
        hist_data = self.hist_window_mdh[feature_id].mDegree[0]

        h_squared = 0.5 * np.sum((np.sqrt(hist_data) - np.sqrt(new_data)) ** 2)
        return np.sqrt(h_squared)

    def _update_epsilon(self, feature_id):
        hellinger = self.probabilitylist[feature_id]

        if len(hellinger) >= self.update_thre:
            recent_values = hellinger[-self.update_thre:]
            mean_hellinger = np.mean(recent_values)
            std_hellinger = np.std(recent_values)

            new_threshold = mean_hellinger + 3 * std_hellinger

            min_threshold = 0.1
            max_threshold = 1.0
            self.epsilon[feature_id] = np.clip(new_threshold, min_threshold, max_threshold)

    def _handle_drift(self, feature_id):
        self.hist_window[feature_id] = self.new_window[feature_id].copy()

        self.is_collecting_hist[feature_id] = True

        self.new_window[feature_id] = []
        self.hist_window_mdh[feature_id] = MDHistogramDyn()
        self.new_window_mdh[feature_id] = MDHistogramDyn()

        self.probabilitylist[feature_id] = []
        self.epsilon[feature_id] = self.default_epsilon

        self.warning_num[feature_id] = 0
        self.safe_num[feature_id] = 0