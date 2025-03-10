import numpy as np
import math
import random
import copy
import time
import csv

UB_CELL = 20
CMS_CELL = 2


class CMS:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.width = math.ceil(np.exp(1)/epsilon)
        self.delta = delta
        self.depth = math.ceil(np.log(1/delta))
        self.sketch_table = [[0 for i in range(self.width)] for j in range(self.depth)]
        self.hash_maps = None

    def count_items(self, count_data):
        self.hash_maps = [[random.randint(0, self.width - 1) for i in range(len(count_data))] for j in range(self.depth)]

        for i in range(len(count_data)):
            for j in range(self.depth):
                self.sketch_table[j][self.hash_maps[j][i]] += count_data[i]
        
        estimation_count = [0] * len(count_data)
        for i in range(len(count_data)):
            tmp = self.sketch_table[0][self.hash_maps[0][i]]
            for j in range(1, self.depth):
                if tmp > self.sketch_table[j][self.hash_maps[j][i]]:
                    tmp = self.sketch_table[j][self.hash_maps[j][i]]
            estimation_count[i] = tmp
        return estimation_count

    def memory_usage(self):
        return self.width * self.depth * CMS_CELL
    

def order_y_wkey(y, results, key):
    """ Order items based on the scores in results """
    print('loading results from %s' % results)
    results = np.load(results)
    pred_prob = results[key].astype(int).squeeze()
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

total_memory = [i * 10**5 for i in range(1, 12)]
d_list = [1, 2, 3, 4]  # CMSのdepth list
valid_data_path = "../data/query_counts_day_0005.npz"
valid_data_result_path = "../paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz"
data = np.load(valid_data_path)
c_data = data['counts']
counts_data, scores = order_y_wkey(c_data, valid_data_result_path, "valid_output")
queries = copy.deepcopy(counts_data)
Q = np.sum(queries)

best_thresholds = []
best_deltas = []
best_epsilons = []
epsilons = []
time_list = []

for M in total_memory:
    start_time = time.time()
    ep = CMS_CELL*np.exp(1)/M
    max_thresholds_index = min(M // UB_CELL, len(scores))
    thresholds_index_list = [int(0.1 * max_thresholds_index * i) for i in range(1, 9)]
    min_err = None
    best_delta = None
    best_epsilon = None
    best_threshold = None
    
    for t_index in thresholds_index_list:
        cms_counts_data = counts_data[t_index:]
        for d in d_list:
            delta = np.exp(-d)
            w = (M - t_index * UB_CELL) / (d * CMS_CELL)
            epsilon = np.exp(1) / w
            cms = CMS(epsilon, delta)
            estimated_counts = cms.count_items(counts_data)
            err_d = 0
            for i in range(len(counts_data)):
                err_d += (estimated_counts[i] - counts_data[i]) * queries[i] / Q
            if min_err is None or min_err > err_d:
                min_err = err_d
                best_delta = delta
                best_epsilon = epsilon
                best_threshold = scores[t_index]
    
    best_epsilons.append(best_epsilon)
    best_deltas.append(best_delta)
    best_thresholds.append(best_threshold)
    epsilons.append(ep)

    end_time = time.time()  
    elapsed_time = end_time - start_time
    time_list.append(elapsed_time)

np.savez('params/LCMS_best_params.npz', 
         epsilons=np.array(epsilons), 
         cms_epsilons=np.array(best_epsilons), 
         cms_deltas=np.array(best_deltas), 
         best_thresholds=np.array(best_thresholds))

with open('time_list_LCMS.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['total_memory', 'elapsed_time'])  # ヘッダーを書く
    for total_memory, elapsed_time in zip(total_memory, time_list):
        writer.writerow([total_memory, elapsed_time])  # 各total_memoryと対応するelapsed_timeを書き込む