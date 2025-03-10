import numpy as np
import math
import random 
import pandas as pd
import copy

UB_CELL = 20
CMS_CELL = 2
TRIAL = 50

class CMS:
    def __init__(self,epsilon,delta):
        self.epsilon = epsilon
        self.width = math.ceil(np.exp(1)/epsilon)
        self.delta = delta
        self.depth = math.ceil(np.log(1/delta))
        self.sketch_table = [[0 for i in range(self.width)] for j in range(self.depth)]
        self.hash_maps = None

    def count_items(self,count_data):
        self.hash_maps = [[random.randint(0,self.width-1) for i in range(len(count_data))] for j in range(self.depth)]

        for i in range(len(count_data)):
            for j in range(self.depth):
                self.sketch_table[j][self.hash_maps[j][i]] += count_data[i]
        
        estimation_count = [0]*len(count_data)
        for i in range(len(count_data)):
            tmp = self.sketch_table[0][self.hash_maps[0][i]]
            for j in range(1,self.depth):
                if tmp > self.sketch_table[j][self.hash_maps[j][i]]:
                    tmp = self.sketch_table[j][self.hash_maps[j][i]]
            estimation_count[i] = tmp
        return estimation_count

    def memory_usage(self):
        return self.width*self.depth*CMS_CELL
        
def cut_off_index(threshold):
    """
    thresholdsに基づいてscoresのインデックスを決定する関数
    """
    index = 0
    for i in range(len(scores)):
        if threshold > scores[i]:
            index = i
            break
    return index

def partition_items(threshold):
    """
    thresholdsに基づいてitemsをパーティションに分ける関数
    """
    c_index = cut_off_index(threshold)
    ub_index = (0,c_index)
    cms_index = c_index

    return ub_index,cms_index

def calculate_A(queries,epsilons,total_memory,n_G):
    X = (total_memory-UB_CELL*n_G)/(CMS_CELL*np.exp(1))
    Y = 0
    for g in range(len(queries)):
        Y += 1/epsilons[g]*np.log(queries[g]*epsilons[g])
    Z = 0
    for g in range(len(epsilons)):
        Z += 1/epsilons[g]
    A = (X-Y)/Z
    return A

def evaluate_lcms(thresholds,epsilon,cms_epsilon,cms_delta):
    ub_index, cms_index = partition_items(thresholds)

    ub_memory = (ub_index[1] - ub_index[0])*UB_CELL
    ub_ratio = ub_index[1] / len(scores)
    cms_memory = 0
    false_rate = 0
    error = 0
    count_data = counts_data[cms_index:]
    cms = CMS(cms_epsilon,cms_delta)
    # cmsでカウント->推定値を返す
    esti_data = cms.count_items(count_data)
    # 失敗確率と誤差
    for i in range(len(count_data)):
        error += (esti_data[i]-count_data[i])*queries[cms_index+i]/Q
        if esti_data[i]-count_data[i] > epsilon*N:
            false_rate += queries[cms_index+i]/Q
    # メモリ使用量
    cms_memory += cms.memory_usage()
    total_memory_usage = cms_memory + ub_memory
    if cms_memory < 0 or ub_memory < 0:
        exit(1)
    ub_memory_usage_ratio = ub_memory / total_memory_usage

    return ub_ratio, ub_memory_usage_ratio,total_memory_usage, false_rate, error


def get_data_aol_query(data_path):
    data = np.load(data_path)
    q = data["queries"]
    counts = data["counts"]
    assert len(q) == len(counts)
    return counts

def order_y_wkey(y, results, key):
    """ Order items based on the scores in results """
    print('loading results from %s' % results)
    results = np.load(results)
    pred_prob = results[key].astype(int).squeeze()
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

if __name__ == "__main__":  
    test_result_path = "../paper_predictions/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz"
    test_data_path = "../data/query_counts_day_0050.npz"
    params = np.load("params/LCMS_best_params.npz", allow_pickle=True)
    c_data = get_data_aol_query(test_data_path)
    counts_data, scores = order_y_wkey(c_data,test_result_path,"test_output")
    queries = copy.deepcopy(counts_data)

    Q = np.sum(queries)
    N = np.sum(counts_data)
    if Q != N:
        assert("Error")

    results = []
    for i in range(len(params["epsilons"])):
        ep = params["epsilons"][i]
        threshold = params["best_thresholds"][i]
        cms_epsilon = params["cms_epsilons"][i]
        cms_delta = params["cms_deltas"][i]
        ub_ratios = []
        memory_results = []
        false_rate_results = []
        error_results = []
        ub_m_ratios = []
        
        for _ in range(TRIAL):
            u,ub_m,m, f, e = evaluate_lcms(threshold, ep,cms_epsilon,cms_delta)
            ub_ratios.append(u)
            memory_results.append(m)
            false_rate_results.append(f)
            error_results.append(e)
            ub_m_ratios.append(ub_m)
        
        avg_u = np.mean(ub_ratios)
        avg_m = np.mean(memory_results)
        avg_f = np.mean(false_rate_results)
        avg_e = np.mean(error_results)
        avg_ub_m = np.mean(ub_m_ratios)
        
        print(f"Average Memory Usage: {avg_m}")
        print(f"Average False Rate: {avg_f}")
        print(f"Average Error: {avg_e}")
        print(f"UB Ratio: {avg_u}")
        print(f"UB Memory Usage Ratio: {avg_ub_m}")

        results.append({
            'Epsilon': ep,
            'Memory Usage': avg_m,
            'False Rate': avg_f,
            'Error': avg_e,
            'UB Ratio': avg_u,
            'UB Memmory Ratio':avg_ub_m
        })

    df = pd.DataFrame(results)

    df.to_csv('results/LCMS_result.csv', index=False)
