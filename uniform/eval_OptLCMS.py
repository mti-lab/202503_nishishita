import numpy as np
import math
import random 
import pandas as pd

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
        


def cut_off_index(thresholds):
    """
    thresholdsに基づいてscoresのインデックスを決定する関数
    """
    index = []
    tmp_index = 0
    for i in range(len(scores)):
        # thresholdsの現在のインデックスより小さいスコアが見つかるたびにインデックスを追加
        if thresholds[tmp_index] > scores[i]:
            index.append(i)
            tmp_index += 1
            if tmp_index >= len(thresholds):  # thresholdsをすべて使った場合は終了
                break
    return index

def partition_items(thresholds):
    """
    thresholdsに基づいてitemsをパーティションに分ける関数
    """
    cms_group = []

    c_index = cut_off_index(thresholds)
    ub_index = (0,c_index[0])
    
    
    for i in range(1, len(c_index)):
        cms_group.append((c_index[i-1],c_index[i]))
    
    if c_index:
        cms_group.append((c_index[-1],len(scores)))

    return ub_index,cms_group

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


def evaluate_optlcms(thresholds,epsilon,cms_epsilons,cms_deltas):
    ub_index, cms_group = partition_items(thresholds)

    ub_memory = (ub_index[1] - ub_index[0])*UB_CELL
    ub_ratio = ub_index[1] / len(scores) #ubに記録される要素の割合
    cms_memory = 0
    false_rate = 0
    error = 0
    for g in range(len(cms_group)):
        count_data = counts_data[cms_group[g][0]:cms_group[g][1]]
        cms = CMS(cms_epsilons[g],cms_deltas[g])
        # cmsでカウント->推定値を返す
        esti_data = cms.count_items(count_data)
        # 失敗確率と誤差
        for i in range(len(count_data)):
            error += (esti_data[i]-count_data[i])*queries[cms_group[g][0]+i]/Q
            if esti_data[i]-count_data[i] > epsilon*N:
                false_rate += queries[cms_group[g][0]+i]/Q
        # メモリ使用量
        cms_memory += cms.memory_usage()
    total_memory_usage = cms_memory + ub_memory
    ub_memory_ratio = ub_memory / total_memory_usage

    return ub_ratio,ub_memory_ratio,total_memory_usage, false_rate, error

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
    
    params = np.load("params/OptLCMS_best_params.npz", allow_pickle=True)
    
    c_data = get_data_aol_query(test_data_path)
    counts_data, scores = order_y_wkey(c_data,test_result_path,"test_output")

    queries = [1 for _ in range(len(counts_data))]

    Q = np.sum(queries)
    N = np.sum(counts_data)

    results = []
    for i in range(len(params["epsilon"])):
        ep = params["epsilon"][i]
        total_memory = params["total_memory"][i]
        thresholds = params["min_thresholds"][i]
        epsilons_distribution = params["cms_epsilons"][i]
        deltas_distribution = params["cms_deltas"][i]
        memory_results = []
        false_rate_results = []
        error_results = []
        ub_ratios = []
        ub_memory_ratios = []
        
        for _ in range(TRIAL):
            u,ub_m,m, f, e = evaluate_optlcms(thresholds, ep, epsilons_distribution,deltas_distribution)
            memory_results.append(m)
            false_rate_results.append(f)
            error_results.append(e)
            ub_ratios.append(u)
            ub_memory_ratios.append(ub_m)
        
        avg_m = np.mean(memory_results)
        avg_f = np.mean(false_rate_results)
        avg_e = np.mean(error_results)
        avg_u = np.mean(ub_ratios)
        avg_ub_m = np.mean(ub_memory_ratios)
        
        print(f"Total Memory: {total_memory}, Epsilon: {ep}")
        print(f"Average Memory Usage: {avg_m}")
        print(f"Average False Rate: {avg_f}")
        print(f"Average Error: {avg_e}")
        print(f"UB Ratio:{avg_u}")
        print(f"ub memory ratio:{avg_ub_m}")

        results.append({
            'Epsilon': ep,
            'Memory Usage': avg_m,
            'False Rate': avg_f,
            'Error': avg_e,
            'UB Ratio': avg_u,
            "UB Memory Ratio":avg_ub_m
        })

    df = pd.DataFrame(results)

    df.to_csv('results/optlcms_results.csv', index=False)