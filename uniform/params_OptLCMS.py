import numpy as np
import copy
import time
import csv

UB_CELL = 20
CMS_CELL = 2


def calculate_params(total_memory,thresholds_index,epsilon):
    epsilons = []
    deltas = []
    query_distribution = []
    data_distribution = []
    
    for g in range(1,len(thresholds_index)):
        N_g = Counts_Sum_list[thresholds_index[g]] - Counts_Sum_list[thresholds_index[g-1]]
        if N_g < 0:
            print(N_g)
        Q_g = Query_Sum_list[thresholds_index[g]] - Query_Sum_list[thresholds_index[g-1]]
        ep_g = epsilon * N / N_g
        q_g = Q_g / Q
        p_g = N_g / N
        epsilons.append(ep_g)
        query_distribution.append(q_g)
        data_distribution.append(p_g)
    
    N_g = Counts_Sum_list[-1] - Counts_Sum_list[thresholds_index[-1]]
    Q_g = Query_Sum_list[-1] - Query_Sum_list[thresholds_index[-1]]
    ep_g = epsilon * N / N_g
    q_g = Q_g / Q
    p_g = N_g / N
    epsilons.append(ep_g)
    query_distribution.append(q_g)
    data_distribution.append(p_g)

    n_G = ub_size_list[thresholds_index[0]]
    X = (total_memory - UB_CELL*n_G)/(CMS_CELL*np.exp(1))
    Y = 0
    Z = 0
    for g in range(len(query_distribution)):
        ep_g = epsilons[g]
        q_g = query_distribution[g]
        Y += 1/ep_g * np.log(q_g*ep_g)
        Z += 1/ep_g
    for g in range(len(query_distribution)):
        ep_g = epsilons[g]
        q_g = query_distribution[g]
        deltas.append(1/(q_g*ep_g)*np.exp(-(X-Y)/Z))
    return epsilons,deltas
        

def calculate_ub_size():
    # t_optionには閾値の候補
    t_option = []
    # ub_size[i]には出現回数がt_option[i]以上のユニークな要素数
    ub_size = []
    # N_list[i]には出現回数がt_option[i]以上の要素の出現頻度の合計
    N_list = []
    # Q_list[i]には出現回数がt_option[i]以上の要素のクエリの合計
    Q_list = []
    
    t_option.append(scores[0])
    ub_size.append(1)
    N_list.append(counts_data[0])
    Q_list.append(queries[0])

    for i in range(1,len(counts_data)):
        if t_option[-1] == scores[i]:
            ub_size[-1] += 1
            N_list[-1] += counts_data[i]
            Q_list[-1] += queries[i]
        else:
            t_option.append(scores[i])
            tmp = ub_size[-1] 
            ub_size.append(tmp + 1)
            tmp_N = N_list[-1]
            N_list.append(tmp_N + counts_data[i])
            tmp_Q = Q_list[-1]
            Q_list.append(tmp_Q + queries[i])

    return t_option,ub_size,N_list,Q_list

def information(upper,lower,N_A,Q_A):
        """
        スコアが[thresholds_option[lower],thresholds_option[upper])の範囲に収まる要素に対して,情報量を計算する
        """
        N_g = Counts_Sum_list[lower] - Counts_Sum_list[upper]
        Q_g = Query_Sum_list[lower] - Query_Sum_list[upper]
        p = N_g / N_A
        q = Q_g / Q_A
        if p <= 0 or q <= 0 or p > 1 or q > 1:
            raise ValueError(f"Invalid values for p ({p}) or q ({q})")
        return p * np.log(p / q)

def check_valid(ub_factor,tmp_info,upper,lower,N_A,Q_A):
    if lower == len(thresholds_option) - 1:
        kl_info = tmp_info
        p_g_dash =  (Counts_Sum_list[lower] - Counts_Sum_list[upper]) / N_A
        q_g_dash = (Query_Sum_list[lower] - Query_Sum_list[upper]) / Q_A
        delta_g = p_g_dash/q_g_dash * ub_factor * np.exp(-kl_info)
        if delta_g > 1:
            return False
        return True
    info_res = information(lower,-1,N_A,Q_A)
    kl_info = tmp_info + info_res
    p_g_dash =  (Counts_Sum_list[lower] - Counts_Sum_list[upper]) / N_A
    q_g_dash = (Query_Sum_list[lower] - Query_Sum_list[upper]) / Q_A
    delta_g = p_g_dash/q_g_dash * ub_factor * np.exp(-kl_info)

    p_res = (Counts_Sum_list[-1] - Counts_Sum_list[lower]) / N_A
    q_res = (Query_Sum_list[-1] - Query_Sum_list[lower]) / Q_A
    delta_res = p_res / q_res * ub_factor * np.exp(-kl_info)
    if delta_g > 1 or delta_res > 1:
        return False
    return True

def optimize_thresholds(epsilon,total_memory,partitions):
    min_obj = float('inf')
    min_thresholds = []

    for i in range(len(thresholds_option)):
        if total_memory - UB_CELL * ub_size_list[i] < 0:
            break
        n_G = ub_size_list[i]
        N_G = Counts_Sum_list[i]
        Q_G = Query_Sum_list[i]
        N_A = N - N_G
        Q_A = Q - Q_G
        ub_factor = np.exp(-(epsilon*(total_memory-UB_CELL*n_G)/(CMS_CELL*np.exp(1)*(1-N_G/N))))

        pre_table = [-float('inf') for _ in range(len(thresholds_option))]
        pre_thresholds = [[i] for _ in range(len(thresholds_option))]
        for j in range(i+1,len(thresholds_option)-1):
            tmp_info = information(i,j,N_A,Q_A)
            if check_valid(ub_factor,tmp_info,i,j,N_A,Q_A):
                pre_table[j] = tmp_info
            
        for part in range(2,partitions):
            new_table = [-float('inf') for _ in range(len(thresholds_option))]
            new_thresholds = [[] for _ in range(len(thresholds_option))]
            for j in range(i+2,len(thresholds_option)):
                max_info = pre_table[j]
                max_th_index = None
                for k in range(i+1,j):
                    tmp_info = pre_table[k] + information(k,j,N_A,Q_A)
                    if tmp_info > max_info and check_valid(ub_factor,tmp_info,k,j,N_A,Q_A):
                        max_info = tmp_info
                        max_th_index = k
                new_table[j] = max_info
                if max_th_index:
                    new_thresholds[j] = copy.deepcopy(pre_thresholds[max_th_index])
                    new_thresholds[j].append(max_th_index)
                else :
                    new_thresholds[j] = copy.deepcopy(pre_thresholds[j])
            pre_table = new_table
            pre_thresholds= copy.deepcopy(new_thresholds)
        
        tmp_kl_info = pre_table[-1]
        tmp_obj = ub_factor*np.exp(-tmp_kl_info)
        tmp_thresholds = copy.deepcopy(pre_thresholds[-1])
        if min_obj > tmp_obj:
            min_obj = tmp_obj
            min_thresholds = tmp_thresholds
    
    return min_obj,min_thresholds
        
def order_y_wkey(y, results, key):
    """ Order items based on the scores in results """
    print('loading results from %s' % results)
    results = np.load(results)
    pred_prob = results[key].astype(int).squeeze()
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

data = np.load('../data/query_counts_day_0005.npz')
c_data = data['counts'] #降順にソートされている。indexに対応する出現頻度を要素として持つ
valid_result_path = "../paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz"
counts_data, scores = order_y_wkey(c_data,valid_result_path,"valid_output")

N = np.sum(counts_data)
queries = [1 for _ in range(len(counts_data)) ] #クエリ。indexに対応する出現頻度を要素として持つ
Q = np.sum(queries)
thresholds_option,ub_size_list,Counts_Sum_list,Query_Sum_list = calculate_ub_size()



m = 10**5
totalMemory_values = [m *i for i in range(1,10)]
results = {
    'epsilon': [],
    'total_memory': [],
    'min_obj': [],
    'min_thresholds': [],
    'cms_epsilons' : [],
    'cms_deltas' : []

}
time_list = []

for total_memory in totalMemory_values:
    start_time = time.time()
    epsilon = 2*np.exp(1)/total_memory
    min_obj, min_ths_index = optimize_thresholds(epsilon, total_memory,10)

    min_ths = []
    for i in min_ths_index:
        min_ths.append(thresholds_option[i])

    cms_epsilons,cms_deltas = calculate_params(total_memory,min_ths_index,epsilon)

    results['epsilon'].append(epsilon)
    results['total_memory'].append(total_memory)
    results['min_obj'].append(min_obj)
    results['min_thresholds'].append(min_ths)
    results['cms_epsilons'].append(cms_epsilons)
    results['cms_deltas'].append(cms_deltas)

    end_time = time.time()  # 処理終了時間を記録
    elapsed_time = end_time - start_time
    time_list.append(elapsed_time)
    print(elapsed_time)

    

results['epsilon'] = np.array(results['epsilon'])
results['total_memory'] = np.array(results['total_memory'])
results['min_obj'] = np.array(results['min_obj'])
results['min_thresholds'] = np.array(results['min_thresholds'], dtype=object)
results['cms_epsilons'] = np.array(results['cms_epsilons'],dtype=object)
results['cms_deltas'] = np.array(results['cms_deltas'],dtype=object)

np.savez('params/OptLCMS_best_params.npz', **results)
with open('time_list_OptLCMS.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['total_memory', 'elapsed_time'])  # ヘッダーを書く
    for total_memory, elapsed_time in zip(totalMemory_values, time_list):
        writer.writerow([total_memory, elapsed_time])  # 各total_memoryと対応するelapsed_timeを書き込む