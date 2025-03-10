import numpy as np
import copy
import time
import csv
UB_CELL = 20
CMS_CELL = 2

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

def optimize_thresholds(epsilon,total_memory):
    min_obj = float('inf')
    min_thresholds = []
    min_epsilon = None
    min_delta = None

    for i in range(len(thresholds_option)):
        if total_memory - UB_CELL * ub_size_list[i] < 0:
            print(thresholds_option[i],ub_size_list[i])
            break
        n_G = ub_size_list[i]
        N_G = Counts_Sum_list[i]
        Q_G = Query_Sum_list[i]
        N_A = N - N_G
        Q_A = Q - Q_G
        ub_factor = np.exp(-(epsilon*(total_memory-UB_CELL*n_G)/(CMS_CELL*np.exp(1)*(1-N_G/N))))
        cms_epsilon = N*epsilon/N_A
        cms_delta = ub_factor
        tmp_obj = Q_A / Q * cms_delta
        
        if min_obj > tmp_obj:
            min_obj = tmp_obj
            min_thresholds = i
            min_epsilon = cms_epsilon
            min_delta = cms_delta
    
    return min_obj,min_thresholds,min_epsilon,min_delta
        
def order_y_wkey(y, results, key):
    """ Order items based on the scores in results """
    print('loading results from %s' % results)
    results = np.load(results)
    pred_prob = results[key].astype(int).squeeze()
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

data = np.load('../data/query_counts_day_0005.npz')
c_data = data['counts'] 
valid_result_path = "../paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz"
counts_data, scores = order_y_wkey(c_data,valid_result_path,"valid_output")

time_list = []
N = np.sum(counts_data)
queries = counts_data 
Q = np.sum(queries)
thresholds_option,ub_size_list,Counts_Sum_list,Query_Sum_list = calculate_ub_size()
m = 10**5
totalMemory_values = [m *i for i in range(1,10)]
results = {
    'epsilon': [],
    'total_memory': [],
    'min_obj': [],
    'min_threshold': [],
    'cms_epsilon' : [],
    'cms_delta' : []
}

for total_memory in totalMemory_values:
    start_time = time.time()
    epsilon = CMS_CELL*np.exp(1)/total_memory
    print(total_memory)
    min_obj, min_th_index,min_cms_epsilon,min_cms_delta = optimize_thresholds(epsilon, total_memory)

    min_th = thresholds_option[min_th_index]

    results['epsilon'].append(epsilon)
    results['total_memory'].append(total_memory)
    results['min_obj'].append(min_obj)
    results['min_threshold'].append(min_th)
    results['cms_epsilon'].append(min_cms_epsilon)
    results['cms_delta'].append(min_cms_delta)

    end_time = time.time()  # 処理終了時間を記録
    elapsed_time = end_time - start_time
    time_list.append(elapsed_time)

results['epsilon'] = np.array(results['epsilon'])
results['total_memory'] = np.array(results['total_memory'])
results['min_obj'] = np.array(results['min_obj'])
results['min_threshold'] = np.array(results['min_threshold'])
results['cms_epsilon'] = np.array(results['cms_epsilon'])
results['cms_delta'] = np.array(results['cms_delta'])

np.savez('params/OptLCMS_best_params.npz', **results)

with open('time_list_OptLCMS.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['total_memory', 'elapsed_time']) 
    for total_memory, elapsed_time in zip(totalMemory_values, time_list):
        writer.writerow([total_memory, elapsed_time])  