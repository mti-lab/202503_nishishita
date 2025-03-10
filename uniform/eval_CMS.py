import numpy as np
from dataclasses import dataclass 
import math
import random 
import pandas as pd

CMS_CELL = 2
TRIAL = 10

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
        


if __name__ == "__main__":  
    data = np.load("../data/query_counts_day_0050.npz")
    items = data["counts"]
    queries = [1 for i in range(len(items))]

    results = []

    Q = np.sum(queries)
    N = np.sum(items)
    
    
    memories = [(i+1)*10**5 for i in range(12)]
    for total_memory in memories:
        memory_results = []
        error_results = []
        for _ in range(TRIAL):
            err = None
            memory_usage = None
            for d in range(1,6):
                w = total_memory / (d * CMS_CELL)
                epsilon = np.exp(1)/w
                delta = np.exp(-d)
                cms = CMS(epsilon,delta)
                estimated_counts = cms.count_items(items)
                err_d = 0
                for i in range(len(items)):
                    err_d += (estimated_counts[i] - items[i])*queries[i]/Q
                memory_usage_d = cms.memory_usage()
            if err == None or err / memory_usage > err_d / memory_usage_d:
                err = err_d
                memory_usage = memory_usage_d
            memory_results.append(memory_usage)
            error_results.append(err)
        avg_m = np.mean(memory_results)
        avg_e = np.mean(error_results)
        results.append({
            'Memory Usage': avg_m,
            'Error': avg_e
        })

df = pd.DataFrame(results)

df.to_csv('results/CMS_result.csv', index=False)
