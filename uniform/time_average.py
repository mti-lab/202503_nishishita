import pandas as pd

df = pd.read_csv('time_list_LCMS.csv')
average_elapsed_time = df['elapsed_time'].mean()
print(f"LCMS Average elapsed time: {average_elapsed_time:.4f} seconds")

df = pd.read_csv('time_list_OptLCMS.csv')
average_elapsed_time = df['elapsed_time'].mean()
print(f"OptLCMS Average elapsed time: {average_elapsed_time:.4f} seconds")
