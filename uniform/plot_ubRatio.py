import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

optlcms = pd.read_csv("./results/OptLcms_results.csv")
lcms = pd.read_csv("./results/LCMS_result.csv")


optlcms["Memory Usage"] = optlcms["Memory Usage"] / 10**6 + 0.0152
lcms["Memory Usage"] = lcms["Memory Usage"] / 10**6 + 0.0152

# Memory Usageが0.9未満のデータをフィルタリング
optlcms_filtered = optlcms[optlcms["Memory Usage"] < 0.9]
lcms_filtered = lcms[lcms["Memory Usage"] < 0.9]

# プロット
plt.plot(lcms_filtered["Memory Usage"], lcms_filtered["UB Memmory Ratio"], label="LCMS",marker='o', linestyle=':', color='#ff7f0e',linewidth = 3.0, markersize=14)
plt.plot(optlcms_filtered["Memory Usage"], optlcms_filtered["UB Memory Ratio"], label="Ours", marker='s',linestyle='-',color='#2ca02c',linewidth = 3.0, markersize=14)

plt.xlabel('Memory Usage (MB)',fontsize=20)
plt.ylabel('UB Memory Ratio',fontsize=20)
plt.ylim([0,0.150])

plt.title('Uniform',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Minor ticks (optional)
plt.grid(True,which="both",ls="--",c='gray')
plt.legend()
plt.savefig("img/ub-memory-ratio-uniform.svg",dpi=600,format='svg',bbox_inches="tight")
