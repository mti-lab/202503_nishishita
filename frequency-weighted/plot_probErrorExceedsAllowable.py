import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plcms = pd.read_csv("results/OptLCMS_result.csv")
lcms = pd.read_csv(f"results/LCMS_result.csv")

plcms["Memory Usage"] = plcms["Memory Usage"] / 10**6 + 0.0152
lcms["Memory Usage"] = lcms["Memory Usage"] / 10**6 + 0.0152

plcms_filtered = plcms[plcms["Memory Usage"] < 0.9]
lcms_filtered = lcms[lcms["Memory Usage"] < 0.9]

# プロット
plt.plot(lcms_filtered["Memory Usage"], lcms_filtered["False Rate"], label="LCMS", marker='o', linestyle=':', color='#ff7f0e',linewidth = 3.0, markersize=14)
plt.plot(plcms_filtered["Memory Usage"], plcms_filtered["False Rate"], label="OptLCMS (Ours)", marker='s', linestyle='-', color='#2ca02c',linewidth = 3.0, markersize=14)
plt.yscale('log')
plt.xlabel('Memory Usage (MB)',fontsize=20)
plt.ylabel(r'$\mathrm{Pr}\left[\hat{f}(x)-f(x) > \epsilon N\right]$',fontsize=20)
plt.ylim([10**(-3),10**(-1)])
plt.title('Frequency-Weighted',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=16)  # Minor ticks (optional)
plt.grid(True,which="both",ls="--",c='gray')
plt.legend()
plt.savefig("img/exceedance-probability-fequency-weighted.svg",dpi=600,format='svg',bbox_inches="tight")