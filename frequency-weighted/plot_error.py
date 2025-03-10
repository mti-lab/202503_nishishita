import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cms = pd.read_csv("results/CMS_results.csv")
optlcms = pd.read_csv("results/OptLCMS_result.csv")
lcms = pd.read_csv(f"results/LCMS_result.csv")

cms["Memory Usage"] = cms["Memory Usage"] / 10**6
optlcms["Memory Usage"] = optlcms["Memory Usage"] / 10**6 + 0.0152
lcms["Memory Usage"] = lcms["Memory Usage"] / 10**6 + 0.0152

cms_filtered = cms[cms["Memory Usage"] < 0.9]
optlcms_filtered = optlcms[optlcms["Memory Usage"] < 0.9]
lcms_filtered = lcms[lcms["Memory Usage"] < 0.9]

plt.plot(cms_filtered["Memory Usage"], cms_filtered["Error"], label="CMS", marker='v', linestyle='--', linewidth = 3.0, markersize=14)
plt.plot(lcms_filtered["Memory Usage"], lcms_filtered["Error"], label="LCMS", marker='o', linestyle=':',linewidth = 3.0,  markersize=14)
plt.plot(optlcms_filtered["Memory Usage"], optlcms_filtered["Error"], label="Ours", marker='s', linestyle='-', linewidth = 3.0, markersize=14)
plt.legend()
plt.yscale('log')
plt.xlabel('Memory Usage (MB)', fontsize=20)
plt.ylabel("Error", fontsize=20)
plt.title('Frequency-Weighted', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=16)  # Minor ticks 
plt.grid(True, which="both", ls="--", c='gray')
plt.savefig("img/frequency-weighted-error.svg", dpi=600,format="svg", bbox_inches="tight")
