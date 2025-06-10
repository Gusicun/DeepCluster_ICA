import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import LinearSegmentedColormap

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

M1_df = pd.read_csv(r"D:\processed_data\processed_data\M.csv", index_col=0)
M2_df = pd.read_csv(r"D:\ica_runs_test\130\tmp\robust_M_consensus.csv", index_col=0)
A1_df = pd.read_csv(r"D:\processed_data\processed_data\A.csv", index_col=0)
A2_df = pd.read_csv(r"D:\ica_runs_test\130\tmp\robust_A_consensus.csv", index_col=0)

M1 = M1_df.values 
M2 = M2_df.values  
n1 = M1.shape[1]
n2 = M2.shape[1]

def align_mixing(df, expected_cols, name):

    if df.shape[1] == expected_cols:
        return df.values
    elif df.shape[0] == expected_cols:
        print(f"Warn: {name} is transported")
        return df.T.values
    else:
        raise ValueError(f"{name} and {df.shape} and {expected_cols} un-matched！")

A1 = align_mixing(A1_df, n1, 'A1')  
A2 = align_mixing(A2_df, n2, 'A2')  

kurt1 = stats.kurtosis(M1, axis=0, fisher=False)
kurt2 = stats.kurtosis(M2, axis=0, fisher=False)
avg_abs_kurt1 = np.mean(np.abs(kurt1))
avg_abs_kurt2 = np.mean(np.abs(kurt2))
print("=== Average Kurtosis Peak===")
print("optICA:           ", np.round(avg_abs_kurt1, 3))
print("DeepClusterICA:   ", np.round(avg_abs_kurt2, 3))

skew1 = stats.skew(M1, axis=0)
skew2 = stats.skew(M2, axis=0)

var1 = M1.var(axis=0)
var2 = M2.var(axis=0)

corr1 = np.corrcoef(M1.T)
corr2 = np.corrcoef(M2.T)
abs_corr1 = np.abs(corr1)
abs_corr2 = np.abs(corr2)
avg_internal_corr1 = (np.sum(abs_corr1) - np.trace(abs_corr1)) / (n1 * (n1 - 1))
avg_internal_corr2 = (np.sum(abs_corr2) - np.trace(abs_corr2)) / (n2 * (n2 - 1))
print("\n=== Average inner corr ===")
print("optICA:           ", np.round(avg_internal_corr1, 4))
print("DeepClusterICA:   ", np.round(avg_internal_corr2, 4))

M_corr_mat = np.corrcoef(M1.T, M2.T)[:n1, n1:]
abs_M_corr_mat = np.abs(M_corr_mat)
row_ind, col_ind = linear_sum_assignment(-abs_M_corr_mat)
matched_corr = abs_M_corr_mat[row_ind, col_ind]
avg_matched_corr = np.mean(matched_corr)
print("\n=== Average matched corr===")
print("Average:  ", np.round(avg_matched_corr, 4))

import seaborn as sns
plt.figure(figsize=(8, 6))

sns.kdeplot(kurt1, label='optICA', fill=True)
sns.kdeplot(kurt2, label='DeepClusterICA', fill=True)

plt.xlabel('Kurtosis')
plt.ylabel('Density')
plt.title('Kurtosis Distribution Comparison (KDE)')
plt.legend()
plt.tight_layout()
plt.savefig('kurtosis_kde.pdf')
plt.close()

plt.figure()
plt.hist(kurt1, bins=30, alpha=0.5, label='optICA')
plt.hist(kurt2, bins=30, alpha=0.5, label='DeepClusterICA')
plt.xlabel('Kurtosis')
plt.ylabel('Frequency')
plt.title('Kurtosis Histogram Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('kurtosis_histogram.pdf')
plt.close()

plt.figure()
plt.imshow(abs_corr1, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(label='Absolute Correlation')
plt.title('optICA Internal Absolute Correlation')
plt.tight_layout()
plt.savefig('optica_internal_corr.pdf')
plt.close()

plt.figure()
plt.imshow(abs_corr2, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(label='Absolute Correlation')
plt.title('DeepClusterICA Internal Absolute Correlation')
plt.tight_layout()
plt.savefig('deepclusterica_internal_corr.pdf')
plt.close()

df_pairs = pd.DataFrame({
    "DeepClusterICA_Index": row_ind,
    "optICA_Index":         col_ind,
    "Absolute_Corr":        matched_corr
})
df_pairs.to_csv("matched_pairs.csv", index=False)

BlueMap = LinearSegmentedColormap.from_list('BlueMap', ['#e0f7fa', '#0288d1'])

BlueMap = LinearSegmentedColormap.from_list('BlueMap', ['#e0f7fa', '#0288d1'])

plt.figure()
plt.imshow(abs_M_corr_mat, cmap=BlueMap, interpolation='nearest', vmin=0, vmax=1, origin='lower')
plt.colorbar(label='Absolute Correlation')
plt.scatter(col_ind, row_ind, marker='o',color='orange',  s=5, label='Matched Pair')
plt.xlabel('DeepClusterICA Components')
plt.ylabel('optICA Components')
plt.title('Cross Absolute Correlation with Matched Pairs')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('cross_corr_with_matched.pdf')
plt.close()

row_ind, col_ind = linear_sum_assignment(-abs_M_corr_mat)
matched_corr = abs_M_corr_mat[row_ind, col_ind]
A_matched = abs_M_corr_mat[np.ix_(row_ind, col_ind)]
threshold = 0.8
mask = matched_corr >= threshold
ri = np.arange(len(row_ind))[mask]
ci = np.arange(len(col_ind))[mask]
mc = matched_corr[mask]
plt.figure()
im = plt.imshow(
    A_matched,
    cmap=BlueMap,
    interpolation='nearest',
    vmin=0, vmax=1,
    origin='lower'
)
plt.colorbar(im, label='Absolute Correlation')

plt.scatter(
    ci, ri,
    color='orange',
    marker='o',
    s=5,         
    linewidths=0.5,
    label=f'Matched Pair (r ≥ {threshold})',
    zorder=2
)

plt.xticks(np.arange(len(col_ind)), col_ind, rotation=90, fontsize=6)
plt.yticks(np.arange(len(row_ind)), row_ind, fontsize=6)

plt.xlabel('DeepClusterICA Component (original index)')
plt.ylabel('optICA Component (original index)')
plt.title('Cross Absolute Correlation with Matched Pairs')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('cross_corr_with_matched_thresholded.pdf')
plt.close()

labels1 = np.argmax(M1, axis=1)
labels2 = np.argmax(M2, axis=1)
C = confusion_matrix(labels1, labels2, normalize='true')
row_ind, col_ind = linear_sum_assignment(-C)
C_matched = C[row_ind[:, None], col_ind]
cmap = LinearSegmentedColormap.from_list('BlueOrange', ['lightblue', 'orange'])
plt.figure()
plt.imshow(C_matched, cmap=cmap, aspect='auto', interpolation='nearest', origin='lower')
plt.colorbar(label='Normalized Overlap')
plt.xlabel('DeepClusterICA Matched')
plt.ylabel('optICA Matched')
plt.title('Matched Cluster Overlap Matrix')
plt.tight_layout()
plt.savefig('matched_confusion_matrix.pdf')
plt.close()

x1, y1 = ecdf(kurt1)
x2, y2 = ecdf(kurt2)
plt.figure()
plt.step(x1, y1, where='post', label='optICA')
plt.step(x2, y2, where='post', label='DeepClusterICA')
plt.xlabel('Kurtosis')
plt.ylabel('ECDF')
plt.title('Kurtosis ECDF Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('kurtosis_ecdf.pdf')
plt.close()

x1, y1 = ecdf(skew1)
x2, y2 = ecdf(skew2)
plt.figure()
plt.step(x1, y1, where='post', label='optICA')
plt.step(x2, y2, where='post', label='DeepClusterICA')
plt.xlabel('Skewness')
plt.ylabel('ECDF')
plt.title('Skewness ECDF Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('skewness_ecdf.pdf')
plt.close()

plt.figure()
plt.boxplot([var1, var2], labels=['optICA', 'DeepClusterICA'], showfliers=False)
plt.ylabel('Variance')
plt.title('Variance Distribution Comparison')
plt.tight_layout()
plt.savefig('variance_boxplot.pdf')
plt.close()


def zscore_cols(mat):
    return (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=1)
A1n = zscore_cols(A1)   
A2n = zscore_cols(A2)     
A_corr_mat = np.corrcoef(A1n.T, A2n.T)[:n1, n1:]
abs_A_corr = np.abs(A_corr_mat)
row_ind_A, col_ind_A = linear_sum_assignment(-abs_A_corr)   
A_corr_reordered = A_corr_mat[np.ix_(row_ind_A, col_ind_A)]
plt.figure()
cmap = LinearSegmentedColormap.from_list(
    'DeepBlueMap',
    [
        (0.00, "#ffffff"),  
        (0.05, "#e0f3f8"),  
        (0.20, "#c6dbef"), 
        (0.40, "#9ecae1"),  
        (0.60, "#6baed6"),  
        (0.80, "#3182bd"),  
        (1.00, "#08306b")   
    ],
    N=256              
)

plt.imshow(A_corr_reordered, cmap=cmap, origin='lower', vmin=-1, vmax=1)
plt.colorbar(label='Pearson r')
plt.xlabel('DeepClusterICA Mixing Components (matched order)')
plt.ylabel('optICA Mixing Components (matched order)')
plt.title('Mixing Matrix Cross-Correlation (Matched & Normalized)')
plt.tight_layout()
plt.savefig('mixing_corr_heatmap.pdf')
plt.close()

print("\n已生成并保存以下 PDF：")
print("- kurtosis_kde.pdf")
print("- kurtosis_histogram.pdf")
print("- optica_internal_corr.pdf")
print("- deepclusterica_internal_corr.pdf")
print("- cross_corr_with_matched.pdf")
print("- matched_confusion_matrix.pdf")
print("- kurtosis_ecdf.pdf")
print("- skewness_ecdf.pdf")
print("- variance_boxplot.pdf")
print("- mixing_corr_heatmap.pdf")