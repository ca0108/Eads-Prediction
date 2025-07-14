#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

inputfile = r"your inputfile"
outputfile = r"your outputfile"

df = pd.read_csv(inputfile, encoding='gbk')
original_columns = df.columns.tolist()  # 记录原始列名

df.drop(['System', 'target'], axis=1, inplace=True)

correlation_matrix = df.corr()

threshold = 0.6
high_corr_pairs = []

for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):  # 只遍历上三角
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            f1 = correlation_matrix.index[i]
            f2 = correlation_matrix.columns[j]
            high_corr_pairs.append((f1, f2, correlation_matrix.iloc[i, j]))

print("Pearson's score > {:.2f} 的特征对：".format(threshold))
for pair in high_corr_pairs:
    print(f"features: {pair[0]} and {pair[1]}, Pearson: {pair[2]:.2f}")

features_to_drop = set()
for f1, f2, corr in high_corr_pairs:
    features_to_drop.add(f2)

print("\ndelete：")
for feature in features_to_drop:
    print(feature)

reduced_df = df.drop(columns=features_to_drop)

if 'System' in original_columns:
    reduced_df['System'] = pd.read_csv(inputfile, encoding='gbk')['System']
if 'target' in original_columns:
    reduced_df['target'] = pd.read_csv(inputfile, encoding='gbk')['target']

reduced_df.to_csv(outputfile, index=False, encoding='utf-8')
print(f"\nsave as {outputfile}")

def plot_correlation_heatmap(dataframe, title, figsize=(30, 30)):
    colormap = plt.cm.coolwarm
    plt.figure(figsize=figsize)
    plt.title(title, y=1.05, size=15)
    sns.set(font_scale=1.0)

    ax = sns.heatmap(
        dataframe.corr(),
        linewidths=0.01,
        vmin=-1.2,
        vmax=1.2,
        square=True,
        cmap=colormap,
        linecolor="white",
        annot=False,
        center=0,
        xticklabels=True,
        yticklabels=True,
    )
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.show()

plot_correlation_heatmap(df, "Pearson Correlation of Original Features")

reduced_df_without_target_system = reduced_df.drop(['System', 'target'], axis=1)
plot_correlation_heatmap(reduced_df_without_target_system, "Pearson Correlation of Reduced Features")
