#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency

def plot_histograms(df, numeric_columns, ncols, kde, bins):
    n = len(numeric_columns)
    nrows = (n + ncols - 1) // ncols  # Количество строк
    ncols=ncols
    kde=kde
    bins=bins
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows), sharey=True)
    axes = axes.flatten()  # Преобразуем в одномерный массив

    for i, column in enumerate(numeric_columns):
        sns.histplot(df[column], kde=kde, bins=bins, ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    # Удаляем пустые подграфики, если есть
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout() 
    plt.show()


def plot_distribution_with_hue(df, numeric_columns, hue_column, n_cols, bins, alpha=0.4):
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharey=True)  
    axes = axes.flatten() 
    hue_column=hue_column
    n_cols=n_cols
    bins=bins
    
    for i, column in enumerate(numeric_columns):
        ax = sns.histplot(df, x=column, hue=hue_column, kde=True, bins=bins, ax=axes[i], alpha=alpha)
        sns.move_legend(ax, "upper left")
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0, h_pad=0.8, w_pad=0.8, rect=[0, 0, 1, 1])
    plt.show()

#Функция, объединяющая два блока графиков:
#    1. Общее распределение бинарных признаков.
#    2. Распределение бинарных признаков в разрезе диагноза.

def binary_distributions(data, binary_cols, target_col="Diagnosis"):
   
    num_columns = 3  # Количество графиков в ряду
    
    # ОБЩЕЕ РАСПРЕДЕЛЕНИЕ
    num_rows = (len(binary_cols) + num_columns - 1) // num_columns
    custom_palette = ["#6a93cb", "#ffb3a7"]  # Палитра

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 4 * num_rows), sharey=True)
    fig.suptitle("Общее распределение бинарных признаков", fontsize=24, fontweight="bold", color="#2c3e50")

    axes = axes.flatten()
    for i, column in enumerate(binary_cols):
        countplot = sns.countplot(data=data, x=column, ax=axes[i], hue=column, palette=custom_palette, edgecolor="black", legend=False)

        # Вычисление долей в % + добавление аннотации
        total_count = data[column].value_counts().sum()
        for p in countplot.patches:
            if p.get_height() > 0:
                percentage = round((p.get_height() / total_count) * 100) 
                countplot.annotate(f"{int(p.get_height())} ({percentage}%)",
                                   (p.get_x() + p.get_width() / 2, p.get_height() + 3),
                                   ha="center", va="bottom", fontsize=10, fontweight="bold", color="#34495e")

        axes[i].set_ylabel("Количество", fontsize=12, color="#2c3e50")
        axes[i].set_xlabel("")
        axes[i].set_title(f"{column}", fontsize=12, fontweight="bold", color="#2c3e50")
        axes[i].grid(axis='y', linestyle="--", alpha=0.7)

    for j in range(i + 1, len(axes)): 
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    #РАСПРЕДЕЛЕНИЕ В РАЗРЕЗЕ ДИАГНОЗА
    filtered_binary_cols = [col for col in binary_cols if col != target_col]
    num_rows = (len(filtered_binary_cols) + num_columns - 1) // num_columns
    custom_palette_diag = ["#4C72B0", "#DD8452"]  #Палитра

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 5 * num_rows), sharey=True)
    fig.suptitle("Распределение бинарных признаков в разрезе диагноза", fontsize=24, fontweight="bold", color="#2c3e50")

    axes = axes.flatten()
    for i, column in enumerate(filtered_binary_cols):
        countplot = sns.countplot(data=data, x=column, hue=target_col, ax=axes[i],
                                  palette=custom_palette_diag, order=data[column].value_counts().index,
                                  edgecolor="black")

        for p in countplot.patches:
            if p.get_height() > 0:
                countplot.annotate(f"{int(p.get_height())}",
                                   (p.get_x() + p.get_width() / 2, p.get_height() + 3),
                                   ha="center", va="bottom", fontsize=12, fontweight="bold", color="#34495e")

        axes[i].set_ylabel("Количество", fontsize=12, color="#2c3e50")
        axes[i].set_xlabel("")
        axes[i].set_title(f"Распределение: {column}\n в разрезе {target_col}", fontsize=14, fontweight="bold", color="#2c3e50")
        axes[i].grid(axis='y', linestyle="--", alpha=0.5)

    for j in range(i + 1, len(axes)):  
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def correlation_with_target(df_dum, target_var):
    numeric_cols = [col for col in df_dum.select_dtypes(include=[np.number]).columns if df_dum[col].nunique() > 2]
    binary_cols = [col for col in df_dum.columns if set(df_dum[col].unique()) == {0, 1}]
    numeric_cols_with_diag = numeric_cols + [target_var]

    # Тепловая карта для числовых переменных
    if len(numeric_cols) > 1:
        plt.figure(figsize=(20, 6))
        corr_matrix = df_dum[numeric_cols_with_diag].corr(method='spearman')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
          cmap=sns.diverging_palette(220, 10, as_cmap=True),
          center=0,  # Добавьте этот параметр
          linewidths=0.5, linecolor="#D3D3D3",
          cbar_kws={"shrink": 0.8})
        plt.title("Тепловая карта Спирмена (числовые переменные)")
        plt.show()

    # Числовые переменные и target_var (Спирмен)
    significant_numeric = []
    for num_col in numeric_cols:
        if num_col == target_var:
            continue
        rho, p_value = spearmanr(df_dum[num_col], df_dum[target_var])
        if p_value < 0.05:
            significant_numeric.append([num_col, round(rho, 2), round(p_value, 3)])

    if significant_numeric:
        df_num = pd.DataFrame(significant_numeric, columns=["Числовая", "ρ (Спирмен)", "p-value"])
        df_num["|ρ (Спирмен)|"] = df_num["ρ (Спирмен)"].abs()  # Добавляем столбец с модулем коэффициента Спирмена
        df_num = df_num.sort_values(by="|ρ (Спирмен)|", ascending=False).drop(columns=["|ρ (Спирмен)|"])  # Сортируем по модулю и удаляем столбец
        print("\nЗначимые корреляции числовых переменных с Diagnosis (p < 0.05)")
        display(df_num)

    # Бинарные переменные и target_var (хи-квадрат)
    significant_binary = []
    for bin_col in binary_cols:
        contingency_table = pd.crosstab(df_dum[bin_col], df_dum[target_var])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < 0.05:
            significant_binary.append([bin_col, round(chi2, 2), round(p_value, 3)])

    if significant_binary:
        df_bin = pd.DataFrame(significant_binary, columns=["Бинарная", "χ² (хи-квадрат)", "p-value"])
        df_bin = df_bin.sort_values(by="χ² (хи-квадрат)", ascending=False)
        print("\nЗначимые связи бинарных переменных с Diagnosis (хи-квадрат, p < 0.05)")
        display(df_bin)

