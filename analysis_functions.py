from itertools import combinations
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif_classical(df: pd.DataFrame) -> dict:
    """
    Считает VIF (classic) для всех числовых столбцов датафрейма df.
    Возвращает словарь вида {col_name: vif_value}.
    Если для какого-то столбца не удаётся вычислить – значение будет None.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].dropna()
    if X.empty:
        return {}
    X = sm.add_constant(X)
    vif_dict = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif_val = variance_inflation_factor(X.values, i)
            vif_dict[col] = vif_val
        except np.linalg.LinAlgError:
            vif_dict[col] = None
    return vif_dict


def iqr_outliers_count(series: pd.Series, coef: float = 1.5) -> int:
    """
    Возвращает число выбросов по правилу IQR.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - coef * iqr
    upper_bound = q3 + coef * iqr
    return ((series < lower_bound) | (series > upper_bound)).sum()


def variable_analysis(df: pd.DataFrame, column_name: str, vif_dict: dict):
    """
    Анализ одной переменной:
      - Определение типа (Бинарная/Дискретная/Интервальная/Ошибка)
      - Описательная статистика
      - Вычисление количества выбросов и тест Шапиро для ненормальных переменных
      - Поиск значимых корреляций (с использованием корреляции Спирмена)
      - Расчёт VIF
    """
    stats_results = {
        "Variable": column_name,
        "Type": None,
        "Unique Values": df[column_name].nunique(),
        "Missing Values": df[column_name].isnull().sum(),
        "Missing Percentage": round(df[column_name].isnull().mean() * 100, 2),
        "Normality p-value": np.nan,
        "Outliers": np.nan,
        "mean": None,
        "std": None,
        "min": None,
        "25%": None,
        "50%": None,
        "75%": None,
        "max": None
    }
    correlation_results = {
        "Variable": column_name,
        "Significant Variables": None,
        "VIF": None,
        "Multicollinearity": None
    }
    column_data = df[column_name]

    if pd.api.types.is_numeric_dtype(column_data):
        if column_data.dropna().nunique() == 2:
            var_type = "Бинарная"
        elif column_data.nunique() < 7:
            var_type = "Дискретная"
        else:
            var_type = "Интервальная"
    else:
        var_type = "Ошибка: переменная не подходит для регрессии"
    stats_results["Type"] = var_type

    if var_type == "Ошибка: переменная не подходит для регрессии":
        correlation_results["Significant Variables"] = "Нет (ошибка кодировки)"
        correlation_results["Multicollinearity"] = "Нет (ошибка кодировки)"
        return stats_results, correlation_results

    desc_stats = column_data.describe()
    for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        if stat_name in desc_stats:
            stats_results[stat_name] = desc_stats[stat_name]

    if var_type != "Бинарная":
        stats_results["Outliers"] = iqr_outliers_count(column_data, coef=1.5)
        if column_data.dropna().nunique() > 1:
            _, p_val_shapiro = sps.shapiro(column_data.dropna())
            stats_results["Normality p-value"] = round(p_val_shapiro, 4)

    if pd.api.types.is_numeric_dtype(column_data):
        significant_vars_info = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == column_name:
                continue
            temp_df = df[[column_name, col]].dropna()
            if len(temp_df) > 1:
                r, p_val = sps.spearmanr(temp_df[column_name], temp_df[col])
                if abs(r) >= 0.001 and p_val <= 0.05:
                    significant_vars_info.append(f"{col} ({round(r, 3)}, {round(p_val, 3)})")
        if not significant_vars_info:
            correlation_results["Significant Variables"] = "Нет значимых корреляций"
        else:
            correlation_results["Significant Variables"] = ", ".join(significant_vars_info)
    else:
        correlation_results["Significant Variables"] = "Не числовая переменная"

    vif_val = vif_dict.get(column_name, None)
    if vif_val is None:
        correlation_results["VIF"] = None
        correlation_results["Multicollinearity"] = "Нет (невозможно вычислить)"
    else:
        vif_val_rounded = round(vif_val, 3)
        correlation_results["VIF"] = vif_val_rounded
        correlation_results["Multicollinearity"] = "Да" if vif_val > 5 else "Нет"

    return stats_results, correlation_results


def correlation_heatmap_significant(df: pd.DataFrame, 
                                    df_statistics: pd.DataFrame, 
                                    df_correlations: pd.DataFrame,
                                    method: str = "spearman"):
    """
    Визуализирует тепловую карту корреляций для предикторов, которые показали связь с Diagnosis.
    
    Функция отбирает переменные из df_correlations, для которых поле "Significant Variables"
    (без учёта регистра) содержит слово "diagnosis". Затем к этому списку добавляется переменная
    "Diagnosis" и сортируются остальные переменные по их корреляции с Diagnosis
    в порядке возрастания (от самых отрицательных к положительным). Итоговый порядок таков, что Diagnosis 
    оказывается в конце (слева направо и сверху вниз при отображении хитмапа).
    
    Параметры:
      - df: исходный DataFrame с данными.
      - df_statistics: DataFrame со сводной информацией о переменных.
      - df_correlations: DataFrame с информацией о значимых корреляциях.
      - method: метод корреляции ("pearson" или "spearman").
      
    Если нет предикторов, показавших связь с Diagnosis, выводится соответствующее сообщение.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Отбираем переменные, у которых "Significant Variables" содержит слово "diagnosis"
    predictors = df_correlations.loc[
        df_correlations["Significant Variables"].apply(lambda s: "diagnosis" in s.lower()),
        "Variable"
    ].tolist()

    # Добавляем Diagnosis, если её нет, и перемещаем её в конец списка
    if "Diagnosis" not in predictors:
        predictors.append("Diagnosis")
    else:
        predictors = [var for var in predictors if var != "Diagnosis"] + ["Diagnosis"]

    if not predictors:
        print("Нет предикторов, показавших связь с Diagnosis, для построения тепловой карты.")
        return

    # Вычисляем корреляционную матрицу для выбранных переменных 
    numeric_df = df[predictors]
    corr_matrix = numeric_df.corr(method=method)

    if corr_matrix.empty:
        print("Нет числовых переменных среди выбранных предикторов для построения тепловой карты.")
        return

    # Сортируем все переменные, кроме Diagnosis, по их корреляции с Diagnosis (в порядке возрастания)
    if "Diagnosis" in corr_matrix.columns:
        corr_with_diag = corr_matrix["Diagnosis"].drop("Diagnosis")
        sorted_predictors = list(corr_with_diag.sort_values(ascending=True).index)
        predictors_ordered = sorted_predictors + ["Diagnosis"]
    else:
        predictors_ordered = predictors

    # Переставляем строки и столбцы корреляционной матрицы согласно новому порядку
    corr_matrix = corr_matrix.loc[predictors_ordered, predictors_ordered]
    annot_matrix = corr_matrix.copy().apply(lambda s: s.map("{:.2f}".format))

    plt.figure(figsize=(14, 10), dpi=150)
    ax = sns.heatmap(
        corr_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Correlation Heatmap ({method.capitalize()}) for Predictors with Diagnosis", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_distribution_by_diagnosis(df: pd.DataFrame, feature: str, diagnosis_col: str, colors: dict = None):
    """
    Строит график распределения для заданного признака по группам диагноза (0 и 1).
    
    Если переменная числовая и имеет более двух уникальных значений – строится гистограмма 
    распределения для каждой группы.
    Если переменная бинарная или категориальная – строится countplot.
    
    Заголовок графика имеет формат:
    "Распределение {feature} по группам {diagnosis_col}"
    
    Если colors не переданы, то по умолчанию для таргета 0 используется неяркий синий, а 
    для таргета 1 – неяркий красный.
    """
    if colors is None:
        colors = {0: "#6699CC", 1: "#CC6666"}
    
    # Фиксируем категории диагноза как [0, 1].
    categories = [0, 1]
    
    # Если переменная числовая и НЕ бинарная – используем гистограмму
    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].dropna().nunique() > 2:
        n = len(categories)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
        for ax, diag in zip(axes, categories):
            subset = df[df[diagnosis_col] == diag][feature].dropna()
            plt.sca(ax)  # Чтобы sns.histplot знал, на каких осях рисовать
            plt.title(f"{feature} для {diagnosis_col} {diag}", fontsize=16)
            sns.histplot(
                subset,
                bins=20,
                stat="count",
                color=colors.get(diag, None),
                edgecolor="black"
            )
            plt.xlabel(feature, fontsize=14)
            plt.ylabel("Количество", fontsize=14)
        fig.suptitle(f"Распределение {feature} по группам {diagnosis_col}", fontsize=18, y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        # Для бинарных или категориальных переменных используем countplot
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=df,
            x=feature,
            hue=diagnosis_col,
            palette=colors,
            hue_order=categories
        )
        plt.title(f"Распределение {feature} по группам {diagnosis_col}", fontsize=18, pad=15)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Количество", fontsize=14)
        plt.legend(title=diagnosis_col, fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.show()


def print_descriptive_stats(df: pd.DataFrame, features: list, diagnosis_col: str):
    """
    Выводит описательную статистику по указанным переменным для групп диагноза (0 и 1).
    """
    table_style = [
        {'selector': 'table', 'props': [('border', '2px solid black'), ('border-collapse', 'collapse')]},
        {'selector': 'th, td', 'props': [('border', '1px solid black'), ('padding', '5px')]}
    ]
    for feature in features:
        display(Markdown(f"### Описательная статистика для переменной **{feature}**"))
        stats_df = df.groupby(diagnosis_col)[feature].describe()
        styled_df = stats_df.style.set_table_styles(table_style)
        display(styled_df)


def perform_t_tests(df: pd.DataFrame, features: list, diagnosis_col: str, alpha: float = 0.05):
    """
    Проводит попарные t‑тесты для каждой переменной по группам диагноза (0 и 1).
    """
    # Фиксируем категории диагноза как [0, 1].
    categories = [0, 1]
    pairs = list(combinations(categories, 2))
    
    for feature in features:
        display(Markdown(f"### Попарные t‑тесты для переменной **{feature}**"))
        for cat1, cat2 in pairs:
            data1 = df.loc[df[diagnosis_col] == cat1, feature].dropna()
            data2 = df.loc[df[diagnosis_col] == cat2, feature].dropna()
            stat, p_value = sps.ttest_ind(data1, data2, equal_var=False)
            
            if p_value < alpha:
                result_text = "нулевая гипотеза (H0) отвергается (различия статистически значимы)"
            else:
                result_text = "нулевая гипотеза (H0) не отвергается (нет статистически значимых различий)"
            
            display(Markdown(
                f"**Сравнение {diagnosis_col} {cat1} и {cat2}:** p‑value = **{p_value:.5f}**. Результат: {result_text}."
            ))


def plot_all_distributions(df: pd.DataFrame, bins: int, 
                           figsize_interval: tuple, figsize_binary: tuple, 
                           hspace: float):
    """
    Визуализирует распределения числовых переменных из DataFrame.

    Для интервальных переменных (более двух уникальных значений) строится гистограмма с наложенной KDE.
    Для бинарных переменных (ровно два уникальных значения) строится countplot с разделением по цвету.
    
    Переменные выводятся в двух отдельных секциях с заголовками в стиле HTML.
    Дополнительно свободное пространство справа увеличивается (right=0.8), чтобы графики не "прилипали" к краю.

    Параметры:
      - df: исходный DataFrame.
      - bins: число корзин для гистограмм.
      - figsize_interval: размер фигуры для интервальных переменных (например, (15, 10)).
      - figsize_binary: размер фигуры для бинарных переменных (например, (10, 12)).
      - hspace: вертикальный отступ между графиками.
    """

    # Определяем все числовые переменные
    variables = df.select_dtypes(include=[np.number]).columns.tolist()

    # Разбиваем переменные на интервальные и бинарные
    interval_vars = [var for var in variables if df[var].dropna().nunique() > 2]
    binary_vars = [var for var in variables if df[var].dropna().nunique() == 2]

    # Задаём дефолтную палитру для бинарных переменных: '0' – неяркий синий, '1' – неяркий красный
    default_palette = {"0": "#6699CC", "1": "#CC6666"}

    def plot_vars(vars_list, plot_type: str, figsize: tuple, dynamic: bool = False, fixed_cols: int = 3):
        if not vars_list:
            return
        n = len(vars_list)
        # Для бинарных переменных используем динамический (квадратный) вариант, 
        # иначе фиксированное число столбцов.
        if plot_type == "binary" and dynamic:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            cols = fixed_cols if n >= fixed_cols else n
            rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n > 1 else [axes]

        for i, var in enumerate(vars_list):
            ax = axes[i]
            if plot_type == "interval":
                sns.histplot(df[var].dropna(), bins=bins, kde=True, ax=ax)
            else:
                # Для бинарных переменных: приводим данные к строковому типу
                data = df[var].dropna().astype(str)
                # Используем data в качестве hue для применения палитры
                sns.countplot(x=data, ax=ax, hue=data, dodge=False, palette=default_palette)
                # Удаляем легенду, чтобы не дублировать её на каждом графике
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            ax.set_title(var, fontsize=12)
            ax.tick_params(axis='x', labelrotation=0)

        # Отключаем лишние оси (если графиков меньше, чем subplot'ов)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Увеличиваем свободное пространство справа (right=0.8) и задаём вертикальный отступ
        plt.subplots_adjust(hspace=hspace, right=0.8)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # Заголовки секций в HTML-стиле
    if interval_vars:
        display(Markdown("<h2 style='color:#34495e; border-bottom:2px solid #bdc3c7; padding-bottom:5px;'>Распределение интервальных переменных</h2>"))
        plot_vars(interval_vars, plot_type="interval", figsize=figsize_interval, fixed_cols=3)
    else:
        display(Markdown("<h2 style='color:#34495e; border-bottom:2px solid #bdc3c7; padding-bottom:5px;'>Нет интервальных переменных для построения распределений.</h2>"))

    if binary_vars:
        display(Markdown("<h2 style='color:#34495e; border-bottom:2px solid #bdc3c7; padding-bottom:5px;'>Распределение бинарных переменных</h2>"))
        # Для бинарных переменных используем динамическое размещение для оптимального заполнения пространства
        plot_vars(binary_vars, plot_type="binary", figsize=figsize_binary, dynamic=True)
    else:
        display(Markdown("<h2 style='color:#34495e; border-bottom:2px solid #bdc3c7; padding-bottom:5px;'>Нет бинарных переменных для построения распределений.</h2>"))