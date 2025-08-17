import matplotlib.pyplot as plt
import seaborn as sns

def quick_num_plots(df, col):
    plots = []
    fig1, ax1 = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax1)
    ax1.set_title(f"Distribution of {col}")
    plots.append(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(y=df[col].dropna(), ax=ax2)
    ax2.set_title(f"Boxplot of {col}")
    plots.append(fig2)

    return plots

def corr_heatmap(df, cols):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def cat_value_counts(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().head(20).plot(kind='bar', ax=ax)
    ax.set_title(f"Value Counts of {col}")
    plt.xticks(rotation=45)
    return fig, ax
