# Statistics Analysis on Titanic Passenger Survival
# Student Name: Md Sajid Hasan
# Student ID: 24156348

# ===========================
# IMPORT LIBRARIES
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ===========================
# 1. DATA LOADING
# ===========================
df = pd.read_csv('/content/drive/MyDrive/Titanic-Dataset.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ===========================
# 2. DATA CLEANING
# ===========================
df_clean = df.copy()

# Handle missing Age values - fill with median
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

# Handle missing Embarked values - fill with mode
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

# Drop Cabin column (too many missing values)
df_clean = df_clean.drop('Cabin', axis=1)

# Create age groups for categorical analysis
df_clean['Age_Group'] = pd.cut(df_clean['Age'],
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

print("\n\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# ===========================
# 3. CALCULATE 4 STATISTICAL MOMENTS
# ===========================
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

print("\n" + "=" * 70)
print("4 MAIN STATISTICAL MOMENTS")
print("=" * 70)

moments_summary = []

for col in numerical_cols:
    print(f"\n{col.upper()}:")
    print("-" * 50)

    # 1st Moment: Mean (Central Tendency)
    mean_val = df_clean[col].mean()
    print(f"1. Mean (μ):              {mean_val:.4f}")

    # 2nd Moment: Variance (Dispersion)
    variance_val = df_clean[col].var()
    std_val = df_clean[col].std()
    print(f"2. Variance (σ²):         {variance_val:.4f}")
    print(f"   Standard Deviation:    {std_val:.4f}")

    # 3rd Moment: Skewness (Asymmetry)
    skewness_val = df_clean[col].skew()
    print(f"3. Skewness:              {skewness_val:.4f}")
    if skewness_val > 0:
        print(f"   → Right-skewed (positive)")
    elif skewness_val < 0:
        print(f"   → Left-skewed (negative)")

    # 4th Moment: Kurtosis (Tailedness)
    kurtosis_val = df_clean[col].kurtosis()
    print(f"4. Kurtosis:              {kurtosis_val:.4f}")
    if kurtosis_val > 0:
        print(f"   → Heavy-tailed (leptokurtic)")
    elif kurtosis_val < 0:
        print(f"   → Light-tailed (platykurtic)")

    moments_summary.append({
        'Variable': col,
        'Mean': mean_val,
        'Variance': variance_val,
        'Std_Dev': std_val,
        'Skewness': skewness_val,
        'Kurtosis': kurtosis_val
    })

# Create summary dataframe
moments_df = pd.DataFrame(moments_summary)
print("\n" + "=" * 70)
print("SUMMARY TABLE OF STATISTICAL MOMENTS")
print("=" * 70)
print(moments_df.to_string(index=False))

# ===========================
# 4. RELATIONAL PLOT - Scatter Plot
# ===========================
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_clean['Age'],
                     df_clean['Fare'],
                     c=df_clean['Survived'],
                     cmap='RdYlGn',
                     alpha=0.6,
                     edgecolors='black',
                     s=50)

plt.xlabel('Age (years)', fontsize=12, fontweight='bold')
plt.ylabel('Fare (£)', fontsize=12, fontweight='bold')
plt.title('Relational Plot: Age vs Fare by Survival Status\n' +
          f'Mean Age: {moments_df.loc[0, "Mean"]:.2f}, Mean Fare: {moments_df.loc[1, "Mean"]:.2f}',
          fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Survived (0=No, 1=Yes)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('relational_plot_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nRelational plot saved: relational_plot_scatter.png")

# ===========================
# 5. CATEGORICAL PLOT - Bar Chart
# ===========================
# Calculate survival rate by passenger class
survival_by_class = df_clean.groupby('Pclass')['Survived'].agg(['sum', 'count']).reset_index()
survival_by_class['Survival_Rate'] = (survival_by_class['sum'] / survival_by_class['count']) * 100
survival_by_class['Class_Label'] = survival_by_class['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

plt.figure(figsize=(10, 6))
bars = plt.bar(survival_by_class['Class_Label'],
               survival_by_class['Survival_Rate'],
               color=['#2E86AB', '#A23B72', '#F18F01'],
               edgecolor='black',
               linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, survival_by_class['Survival_Rate'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.xlabel('Passenger Class', fontsize=12, fontweight='bold')
plt.ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
plt.title('Categorical Plot: Survival Rate by Passenger Class\n' +
          f'Overall Survival Rate: {df_clean["Survived"].mean()*100:.2f}%',
          fontsize=14, fontweight='bold')
plt.ylim(0, 70)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('categorical_plot_bar.png', dpi=300, bbox_inches='tight')
plt.show()

print("Categorical plot saved: categorical_plot_bar.png")

# ===========================
# 6. STATISTICAL PLOT - Box Plot
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot for Age by Class
df_clean['Class_Label'] = df_clean['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
box1 = df_clean.boxplot(column='Age', by='Class_Label', ax=axes[0], patch_artist=True)
axes[0].set_xlabel('Passenger Class', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Age (years)', fontsize=11, fontweight='bold')
axes[0].set_title(f'Age Distribution by Class\nSkewness: {moments_df.loc[0, "Skewness"]:.3f}',
                  fontsize=12, fontweight='bold')
axes[0].get_figure().suptitle('')

# Box plot for Fare by Class
box2 = df_clean.boxplot(column='Fare', by='Class_Label', ax=axes[1], patch_artist=True)
axes[1].set_xlabel('Passenger Class', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Fare (£)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Fare Distribution by Class\nSkewness: {moments_df.loc[1, "Skewness"]:.3f}',
                  fontsize=12, fontweight='bold')
axes[1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('statistical_plot_box.png', dpi=300, bbox_inches='tight')
plt.show()

print("Statistical plot (box) saved: statistical_plot_box.png")

# ===========================
# 7. STATISTICAL PLOT - Correlation Heatmap
# ===========================
corr_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            vmin=-1, vmax=1)

plt.title('Statistical Plot: Correlation Heatmap of Numerical Variables\n' +
          'Shows relationships between passenger features',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('statistical_plot_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Statistical plot (heatmap) saved: statistical_plot_heatmap.png")

print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)
print(f"\nTotal Passengers: {len(df_clean)}")
print(f"Survival Rate: {df_clean['Survived'].mean()*100:.2f}%")
print(f"\nKey Findings:")
print(f"- Fare shows highest skewness ({moments_df.loc[1, 'Skewness']:.2f}) - heavily right-skewed")
print(f"- Age distribution is relatively symmetric (skewness: {moments_df.loc[0, 'Skewness']:.2f})")
print(f"- 1st Class passengers had {survival_by_class.loc[0, 'Survival_Rate']:.1f}% survival rate")
print(f"- 3rd Class passengers had {survival_by_class.loc[2, 'Survival_Rate']:.1f}% survival rate")
print(f"- Strongest correlation: Pclass and Fare ({correlation_matrix.loc['Pclass', 'Fare']:.3f})")
print("\n" + "=" * 70)
