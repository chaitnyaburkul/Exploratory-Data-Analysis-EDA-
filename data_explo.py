import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('country_wise_latest.csv')

## Step 1: Initial Data Exploration
print("=== Dataset Overview ===")
print(f"Shape of dataset: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and non-null counts:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

## Step 2: Data Cleaning
print("\n=== Data Cleaning ===")
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Fill missing values (if any)
df.fillna(0, inplace=True)

# Check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Check for outliers in numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("\nPotential outliers (values beyond 3 standard deviations):")
for col in numerical_cols:
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    outliers = df[abs(z_scores) > 3]
    if not outliers.empty:
        print(f"\n{col}: {len(outliers)} potential outliers")
        print(outliers[['Country/Region', col]].head())

## Step 3: Univariate Analysis
print("\n=== Univariate Analysis ===")
# Distribution of key metrics
plt.figure(figsize=(15, 10))

# Confirmed cases distribution
plt.subplot(2, 2, 1)
sns.histplot(df['Confirmed'], bins=50, kde=True)
plt.title('Distribution of Confirmed Cases')
plt.xscale('log')

# Death rate distribution
plt.subplot(2, 2, 2)
sns.histplot(df['Deaths / 100 Cases'], bins=30, kde=True)
plt.title('Distribution of Death Rates')

# Recovery rate distribution
plt.subplot(2, 2, 3)
sns.histplot(df['Recovered / 100 Cases'], bins=30, kde=True)
plt.title('Distribution of Recovery Rates')

# Weekly percentage increase
plt.subplot(2, 2, 4)
sns.histplot(df['1 week % increase'], bins=30, kde=True)
plt.title('Distribution of Weekly % Increase')

plt.tight_layout()
plt.show()

## Step 4: Bivariate Analysis
print("\n=== Bivariate Analysis ===")
# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Scatter plot: Death rate vs Recovery rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recovered / 100 Cases', y='Deaths / 100 Cases', 
                data=df, hue='WHO Region', size='Confirmed', sizes=(20, 200))
plt.title('Death Rate vs Recovery Rate by WHO Region')
plt.xlabel('Recovery Rate (%)')
plt.ylabel('Death Rate (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Boxplot of death rates by region
plt.figure(figsize=(12, 6))
sns.boxplot(x='WHO Region', y='Deaths / 100 Cases', data=df)
plt.title('Death Rate Distribution by WHO Region')
plt.xticks(rotation=45)
plt.show()

## Step 5: Multivariate Analysis
print("\n=== Multivariate Analysis ===")
# Pairplot of key metrics
sns.pairplot(df[['Confirmed', 'Deaths', 'Recovered', 'Active', 'WHO Region']], 
             hue='WHO Region', diag_kind='kde')
plt.suptitle('Pairplot of Key Metrics by WHO Region', y=1.02)
plt.show()

# Top 10 countries by different metrics
def plot_top_countries(metric, title, color_palette='viridis'):
    top_10 = df.nlargest(10, metric)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=metric, y='Country/Region', data=top_10, palette=color_palette)
    plt.title(f'Top 10 Countries by {title}')
    plt.xlabel(title)
    plt.ylabel('Country')
    plt.show()

plot_top_countries('Confirmed', 'Confirmed Cases')
plot_top_countries('Deaths', 'Total Deaths', 'magma')
plot_top_countries('Deaths / 100 Cases', 'Death Rate (%)', 'rocket')
plot_top_countries('1 week % increase', 'Weekly % Increase', 'flare')

## Step 6: Time Series Analysis (using weekly change data)
print("\n=== Time Series Analysis ===")
# Countries with highest weekly increase
high_increase = df.nlargest(10, '1 week change')
plt.figure(figsize=(12, 6))
sns.barplot(x='1 week change', y='Country/Region', data=high_increase, palette='mako')
plt.title('Top 10 Countries by Weekly Case Increase')
plt.xlabel('New Cases in Last Week')
plt.ylabel('Country')
plt.show()

## Step 7: Key Insights and Observations
print("\n=== Key Insights ===")
# 1. Countries with highest death rates
high_death_rate = df[df['Confirmed'] > 10000].nlargest(5, 'Deaths / 100 Cases')
print("\nCountries with highest death rates (>10k cases):")
print(high_death_rate[['Country/Region', 'Deaths / 100 Cases', 'Confirmed']])

# 2. Countries with highest recovery rates
high_recovery_rate = df[df['Confirmed'] > 10000].nlargest(5, 'Recovered / 100 Cases')
print("\nCountries with highest recovery rates (>10k cases):")
print(high_recovery_rate[['Country/Region', 'Recovered / 100 Cases', 'Confirmed']])

# 3. Regional analysis
regional_stats = df.groupby('WHO Region').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'Active': 'sum',
    'Deaths / 100 Cases': 'mean',
    'Recovered / 100 Cases': 'mean'
}).sort_values('Confirmed', ascending=False)
print("\nRegional Statistics:")
print(regional_stats)

# 4. Countries with unusual patterns
# High death rate but low recovery rate
unusual = df[(df['Deaths / 100 Cases'] > 10) & (df['Recovered / 100 Cases'] < 50)]
print("\nCountries with unusual patterns (high death rate, low recovery rate):")
print(unusual[['Country/Region', 'Deaths / 100 Cases', 'Recovered / 100 Cases']])