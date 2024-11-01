import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Läs in originaldata
url = "https://raw.githubusercontent.com/sachin365123/CSV-files-for-Data-Science-and-Machine-Learning/refs/heads/main/car.csv"
df = pd.read_csv(url)

# Grundläggande dataanalys
print("Dataset Information:")
print("-" * 50)
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nNull Values:")
print(df.isnull().sum())

# Analysera populäraste bilmärkena
plt.figure(figsize=(15, 6))
df['Car_Name'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Most Popular Car Models')
plt.xlabel('Car Model')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analysera bränsletyper
plt.figure(figsize=(10, 6))
df['Fuel_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Fuel Types')
plt.axis('equal')
plt.show()

# Analysera årsmodeller
plt.figure(figsize=(15, 6))
sns.histplot(data=df, x='Year', bins=20)
plt.title('Distribution of Car Models by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Box plot för kilometer körda per bränsletyp
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuel_Type', y='Kms_Driven', data=df)
plt.title('Kilometers Driven by Fuel Type')
plt.xticks(rotation=45)
plt.show()

# Analysera genomsnittligt pris per bränsletyp
plt.figure(figsize=(12, 6))
df.groupby('Fuel_Type')['Selling_Price'].mean().plot(kind='bar')
plt.title('Average Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()

# Konvertera kategoriska variabler till numeriska
le = LabelEncoder()
df_encoded = df.copy()
categorical_columns = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

for col in categorical_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Visa korrelationsmatris
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Skriv ut sammanfattande statistik
print("\nSummary Statistics:")
print("-" * 50)
print(df.describe())

# Detaljerad analys av kategoriska variabler
print("\nCategory Distributions:")
print("-" * 50)
for col in categorical_columns:
    print(f"\n{col.upper()} Distribution:")
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")

# Kontrollera extremvärden
print("\nExtreme Values Analysis:")
print("-" * 50)
numerical_columns = ['Year', 'Selling_Price', 'Kms_Driven']
for col in numerical_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    print(f"\nOutliers in {col}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")
    print(f"Range: {df[col].min()} to {df[col].max()}")
