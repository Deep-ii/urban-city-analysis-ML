import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = "final_crime_dataset.csv"
df = pd.read_csv(file_path)

# Convert categorical columns to numerical using label encoding
df_encoded = df.copy()
for col in df.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes

# Normalize Data for Box Plot
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

# Box Plot - Passenger Count vs Other Features
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot Passenger Count separately
sns.boxplot(data=df_scaled[['Passenger Count']], ax=axes[0], palette="Set2")
axes[0].set_title("Box Plot of Passenger Count (Scaled)")

# Plot remaining numerical variables
sns.boxplot(data=df_scaled.drop(columns=['Passenger Count']), ax=axes[1], palette="Set3")
axes[1].set_title("Box Plot of Other Features (Scaled)")

plt.tight_layout()
plt.show()

# Scatter Plot - Vehicle Count vs. Accident Severity
if 'Vehicle Count' in df.columns and 'Accident Severity' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Vehicle Count'], df['Accident Severity'], alpha=0.6, color='blue', marker='o')
    plt.title("Scatter Plot: Vehicle Count vs. Accident Severity")
    plt.xlabel("Vehicle Count")
    plt.ylabel("Accident Severity")
    plt.grid(True)
    plt.show()
else:
    print("Error: Required columns not found in the dataset.")

# Histograms for Vehicle Count & Weather Condition
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram for Vehicle Count
if 'Vehicle Count' in df.columns:
    axes[0].hist(df['Vehicle Count'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    axes[0].set_title("Histogram: Distribution of Vehicle Count")
    axes[0].set_xlabel("Vehicle Count")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
else:
    print("Error: 'Vehicle Count' column not found in the dataset.")

# Histogram for Weather Condition
if 'Weather Condition' in df.columns:
    axes[1].hist(df['Weather Condition'], bins=10, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_title("Histogram: Distribution of Weather Condition")
    axes[1].set_xlabel("Weather Condition")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
else:
    print("Error: 'Weather Condition' column not found in the dataset.")

plt.tight_layout()
plt.show()

# Compute the correlation matrix for all numerical features
correlation_matrix = df.corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Overall Feature Correlation Heatmap")
plt.show()

# Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Crime Category vs. Vehicle Count
sns.scatterplot(x=df["Vehicle Count"], y=df["Crime Category"], alpha=0.6, color="red", ax=axes[0, 0])
axes[0, 0].set_title("Crime Category vs. Vehicle Count")
axes[0, 0].set_xlabel("Vehicle Count")
axes[0, 0].set_ylabel("Crime Category")
axes[0, 0].grid(True)

# Accident Severity vs. Weather Condition
sns.scatterplot(x=df["Weather Condition"], y=df["Accident Severity"], alpha=0.6, color="blue", ax=axes[0, 1])
axes[0, 1].set_title("Accident Severity vs. Weather Condition")
axes[0, 1].set_xlabel("Weather Condition")
axes[0, 1].set_ylabel("Accident Severity")
axes[0, 1].grid(True)

# Passenger Count vs. Time of Day
sns.scatterplot(x=df["Hour"], y=df["Passenger Count"], alpha=0.6, color="green", ax=axes[1, 0])
axes[1, 0].set_title("Passenger Count vs. Time of Day")
axes[1, 0].set_xlabel("Hour of the Day")
axes[1, 0].set_ylabel("Passenger Count")
axes[1, 0].grid(True)

# Vehicle Count vs. Crime Severity (With Regression Line)
sns.regplot(x=df["Vehicle Count"], y=df["Accident Severity"], scatter_kws={"alpha":0.6}, line_kws={"color":"red"}, ax=axes[1, 1])
axes[1, 1].set_title("Vehicle Count vs. Crime Severity")
axes[1, 1].set_xlabel("Vehicle Count")
axes[1, 1].set_ylabel("Accident Severity")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
