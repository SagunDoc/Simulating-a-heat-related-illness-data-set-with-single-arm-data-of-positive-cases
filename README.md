# 🧪 Heat Illness Synthetic Dataset Simulation

This repository contains a Jupyter Notebook that simulates a synthetic dataset for modeling **heat-related illness (HRI)**, particularly in **military or field settings**. The synthetic data can be used for developing and testing predictive models. It is to be understood with caution that the data is simulated from the original 27 cases of heat related illness and thus introduce bias. 

---

## 📌 Project Overview

The notebook provides a pipeline to:
- Load real positive HRI cases
- Simulate synthetic **positive cases** (HRI occurred)
- Simulate **negative cases** (no HRI) by incorporating protective modifications
- Combine, shuffle, and export a balanced dataset for analysis or model training

---

## 📦 Step-by-Step Breakdown

### ✅ Step 1: Load Real Positive Cases Dataset

We begin by importing required libraries and loading real-world confirmed heat illness cases. This forms the foundation for our synthetic data generation.

Key actions:
- Set random seed for reproducibility
- Load CSV with Latin-1 encoding to handle special characters
- Store original data in `df_real_positive`

```python
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Set random seed for reproducibility
np.random.seed(42)

# Load your original positive cases data
df_real_positive = pd.read_csv("D:/Predictive model HRI/heat.csv", encoding='latin1')

```
### 🔁 Step 2: Simulate Positive Cases (Heat Illness Occurred)

We simulate synthetic positive cases by:

- Sampling categorical columns based on their observed frequency distribution
- Generating numerical columns using normal distribution based on the real data's mean and standard deviation
- Ensuring values are clipped to remain within realistic minimum and maximum bounds
```python
def simulate_positive_cases(original_df, n=100):
    """Simulate positive cases (heat illness occurred)"""
    simulated = {}
    categorical_cols = ['rank', 'water', 'urine', 'exercise', 'warmup', 'event']
    
    for col in original_df.columns:
        if col.lower() in categorical_cols:
            counts = original_df[col].value_counts(normalize=True)
            simulated[col] = np.random.choice(counts.index, size=n, p=counts.values)
        elif col.lower() != 'event':
            simulated[col] = np.random.normal(
                original_df[col].mean(),
                original_df[col].std(),
                n
            ).clip(
                original_df[col].min(),
                original_df[col].max()
            )
    
    simulated_df = pd.DataFrame(simulated)
    simulated_df['event'] = 0  # Mark as positive cases
    return simulated_df

# Generate 100 simulated positive cases
df_sim_positive = simulate_positive_cases(df_real_positive, n=100)
```
### 🛡️ Step 3: Simulate Negative Cases (No Heat Illness)

We simulate synthetic **negative cases** by modifying the previously generated positive cases to reflect **protective measures** that reduce the risk of heat-related illness. These modifications include:

- 💧 Increased water intake  
- 🔄 Pre-exercise warm-up  
- 🌡️ Lower environmental temperature  
- 💦 Reduced humidity  
- 🚽 Improved urine output (indicating hydration)

This helps generate plausible non-HRI samples for a balanced binary classification dataset.

```python
def simulate_negative_from_positive(df_pos, n=100):
    """Simulate negative cases by modifying positive cases with protective factors"""
    n_samples = min(n, len(df_pos))
    neg = df_pos.sample(n_samples, replace=n_samples > len(df_pos)).copy()
    
    # Modify urine output (simulate better hydration)
    if 'urine' in neg:
        neg['urine'] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Simulate increased water intake
    if 'water' in neg:
        water_increase = np.random.normal(0.5, 0.2, n_samples)
        neg['water'] = (neg['water'] + water_increase).clip(
            df_pos['water'].min(), 
            df_pos['water'].max() * 1.5
        )
    
    # Simulate universal warm-up participation
    if 'warmup' in neg:
        neg['warmup'] = 0  # All did warm-up
    
    # Simulate lower ambient temperature
    if 'temperature' in neg:
        temp_decrease = np.random.normal(1.0, 0.5, n_samples)
        neg['temperature'] = (neg['temperature'] - temp_decrease).clip(
            df_pos['temperature'].min() - 2,
            df_pos['temperature'].max()
        )
    
    # Simulate lower humidity
    if 'humidity' in neg:
        humidity_decrease = np.random.normal(5.0, 2.0, n_samples)
        neg['humidity'] = (neg['humidity'] - humidity_decrease).clip(
            df_pos['humidity'].min(),
            df_pos['humidity'].max()
        )
    
    # Label as non-HRI (negative event)
    neg['event'] = 1

    # If not enough samples, recursively create more
    if n_samples < n:
        additional_needed = n - n_samples
        additional_cases = simulate_negative_from_positive(df_pos, n=additional_needed)
        neg = pd.concat([neg, additional_cases], ignore_index=True)
    
    return neg

# Generate 100 simulated negative cases
df_sim_negative = simulate_negative_from_positive(df_sim_positive, n=100)
```
### 📊 Step 4: Combine, Shuffle, and Save Dataset

Once positive and negative cases are simulated, they are combined into a single dataset. To prevent learning order bias, the rows are shuffled. The final dataset is saved as a `.csv` file for further modeling or analysis.
```python
# Combine positive and negative cases
df_final = pd.concat([df_sim_positive, df_sim_negative], ignore_index=True)

# Shuffle the dataset
df_final = shuffle(df_final, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = r"D:/Predictive model HRI/synthetic_heat.csv"
df_final.to_csv(output_path, index=False)

# Output dataset details
print(f"Successfully saved synthetic dataset to: {output_path}")
print(f"Dataset shape: {df_final.shape}")
print("Class distribution:")
print(df_final['event'].value_counts())
```
## 📁 Output

- **Filename:** `synthetic_heat.csv`
- **Records:** 200 (100 simulated positive + 100 simulated negative)
- **Balanced Binary Target Variable:** `event`
  - `0` → Heat Illness Occurred
  - `1` → No Heat Illness
- **Columns:** Match real dataset schema (e.g., temperature, humidity, rank, urine, warmup, etc.)
- **Format:** CSV, ready for ML pipelines
## 📚 Use Cases

This synthetic dataset can support a wide range of machine learning, data science, and public health research applications:

- 🧠 **Train predictive models** for heat-related illness in field or military environments
- 🔬 **Augment small or sensitive datasets** using realistic simulations
- 📊 **Perform exploratory data analysis (EDA)** without exposing confidential data
- 🛡️ **Benchmark risk mitigation strategies** (e.g., water intake, warm-up)
- 🤖 **Test classification algorithms** in imbalanced and balanced scenarios

## 🛠️ Dependencies

This simulation requires the following Python packages:
pandas
numpy
scikit-learn (later for ML model)

## 📜 License

This project is released for **educational and research purposes only**.

Please cite or acknowledge appropriately if this code or dataset is used in:

- Academic publications  
- Training materials  
- Institutional reports  

### **Disclaimer:** The synthetic dataset is not real and should not be used for clinical or operational decision-making.
