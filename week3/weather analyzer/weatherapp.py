# Install necessary libraries if not already installed
# !pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Läs in CSV-filen
data = pd.read_csv('weather_data.csv', parse_dates=['Date'])

# Hantera saknade värden genom att fylla med genomsnitt eller ta bort rader
data.fillna(method='ffill', inplace=True)

# Skapa en ny kolumn för månad från datumet
data['Month'] = data['Date'].dt.to_period('M')

# Beräkna genomsnittlig temperatur per månad
monthly_avg_temp = data.groupby('Month')['Temperature'].mean()
print(monthly_avg_temp)
