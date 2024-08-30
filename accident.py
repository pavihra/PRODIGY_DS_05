import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px

df = pd.read_csv('traffic_accident_data.csv')

print(df.head())

df = df.dropna()

df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour


plt.figure(figsize=(10, 6))
sns.countplot(x='Hour', data=df, palette='viridis')
plt.title('Number of Accidents by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Weather', data=df, palette='coolwarm')
plt.title('Number of Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Road_Condition', data=df, palette='magma')
plt.title('Number of Accidents by Road Condition')
plt.xlabel('Road Condition')
plt.ylabel('Number of Accidents')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Weather', y='Severity', data=df, palette='muted')
plt.title('Severity of Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Severity')
plt.show()

fig = px.scatter(df, x='Weather', y='Severity', color='Road_Condition',
                 title='Accidents by Weather and Road Condition')
fig.show()

df['Latitude'] = np.random.uniform(low=34.0, high=42.0, size=len(df))
df['Longitude'] = np.random.uniform(low=-118.0, high=-74.0, size=len(df))

accident_map = folium.Map(location=[37.0, -95.0], zoom_start=4)

heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(accident_map)

accident_map.save("accident_hotspots.html")
