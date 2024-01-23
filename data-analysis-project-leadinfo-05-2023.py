## Company X wants to change their strategy by focussing on new industries.
## They have a general idea of the industries they want to focus on: growth industries like logistics and tech.

## Research questions:
# By which kind of companies is company X' website visited? In terms of page visits, visit duration and last visit date.
# Is this the same industries as company X plans to target?
# Can we find other industries that are worth keeping in mind? Based on website visits in combination with company characteristics such as revenue, amount of employees and location.
# Where to reach these companies? Location, social media...

import pandas as pd

# Read the data file
data = pd.read_csv('C:\\Users\\laura\\My Drive\\Data Analytics Bootcamp Allwomen\\_DA_course\\finalProject_\\Leadinfo_export_120523.csv', sep=';')
data.head()

# Make a copy
data_copy = data.copy()

# Show info of the dataset, including datatype, null values and amount of rows/columns
data.info()

# Check for duplicates
data[data.duplicated()]

# Number of null values per column
num_nulls_columns = data.isnull().sum()
num_nulls_columns

# Total amount of values per column
num_total_columns = len(data)
num_total_columns

# Calculate the percentage of null values per column
pct_nulls_columns = (100 * num_nulls_columns / num_total_columns).round(0).astype('Int64')

# Print the results
print("Percentage null values per kolom:")
print(pct_nulls_columns)

#Columns where percentage null values is higher than 50%
pct_nulls_columns[pct_nulls_columns > 50]

# Number of null values per row
num_nulls_rows = data.isnull().sum(axis=1)

# Total amount of values per row
num_total_rows = data.shape[1]

# Calculate the percentage of null values per row
pct_nulls_rows = 100 * num_nulls_rows / num_total_rows

# Amount of rows where percentage null values is higher than 50%
len(pct_nulls_rows[pct_nulls_rows > 50])

# Shape of dataset
data.shape

# Check info about most common industries
data[['Branch naam nationaal', 'Branche']].groupby('Branche')['Branch naam nationaal'].value_counts().sort_values(ascending=False)[:10]

# Rename column
data['Paginas'] = data["Pagina's"]

# Check type of date column
data['Laatste bezoek']

# Convert column to datetime column
import datetime
from datetime import datetime
from datetime import time

data['Last_visit_date'] = pd.to_datetime(data['Laatste bezoek'])
data['Last_visit_date'] = data['Last_visit_date'].dt.date
data['Last_visit_date'] = data['Last_visit_date'].astype('datetime64[ns]')
data['Last_visit_date']

# Convert other column to datetime column
data['First_visit_date'] = pd.to_datetime(data['Eerste bezoek'])
data['First_visit_date'] = data['First_visit_date'].dt.date
data['First_visit_date'] = data['First_visit_date'].astype('datetime64[ns]')
data['First_visit_date']

# Create new column with weekdays
data['Last_visit_weekday'] = data['Laatste bezoek'].dt.strftime('%A')
data['Last_visit_weekday']

# Create new column with weekdays
data['First_visit_weekday'] = data['Eerste bezoek'].dt.strftime('%A')
data['First_visit_weekday']

# Set day of today
data['Date_today'] = '2023-05-12'

# Change column to datetime column 
data['Date_today'] = data['Date_today'].astype('datetime64[ns]')

# Calculate days since last visit
d0 = data['Date_today']
d1 = data['Last_visit_date']
data['Days_since_last_visit'] = d0 - d1

# Change data type for data cleaning
data['Days_since_last_visit'] = data['Days_since_last_visit'].astype(str)
data['Days_since_last_visit']

# Strip text from column to convert to int
data['Days_since_last_visit'] = data['Days_since_last_visit'].str.replace(" days", "")
data['Days_since_last_visit']

# Replace missing values by NaN values
data['Days_since_last_visit'] = data['Days_since_last_visit'].replace("NaT", np.nan)

# Change data type to integer
data['Days_since_last_visit'] = data['Days_since_last_visit'].astype(float).astype('Int64')

# Amount of missing values for columns
data['Days_since_last_visit'].isnull().sum()

data['Bezoekduur'].isnull().sum()

data['Sessies'].isnull().sum()

data['Paginas'].isnull().sum()

data['Dagen'].isnull().sum()

data['Omzetvolume'].isnull().sum()

data['Medewerkers'].isnull().sum()

# Drop missing values for column 'Bezoekduur'
data = data.dropna(axis=0, subset=['Bezoekduur'])

# Change data type for data cleaning
data['Leadscore'] = data['Leadscore'].astype(str)

# Remove special characters
data['Leadscore'] = data['Leadscore'].str.replace("%", "")

# Change data type to integer
data['Leadscore'] = data['Leadscore'].astype(int)

# Replace value
data['Omzetvolume'] = data['Omzetvolume'].str.replace(".", "")

# Change data type for data cleaning
data['Omzetvolume'] = data['Omzetvolume'].astype(str)

# Replace value
data['Omzetvolume'] = data['Omzetvolume'].str.replace("$", "$ ")

# Split special characters and numbers from column
data[['Currency', 'Revenue']] = data['Omzetvolume'].str.split(' ', 1, expand=True)

# Check types of currencies
data['Currency'].value_counts()

# Select only currencies we want to use for analysis
data = data[(data['Currency'] == '$') | (data['Currency'] == '€') | (data['Currency'] == '£') | (data['Currency'] == 'nan')]

# Check types of currencies again
data['Currency'].value_counts()

# Replace values and prepare for converting
data['Currency'] = data['Currency'].replace('€', 'EUR')
data['Currency'] = data['Currency'].replace('£', 'GBP')
data['Currency'] = data['Currency'].replace('$', 'USD')

# Change data type for data cleaning
data['Currency'] = data['Currency'].astype(str)

# Replace missing values with NaN
data['Revenue'] = data['Revenue'].replace("", np.NaN)

# Check values in column
data['Revenue'].value_counts(dropna=False)

# Strip column
data['Revenue'] = data['Revenue'].str.strip()

# Replace values
data['Revenue'] = data['Revenue'].str.replace(',', '.').astype(float)

# Change data type to integer
data['Revenue'] = data['Revenue'].astype('Int64')
data['Revenue']

# Install currency converter
get_ipython().system('pip install currencyconverter')

# Select currencies for convertion
currencies = data.Currency.unique().tolist()
currencies = dict.fromkeys(currencies, pd.NA)

# Assign currencies (from 05-05-2023)
currencies['EUR'] = 1
currencies['USD'] = 1.1014
currencies['GBP'] = 0.87378

# Create function for converting values
from currency_converter import CurrencyConverter
c = CurrencyConverter()
for key in currencies:
   try:
      currencies[key] = c.convert(1, 'EUR', key)
   except:
      pass

# Convert values from column 'Revenue' based on currency
data['Revenue_conv'] = data.apply(lambda x: x.Revenue * currencies[x.Currency], axis=1)

# Check converted data
data['Revenue_conv']

# Convert column to numeric data type
data['Revenue_conv'] = pd.to_numeric(data['Revenue_conv'], errors='coerce')
data['Revenue_conv']

# Change data type to integer
data['Revenue_conv'] = data['Revenue_conv'].round(0).astype('Int64')
data['Revenue_conv']

# Overview of outliers for column
data.boxplot(['Bezoekduur'])

# Import function for outliersbox
import sys
from Functions_EDA import *

# Overview of outliers for columns
OutLiersBox(data, 'Bezoekduur')

OutLiersBox(data, 'Paginas')

OutLiersBox(data, 'Sessies')

OutLiersBox(data, 'Dagen')

OutLiersBox(data, 'Days_since_last_visit')

# Reset index so the outlier treatment function will work
data = data.reset_index()

# Outlier treatment for column
data = outlier_treatment(data, 'Bezoekduur')

# Outlier treatment for column
data = outlier_treatment(data, 'Sessies')

# Outlier treatment for column
data = outlier_treatment(data, 'Paginas')

# Outlier treatment for column
data = outlier_treatment(data, 'Days_since_last_visit')

# Check for outliers in column (no treatment executed)
data_cleaned = outlier_treatment(data, 'Dagen')

# See if values in this column are really outliers
data['Dagen'].value_counts()

# Change data type to integer
data['Leadscore'] = data['Leadscore'].astype('Int64')

# Plot scatterplot to check for correlations
import plotly.graph_objects as go

fig_leadscore_bezoekduur = go.Figure(data=go.Scatter(x=data['Bezoekduur'],
                                y=data['Leadscore'],
                                mode='markers',
                                marker_color=data['Leadscore'])
                                     )
fig_leadscore_bezoekduur.show()

fig_leadscore_paginas = go.Figure(
    data=go.Scatter(x=data['Paginas'],
    y=data['Leadscore'],
    mode='markers',
    marker_color=data['Leadscore'])
)
fig_leadscore_paginas.show()

fig_leadscore_dagen = go.Figure(
    data=go.Scatter(x=data['Dagen'],
    y=data['Leadscore'],
    mode='markers',
    marker_color=data['Leadscore'])
)
fig_leadscore_dagen.show()

fig_leadscore_sessies = go.Figure(
    data=go.Scatter(x=data['Sessies'],
    y=data['Leadscore'],
    mode='markers',
    marker_color=data['Leadscore'])
)
fig_leadscore_sessies.show()

# Create new dataframe with only the variables we want to compare
data_new = data[['Leadscore', 'Bezoekduur','Sessies', 'Paginas', 'Dagen', 'Days_since_last_visit', 'Revenue_conv', 'Medewerkers']]

# Calculate correlation coefficient for all combinations of variables
data_corr = data_new.corr().round(3)
data_corr

# Show scores in heatmap for overview
sns.heatmap(data_corr,
            vmin=-1, vmax=1, #Defining the max and min value of out heatmap 
            cmap="vlag", #Defining the color pallet
            annot=True)

#Dark red means a strong positive correlation.
#Dark blue means a strong negative correlation.

### Prior knowledge:
#Correlation between 0.7 and 0.9 means there is a strong relationship between the variables.
#Correlation between 0.5 and 0.7 means there is a moderate relationship  between the variables.
#Correlation between 0.3 and 0.5 means there is a weak relationship between the variables.

# Check for amount of unique values in column
data['Branche'].nunique()

# Calculate average, replace missing values and change data type
data['Revenue_branche_avg'] = data.groupby('Branche')['Revenue_conv'].transform('mean').round(0)

data['Revenue_branche_avg'] = data['Revenue_branche_avg'].replace("nan", np.NaN)

data['Revenue_branche_avg'] = data['Revenue_branche_avg'].astype('Int64')
data['Revenue_branche_avg']

# Calculate average, replace missing values and change data type
data['Medewerkers_branche_avg'] = data.groupby('Branche')['Medewerkers'].transform('mean').round()

data['Medewerkers_branche_avg'] = data['Medewerkers_branche_avg'].replace("nan", np.NaN)

data['Medewerkers_branche_avg'] = data['Medewerkers_branche_avg'].astype('Int64')
data['Medewerkers_branche_avg']

# Calculate average, replace missing values and change data type
data['Bezoekduur_branche_avg'] = data.groupby('Branche')['Bezoekduur'].transform('mean').round()

data['Bezoekduur_branche_avg'] = data['Bezoekduur_branche_avg'].replace("nan", np.NaN)

data['Bezoekduur_branche_avg'] = data['Bezoekduur_branche_avg'].astype('Int64')
data['Bezoekduur_branche_avg']

# Calculate average, replace missing values and change data type
data['Sessies_branche_avg'] = data.groupby('Branche')['Sessies'].transform('mean').round()

data['Sessies_branche_avg'] = data['Sessies_branche_avg'].replace("nan", np.NaN)

data['Sessies_branche_avg'] = data['Sessies_branche_avg'].astype('Int64')
data['Sessies_branche_avg']

# Calculate average, replace missing values and change data type
data['Paginas_branche_avg'] = data.groupby('Branche')['Paginas'].transform('mean').round()

data['Paginas_branche_avg'] = data['Paginas_branche_avg'].replace("nan", np.NaN)

data['Paginas_branche_avg'] = data['Paginas_branche_avg'].astype('Int64')
data['Paginas_branche_avg']

# Calculate average, replace missing values and change data type
data['Dagen_branche_avg'] = data.groupby('Branche')['Dagen'].transform('mean').round()

data['Dagen_branche_avg'] = data['Dagen_branche_avg'].replace("nan", np.NaN)

data['Dagen_branche_avg'] = data['Dagen_branche_avg'].astype('Int64')
data['Dagen_branche_avg']

# Calculate average, replace missing values and change data type
data['Days_last_visit_branche_avg'] = data.groupby('Branche')['Days_since_last_visit'].transform('mean').round()

data['Days_last_visit_branche_avg'] = data['Days_last_visit_branche_avg'].replace("nan", np.NaN)

data['Days_last_visit_branche_avg'] = data['Days_last_visit_branche_avg'].astype('Int64')
data['Days_last_visit_branche_avg']

# Check values
data['Medewerkers']

# Create groups based on amount of employees
data.loc[(data['Medewerkers'] < 50), 'Mdw_group'] = '0-50'
data.loc[(data['Medewerkers'] >= 50) & (data['Medewerkers'] < 500), 'Mdw_group'] = '50-500'
data.loc[(data['Medewerkers'] >= 500) & (data['Medewerkers'] < 1000), 'Mdw_group'] = '500-1000'
data.loc[(data['Medewerkers'] > 1000), 'Mdw_group'] = '1000+'

# Check groups
data['Mdw_group']

# Change data type to integer
data['Revenue_conv'] = data['Revenue_conv'].round(0).astype('Int64')
data['Revenue_conv']

data['Bezoekduur'] = data['Bezoekduur'].astype('Int64')
data['Bezoekduur']

data['Paginas'] = data['Paginas'].astype('Int64')
data['Paginas']

data['Sessies'] = data['Sessies'].astype('Int64')
data['Sessies']

data['Days_since_last_visit'] = data['Days_since_last_visit'].astype('Int64')
data['Days_since_last_visit']

data['Dagen'] = data['Dagen'].astype('Int64')
data['Dagen']

data['Medewerkers'] = data['Medewerkers'].astype('Int64')
data['Medewerkers']

# Save dataset as csv
data.to_csv('../../DA_final_project.csv')

# Select rich data from dataset
rich_data = data[(data['Branche'].notnull()) & (data['Paginas'] > 1) & (data['Bezoekduur']  > 60) & (data['Days_since_last_visit'] < 365)]
len(rich_data)

# Save dataset as csv
rich_data.to_csv('../../DA_final_project_rich_data.csv')
