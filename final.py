#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 20:54:07 2023
@author: Hari Prasad Yerranna Gari
"""

"""
# Importing required libraries
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis



def read_world_bank_data(filename):
    """
       Read World Bank data from a CSV file, transpose the DataFrame,
       and interchange the 'country' and 'date' columns.

       Parameters:
       - filename (str): The path to the CSV file containing World Bank data.

       Returns:
       - df (pd.DataFrame): DataFrame with country as column.
       - transposed (pd.DataFrame): DataFrame with year as column.
    """
    df = pd.read_csv(filename)
    transposed = df.copy()
    transposed[['country','date']] = transposed[['date','country']]
    transposed = transposed.rename(columns={'country': 'date', 'date': 'country'})

    return df,transposed


def heatmap(correlation_matrix):
    """
        Creates a heatmap for the correlation matrix.

        Parameters:
        - correlation_matrix (pd.DataFrame): The correlation matrix to visualize.

        Returns:
        - None
        """
    # Check if the correlation matrix has data
    if not correlation_matrix.empty:
        # Create a heatmap for the correlation matrix
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        # Add axis labels
        plt.xlabel('Indicators')
        plt.ylabel('Indicators')
        # Add color bar label
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Correlation Coefficient')
        plt.title('Correlation Matrix for Selected World Bank Indicators')
        plt.show()
    else:
        print("Correlation matrix is empty.")


def histogram(data):
    """
        Plots a histogram for the given data.

        Parameters:
        - data (pd.Series): The data for which the histogram is to be plotted.

        Returns:
        - None
        """
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {skew_column_name} with Skewness {skewness:.2f}')
    plt.xlabel(skew_column_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def bargraph(data):
    """
        Plots a bar graph for the given data, comparing 'access_to_electricity%' and 'individuals_using_internet%'.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data for the plot.

        Returns:
        - None
        """
    bar_width = 0.35
    bar_positions_agriculture = data['year'] - bar_width / 2
    bar_positions_forest = data['year'] + bar_width / 2

    # Plotting bar graph
    plt.figure(figsize=(12, 8))
    plt.bar(bar_positions_agriculture, data['access_to_electricity%'], width=bar_width,
            label='access_to_electricity%', color='blue')
    plt.bar(bar_positions_forest, data['individuals_using_internet%'], width=bar_width,
            label='individuals_using_internet%', color='green')

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.title('Access_to_electricity Vs individuals_using_internet')

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

def lineGraph(data):
    """
        Plots a line graph showing the agricultural land percentage over the years for selected countries.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data for the plot.

        Returns:
        - None
        """
    for country in ['India', 'South Africa', 'Pakistan', 'Germany', 'Italy']:
        country_data = data[data['country'] == country]
        plt.plot(country_data['date'], country_data['agricultural_land%'], label=country)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Agricultural Land Percentage')
    plt.title('Agricultural Land Percentage Over the Years')

    # Adding legend
    plt.legend()

    # Display a grid
    plt.grid(True)
    plt.show()

def scatterPlot(data):
    """
        Generates a scatter plot for the correlation between 'GDP_current_US' and 'birth_rate'.
        Also includes a linear regression line fitted to the data.

        Parameters:
        - data: DataFrame containing the relevant columns

        Returns:
        - None
        """
    plt.scatter(data['GDP_current_US'], data['birth_rate'], label='Data')
    # Fit a linear regression line using np.polyfit
    scatterplotData = data.dropna(subset=['GDP_current_US', 'birth_rate'])
    slope, intercept = np.polyfit(scatterplotData['GDP_current_US'], scatterplotData['birth_rate'], 1)
    plt.plot(scatterplotData['GDP_current_US'], slope * scatterplotData['GDP_current_US'] + intercept, color='red',
             label='Regression Line')
    plt.grid(True)
    plt.title('GDP_current_US and birth_rate correlation')
    plt.xlabel('GDP_current_US')
    plt.ylabel('birth_rate')
    plt.legend()
    plt.show()

def scatterPlotUsingSeaborn(data):
    """
        Generates a scatter plot using seaborn's lmplot for the correlation between 'GDP_current_US' and 'land_area'.
        Includes a linear regression line fitted to the data.

        Parameters:
        - data: DataFrame containing the relevant columns

        Returns:
        - None
        """
    scatterplotData = data.dropna(subset=['GDP_current_US', 'birth_rate'])
    sns.lmplot(data=scatterplotData, x='GDP_current_US', y='land_area', line_kws={'color': 'red'})
    plt.grid(True)
    plt.title('GDP_current_US% and land_area correlation')
    plt.show()

# Example Usage
filename = 'world_bank_development.csv'  # Replace with the actual file path
actualData, ag_df_transposed = read_world_bank_data(filename)

print("------------------Actual Data------------------------")
print(actualData.head())
print("------------------Transposed Data---------------------")
print(ag_df_transposed.head())

#statistical analysis
print(""" ------------------Statistical analysis-------------------  """)
#STATISTICAL METHODS

# Describe automatically computes basic statistics for all continous variables.
#  Any Nan values are automatically skipped in these statistics


"""
Mean (Average): The mean value of 37.53 suggests that, on average, the agricultural land percentage is around 37.53% 
across the countries and years in your dataset.

Standard Deviation (std): The standard deviation of 20.54 indicates the extent of variability or dispersion in the 
agricultural land percentages. A higher standard deviation implies greater variability around the mean.

Minimum (min): The minimum value of 0.26 is the smallest agricultural land percentage observed in your dataset.

25th Percentile (25%): The 25th percentile (Q1 or first quartile) value of 21.44 represents the threshold below which 
25% of the agricultural land percentages fall.

50th Percentile (50% or Median): The median value of 37.69 is the middle point of your dataset when arranged 
in ascending order. It indicates that half of the agricultural land percentages are below 37.69, and half are above.

75th Percentile (75%): The 75th percentile (Q3 or third quartile) value of 51.44 represents the threshold below 
which 75% of the agricultural land percentages fall.

Maximum (max): The maximum value of 93.44 is the largest agricultural land percentage observed in your dataset.

"""
describes = actualData['agricultural_land%'].describe()
print("----Describes")
print(describes)

actualData['date'] = pd.to_datetime(actualData['date'], format='%m/%d/%y', errors='coerce')
actualData['month'] = actualData['date'].dt.month
actualData['year'] = actualData['date'].dt.year
actualData['month'] = actualData['date'].dt.strftime('%B')
print(f"Total amount of countries: {actualData['country'].nunique()}")
print(f"Start date: {actualData['date'].min()}")
print(f"End date: {actualData['date'].max()}")
#count median temp for 2020 and 2021. to satisfly curiosity:
print(f"Median birth rate in 2000: {actualData[actualData['date'].dt.year == 2000]['birth_rate'].median()}")
print(f"Median birth rate in 2021: {actualData[actualData['date'].dt.year == 2021]['birth_rate'].median()}")

print("----Skewness")
# Define the column name for skewness
skew_column_name = 'control_of_corruption_estimate'

# Replace missing values with NaN
actualData[skew_column_name] = pd.to_numeric(actualData[skew_column_name], errors='coerce')

# Calculate skewness
skewness = skew(actualData[skew_column_name].dropna())
print(f"Skewness for {skew_column_name}: {skewness}")
histogram(actualData[skew_column_name].dropna())

print("----Kurtosis")
data_for_kurtosis = actualData['GDP_current_US'].dropna()
kurtosis_value = kurtosis(data_for_kurtosis, fisher=False)
print("Kurtosis:", kurtosis_value)


#Correlation
print("------------------------------- Correlation --------------------------")
"""
there is a moderate positive linear relationship between the percentage of people with access to electricity 
(access_to_electricity%) and the percentage of individuals using the internet (individuals_using_internet%)
"""
correlation = actualData['access_to_electricity%'].corr(actualData['individuals_using_internet%'])
print("Correlation between access_to_electricity% and individuals_using_internet%", correlation)

#Heat map
# Select a few indicators for analysis
selected_indicators = ['land_area', 'agricultural_land%', 'forest_land%']
# Extract the relevant data for the selected indicators
df_selected_indicators = actualData[selected_indicators]
# Calculate the correlation matrix
correlation_matrix = df_selected_indicators.corr()
heatmap(correlation_matrix)

#barGraph
selected_columns = ['country','date','renewvable_energy_consumption%','land_area','birth_rate','agricultural_land%','forest_land%','access_to_electricity%','GDP_current_US','individuals_using_internet%']
filteredData = actualData[selected_columns]
filtered_df = filteredData[(filteredData['date'].dt.year >= 2000) & (filteredData['date'].dt.year <= 2020)]
bargraph_data = filtered_df[filtered_df['country'] == 'India']

# Access_to_electricity Vs individuals_using_internet

bargraph_data = bargraph_data.copy()
bargraph_data['year'] = bargraph_data['date'].dt.year
bargraph(bargraph_data)

#Line Graph
# Agdriculture land of different countries
# Plotting separate lines for each country
lineGraphData = filteredData[(filteredData['date'].dt.year >= 2005) & (filteredData['date'].dt.year <= 2020)]
lineGraph(lineGraphData)

#Scatter Plot
# GDP_current_US and birth_rate correlation
scatterPlot(actualData)

#Scatterplot
# GDP_current_US% and land_area correlation'
scatterPlotUsingSeaborn(actualData)









