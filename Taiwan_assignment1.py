# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:06:43 2023

@author: Paula Mörstedt

Lecture: EASYS
Assignment 1
"""
#%% Import of all modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% PART 1
"""1. Read the dates and monthly temperature anomalies data into Python (3 points)."""

#Reading the file
taiwan_temperature = open("taiwan_temperature.txt", 'r')

lines = taiwan_temperature.readlines()

taiwan_temperature.close()

#Processing the data, creating a function that can process the data 
def read_file(value):
    #function that reads two columns from a txt file with tab spaced data into two lists
    #input: cut off value for data lines that are not necessary to read
    #output: two lists with the data from the first two columns
    taiwan_data = lines[value:] #read the relevant part of the txt document (cut the first 97 irrelevant lines)
    years = []
    temp_monthly = [] 
    
    #reading the data into two lists for years and monthly CO2 data    
    for line in taiwan_data: 
        data_list = line.split()
        year = int(data_list[0])
        years.append(year)
        
        temp_value = float(data_list[2])
        temp_monthly.append(temp_value)
    return years, temp_monthly

years, temp_monthly = read_file(97)

  
#%%
"""2. Make a subset of the data to start in January 1960 (2 points)"""

#first option

for element in years:                   #checking at which index the year 1960 starts 
    if element == 1960:
        index1960 = years.index(element)
        break

years1960 = years[index1960:] #slicing the list from the year 1960 on using the index as defined above
temp_monthly1960 = temp_monthly[index1960:]

#second option, when you know at which line the year 1960 starts in the txt document
years_1960, temp_monthly_1960 = read_file(985)
    
#%%
""" 3. Perform a sanity check by calculating the share of missing values (2 points) Write a clear print statement to your console to report your result.
Be careful, you only want to do a sanity check on the data that you are using, not the whole data set!"""

#Counting every Value that has an NaN entry
count = 0
for element in temp_monthly1960:
    if np.isnan(element) == True:
        count += 1

#Calculating the share of missing values
percentage = round(count/len(temp_monthly1960)*100,2)
        
print("There are", count, "values missing. The share of missing values in the monthly temperature anomalies data is", percentage, "percent.")

#%%
"""4. Convert the monthly temperature anomalies to annual temperature anomalies (3 points):
If your final year does not contain data for all months, exclude it from your analysis."""

temp_annualy = []#empty list, will contain the annual temperature anomalys 

for i in range(0,len(temp_monthly1960),12):
    temporary_years = temp_monthly1960[i:i+12]
    # Check if temporary_years is not empty and does not contain only NaN values
    if not np.all(np.isnan(temporary_years)):
    # Calculate the mean while ignoring NaN values
        mean_value = np.nanmean(temporary_years)
        temp_annualy.append(mean_value)
    else:
        #if the data is not sufficient you have to delete the years from the years list, otherwise the annual data is not assigned to the corresponding year         
        del years[i:i+12]
        
years_singled = list(set(years1960))

#deleting the last elements of both lists, because for 2023 there is only one data point (month January)
del temp_annualy[-1]
del years_singled[-1]
        

#%%
"""5. Convert the annual temperature anomalies to absolute temperature values (2 points):
Estimated Jan 1951-Dec 1980 absolute temperature (C): 19.71 """

#defining a function for that, because I use the same operation in Part 2 Task 1
def anomalies_to_temp(mylist):
    #function that calculated absolute temperature values based on temperature anomalies
    #input: list with the annual average anomalies
    #output: a list with the absolute annual temperature averages
    abs_temp_annualy = []
    
    for i in range(len(mylist)):
        abs_temp = mylist[i] + 19.71
        abs_temp_annualy.append(abs_temp)
        
    return abs_temp_annualy

abs_temp_annualy = anomalies_to_temp(temp_annualy)

#%%
"""6. Plot the time series of absolute temperature (4 Points): The code should save the figure to a file called ‘Task_1.png’."""
#plotting dots and a line
plt.plot(years_singled, abs_temp_annualy,"x", markersize = 3, label = "Absolute values")
plt.plot(years_singled, abs_temp_annualy,"-", lw = 0.5, color = 'black', label = "Progression through the years")

#naming the labels and creating a legend
plt.xlabel(f"Year ({years_singled[0]} - {years_singled[-1]})")
plt.ylabel("Mean of the absolute annual temperature \n for Taiwan [°C]")
plt.legend()
plt.grid(True)

#saving and showing 
plt.savefig("Task 1.png", dpi = 300)
plt.show()

#%% PART 2
"""Task 1: Calculate an 11-year running mean of your temperature data (3 points).
To calculate the running mean, select a central value with an equal number of data on either side. 
Therefore, your first 5 and final 5 years should not have reported running mean values."""


temp_anom_runningmean = []#empty list, will contain the running means from year 1965 - 2017 

# using a loop to build the running means from the annual temperature anomalys through the years 1965 - 2017
for i in range(5,len(temp_annualy)-5):
    temporary_meandata = temp_annualy[i:i+11]
    mean_value = np.nanmean(temporary_meandata)
    temp_anom_runningmean.append(mean_value)
    
# slicing the years list into the years that have an assigned running mean
years_runningmean = years_singled[5:-5]

# converting the running anomaly means into temperature data using my own function, defined in Part 1 
temp_abs_runningmean = anomalies_to_temp(temp_anom_runningmean)

#%% 
"""Task 2: Calculate the temperature average for each decade (my comment: using the temp_annualy data) (2 points).
Using the values of the decade only (1960-1969 for example). 
If you have an incomplete decade, exclude it from your results."""

#calculating if I have an incomplete decade with the % operation
#deleting the incomplete decade years from the years list and the annual temp list
decade_value = len(years_singled)%10
if decade_value >= 0:
    years_decades = years_singled[:-decade_value]
    abs_temp_annualy_decades = abs_temp_annualy[:-decade_value]
else:
    years_decades = years_singled.copy()
    abs_temp_annualy_decades = abs_temp_annualy.copy()

# Calculate the mean value per decade based on the annual mean temp
temp_decade = []
for i in range(0,len(years_decades),10):
    temporary_years = abs_temp_annualy_decades[i:i+10]
    mean_value = np.nanmean(temporary_years)
    temp_decade.append(mean_value)

#%% 
"""Task 3: Calculate the linear fit of these decadal averages from the 1960s to the 1970s, 
the 1960s to the 1980s (not using your 1970s result), the 1960s to the 1990s 
(not using your 1970s or 1980s results), etc… (5 Points).
You can calculate this by calculating the slope and intercept 
between two points or using an external library. 
Visually assess (via plotting) which linear fit best represents the temperature data 
(i.e. which one looks the most representative). 
Plot these linear fits out the end of your data to make comparing them easier. 
NOTE, these plots do not have to be submitted."""

decades= list(range(1965,2016,10))#using the middle of each decade, because I calculated the average of the decade
slopes = []
intercepts = []

#temperature data are the y values and years are the x values
#Looping over the decades to get the slope and the intercepts
for i in range(1,len(decades)):
    slope = (temp_decade[i]-temp_decade[0])/(decades[i]-decades[0])
    slopes.append(slope)
    intercept = temp_decade[0] - slope * decades[0]
    intercepts.append(intercept)
  
#Defining the linear function
def f(x,m,b):
    y = m*x + b
    return y

#Create an array of x-values to calculate the y-values for the fit
x_values = np.linspace(1950,2025,100)

# Calculate y-values for each decade
y_values = [f(x_values, slopes[i], intercepts[i]) for i in range(len(decades) - 1)]

# Plot the linear functions
decade_names = list(range(1960,2011,10))
for i in range(len(y_values)):
    plt.plot(x_values, y_values[i], label=f"{decade_names[0]}s and {decade_names[i + 1]}s")

#Scatter plot for the whole temperature data
plt.plot(years_singled, abs_temp_annualy,"o", markersize = 3, label = "Annual mean data")

# Scatter plot for individual data points from the decadal data
plt.scatter(decades, temp_decade, c='red', marker='x', label='Decadal data points')

# Add labels, title, legend, and grid
plt.xlim(1950, 2025)
plt.xlabel("Time in years")
plt.ylabel("Temperature in °C")
plt.legend()
plt.grid(True)
plt.show()

#%% 
"""Task 4: Plot the 11-year running mean, the decadal averages, and your assessment of the linear fit that 
best represents your data in a single plot (5 points).
Think about how to best visualize this so that all the information is clear. 
The code should save the figure to a file called ‘Task_2.png’."""

#Plotting the 11-year running mean
plt.plot(years_runningmean, temp_abs_runningmean, "o-", label='Data from the running mean')

#Plotting the decadal averages
plt.plot(decades, temp_decade, "o-", label = "Data from the Decadal averages")

#Plotting the best fit
plt.plot(x_values, f(x_values, slopes[3], intercepts[3]), label = "Data from the best fit: 1960ies and 2000s")

# Add labels, title, legend, and grid
plt.xlim(1958, 2023)
plt.ylim(19.0,21.5)
plt.xlabel("Time in years")
plt.ylabel("Temperature in °C")
plt.legend()
plt.grid(True)

#saving and showing 
plt.savefig("Task 2.png", dpi = 300)
plt.show()

#%%
"""Calculate the PBIAS and normalized root mean square error (NRMSE) of your best decadal average
 linear fit compared to your annual temperature data (3 points). You can solve these manually or 
 use external libraries such as scipy, numpy, or others."""
 
# My best fit was the fourth one, which was Nr [3] in the list 
#calculating the points of the best fit that correspond to the years 1960 - 2023
temp_fit = []
for i in range(len(years_singled)):
    y = f(years_singled[i],slopes[3],intercepts[3])
    temp_fit.append(y)

#Calculate the predicted temperature for 2050
temp2050 = f(2050,slopes[3],intercepts[3])
print(f"The fit (1960s and 2000s) predicts an annual mean temperature from {round(temp2050,2)}°C for the year 2050.")

#calculate the difference between simulated and actual data
bias = []
bias_squared = []
for i in range(len(temp_fit)):
    biasvalue = temp_fit[i] - abs_temp_annualy[i]
    bias.append(biasvalue)
    bias_squared.append(biasvalue**2) #values for calculation of RMSE

#calculate the PBIAS
pbias =100 * sum(bias)/sum(abs_temp_annualy)

#calculating the RMSE: squaring the PBIAS and calculatin the root from that
RMSE = np.sqrt(sum(bias_squared)/len(abs_temp_annualy))

#normalise with a dataset dependent variable: using the range
NRMSE = RMSE / (max(abs_temp_annualy)-min(abs_temp_annualy))

print(f"The errors of the simulated to the real data are the following:\n PBIAS = {round(pbias,2)}%,\n RMSE = {round(RMSE,2)},\n NRMSE = {round(NRMSE,2)}")

#%% PART 3

"""Calculate the fraction of absorbed electromagnetic radiation leaving the earth this is trapped 
in the atmosphere (8 Points).
Use the temperature data of your country and different assumptions for the global albedo. 
For this we will assume a simple homogeneous atmospheric layer, 
and a balance of incoming and outgoing energy in both the surface of the Earth and the atmosphere. 
This approach will be explained during the beginning of the workshop.  
Create a numpy 2D-array, or a list of lists, or a dictionary of lists 
(a pandas DataFrame is also allowed although it is not officially covered in the course) 
that contains the calculated fraction of absorbed radiation as a function of albedo and
your yearly temperatures. For the albedo, 
assume values ranging from 0.2 to 0.4 (both inclusive) at 0.025 intervals."""

#1 Calculate f

def f_value(A,T):
    #function that calculates the f value (fraction of radiation reflected and emitted by the earth that stays in the atmosphere)
    #Input: A as albedo and T temperature in K
    #Output calculated fraction value f
    c = 1361 #solar constant in W/m^2
    sigma = 5.67 *10**(-8) #Boltzmann-Constant in W/(m^2 K)
    f_value = 2-((c*(1-A))/(2*sigma*T**4))
    return f_value

A_values = np.arange(0.2,0.425,0.025)
temperatures_kelvin = np.array(abs_temp_annualy) + 273.15

f_matrix = [] #columns are the different albedos and rows are the different temperatures
for i in range(len(temperatures_kelvin)):
    rows = []
    for j in range(len(A_values)):
        f = f_value(A_values[j],temperatures_kelvin[i])
        rows.append(f)
    f_matrix.append(rows)
    
#2 Replace all the f > 1 with f = 1 and record the A,T - pair in f_values_bigger1

f_values_bigger1 = []
for i in range(len(temperatures_kelvin)):
    for j in range(len(A_values)):
        if f_matrix[i][j]>1:
            f_matrix[i][j] = 1
            A_T_pair = [A_values[j],temperatures_kelvin[i]]
            f_values_bigger1.append(A_T_pair)
            
# Calculate radiative forcing for each f

def radiative_forcing(f):
    #function calculates the radiative forcing
    #Input: fraction value f
    #Output: radiative forcing in W/m^2 
    deltaT = temperatures_kelvin[-1] - temperatures_kelvin[0] #T_2022 - T_1960
    T_E = temperatures_kelvin[0] # T_E = T_1960
    sigma = 5.67 *10**(-8) #Boltzmann-Constant in W/(m^2 K)
    
    deltaF = deltaT * 4*(1-f/2)*sigma*T_E**3
    return deltaF 

rad_forcing_matrix = []
for i in range(len(f_matrix)):
    row = []
    for j in range(len(f_matrix[0])):
        delta = radiative_forcing(f_matrix[i][j])
        row.append(delta)
    rad_forcing_matrix.append(row)
    
#%% Additional for report
increase = []
for i in range(1,len(temp_abs_runningmean)):
    value = temp_abs_runningmean[i]-temp_abs_runningmean[i-1]
    increase.append(value)

meanincrease = sum(increase)/len(increase)

fraction_example = []
rad_forcing_example = []
for i in range(len(f_matrix)):
    datapoint1 = f_matrix[i][0]
    fraction_example.append(datapoint1)
    datapoint2 = rad_forcing_matrix[i][0]
    rad_forcing_example.append(datapoint2)

fig, ax1 = plt.subplots()

# Plot the first dataset on the left y-axis
ax1.plot(years_singled, fraction_example, marker="o", markersize = "2", color='b', label='Fraction (f)')
ax1.set_xlabel("Time in years")
ax1.set_ylabel("Fraction (f) in 1/K^3", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xlim(1958, 2023)

# Create a second y-axis on the right
ax2 = ax1.twinx()

# Plot the second dataset on the right y-axis
ax2.plot(years_singled, rad_forcing_example, marker="o", markersize = "2", color='orange', label='Radiative Forcing')
ax2.set_ylabel("Radiative Forcing in 1/K^3", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add legends and grid
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(True)
plt.title("Albedo A = 0.2")
# Display the combined plot
plt.savefig("Discussion.png", dpi = 300)
plt.show()
