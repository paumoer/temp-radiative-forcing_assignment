# Temperature modeling from a class on Earth System Science
This code provides a solution for an environmental sciences master course assignment, modeling temperature trends and the resulting radiative forcing.

The assignment involved analyzing Berkeley Earth temperature anomaly data for an assigned country. Steps included data processing, calculating trends (running mean, decadal averages, linear fits), and modeling radiative forcing based on albedo variations. Results were interpreted in a report, assessing temperature projections, absorbed radiation, and climate implications for the country.
## Files:
- Python file
- Temperature data
- Resulting graphs

Tasks:
## Part 1: Data Collection and Preparation 
Read dates and monthly temperature anomalies into Python.
Subset data to start from January 1960.
Perform a sanity check for missing values and print the result.
Convert monthly temperature anomalies to annual temperature anomalies.
Convert annual anomalies to absolute temperature values using the reference period.
Plot and save the time series of absolute temperature (Task_1.png).

## Part 2: Data Calculations and Plotting 
Compute an 11-year running mean of temperature data.
Calculate temperature averages for each decade.
Compute linear fits for decadal temperature trends (e.g., 1960s–1970s, 1960s–1980s, etc.).
Assess which linear fit best represents the temperature trend and visualize it.
Plot the 11-year running mean, decadal averages, and best-fit trend (Task_2.png).
Calculate PBIAS and NRMSE for the best-fit decadal trend.

## Part 3: Intermediate Operations 
Compute the fraction of absorbed radiation for different albedo values (0.2–0.4).
Correct any fractions exceeding 1 and record affected albedo-temperature pairs.
Print corrected values or confirm no invalid values exist.
Compute radiative forcing for each albedo value from 1960 to the last full year.
