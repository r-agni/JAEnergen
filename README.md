# JAEnergen
The program calculates the required storage capacity when averaging the load over different time periods. It does this by starting with an initial storage capacity of 0.1 GWh. It is assumed that the supply is constant at the average over a specific time period (e.g. When averaging over 24 hours, it is assumed that the supply for each day is at the average demand of that day). At each 15-minute interval, the excess (difference between the demand and average) is added to the storage. If the supply is more than the demand, the excess is positive, and if the supply is less than the demand, the excess is negative. When the storage has reached the maximum capacity, the supply is ramped down to exactly match the demand. If, at any point, the amount of energy in storage is less than the deficit, it means that the storage capacity is not large enough, so it is increased by a factor of 1.1. This process is repeated until it reaches a storage capacity which is large enough to supply the required energy whenever there is a deficit. Since the storage is increased by a factor of 1.1, the final storage capacity is within 10% of the exact value.

# Functions:
country_demand - Takes a country code and returns a numpy array of the demand every 15 minutes for that country. It also prints the average, minimum, and maximum demand, as well as the time of the first and last 15-minute interval and the number of 15-minute intervals.

country_supply_clean - Takes a country code and returns a numpy array of the energy supplied from solar and wind every 15 minutes for that country. It also prints the average, minimum, and maximum supply from these sources, as well as the time of the first and last 15-minute interval and the number of 15-minute intervals.

average_demand_over_interval - Takes a numpy array of the demand over successive 15-minute intervals, and the number of 15 minute intervals over which the demand is to be averaged. It returns a numpy array of the average over blocks of the specified number of 15-minute intervals.

required_storage - Takes a numpy array of the demand and a numpy array of the supply over successive 15-minute intervals. It calculates and returns the required storage (within 10% of the exact amount required) using the method described earlier.

plot_country_demand - Takes a country code, the number of hours over which to average (can take multiple values and plot them on the same graph), the day from which to start plotting, and the number of days for which the average should be plotted. It then plots a graph with the specified averages and the load without averaging. It also plots the energy supplied by the clean sources (wind and solar) on a separate axis.

plot_country_clean_supply - Takes the same parameters as the previous function and plots the clean energy supplied in the country, averaged over various time intervals.

# Main Method: 
The main function brings all the functions together into a simple function to plot graphs and calculate the storage capacity. It also displays some statistics of the data to help us analyze it.
1. It creates a numpy array for the demand using the country_demand function and displays the peak supply capacity without averaging by finding the maximum value in the array. 

 2. It finds the average demand over each time interval and prints the required peak capacity and storage for each time period. 

3.  It also finds the storage required when there is a base supply of 30 GW from fossil fuels and the rest is supplied by solar and wind, and the storage required when the energy from clean sources is ten times the current amount. 

4. It calls the functions to plot graphs for the average demand over various time intervals, as well as graphs for the average supply from solar and wind sources over various time intervals. 
