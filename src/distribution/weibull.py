import numpy as np
from scipy.stats import weibull_min

def weibull_convertion(data:np.ndarray)\
	-> (np.ndarray, np.ndarray, np.ndarray):
	# step 1: sort
	data_sorted = np.sort(data)
	n = len(data_sorted)
	
	# step 2: plotting position
	i = np.arange(1, n+1)
	F = (i - 0.3) / (n + 0.4)
	
	# step 3: weibits
	weibits = np.log(-np.log(1 - F))
	
	# x-axis for Weibull plot
	x = np.log(data_sorted)

	return data_sorted, x, weibits

def fit_weibull(data:np.ndarray) -> (float, float):
	shape, _, scale = weibull_min.fit(data, floc=0)
	beta = shape
	eta = scale
	return beta, eta

def weibull_plot(ax, data) -> None:
	data_sorted, _, weibits = weibull_convertion(data)
	beta, eta = fit_weibull(data)
	data_linspace = np.linspace(start=data.min(), stop=data.max(), num=1000)
	fitted_weibits = beta * (np.log(data_linspace) - np.log(eta))
	ax.scatter(data_sorted, weibits, label='Simulation')
	ax.plot(data_linspace, fitted_weibits, label='Fit')
	ax.set_xscale('log')
	ax.set_ylabel("Weibit")
	ax.grid()
	ax.legend()