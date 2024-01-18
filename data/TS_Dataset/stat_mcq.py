import pdb
import sys
import numpy as np
import pickle
import random
import json
from scipy.stats import skew, kurtosis

# 1. Mean
def compute_mean(time_series):
	return np.mean(time_series)

# 2. Median
def compute_median(time_series):
	return np.median(time_series)

# 3. Variance
def compute_variance(time_series):
	return np.var(time_series)

# 4. Standard Deviation
def compute_std_deviation(time_series):
	return np.std(time_series)

# 5. Skewness
def compute_skewness(time_series):
	return skew(time_series)

# 6. Kurtosis
def compute_kurtosis(time_series):
	return kurtosis(time_series)

def get_stats(time_series):
	mean_value = compute_mean(time_series)
	median_value = compute_median(time_series)
	variance_value = compute_variance(time_series)
	std_deviation_value = compute_std_deviation(time_series)
	# skewness_value = compute_skewness(time_series)
	# kurtosis_value = compute_kurtosis(time_series)

	results_arr = [mean_value, median_value, variance_value, std_deviation_value]
	return results_arr

def get_options(original_float, num_numbers=3, variable=1.5, fixed=10):
	tolerance = max(original_float*variable, original_float+fixed)
	random_numbers = [round(original_float,2)]
	
	for _ in range(num_numbers):
		random_offset = random.uniform(-tolerance, tolerance)
		new_number = round(original_float + random_offset, 2)
		while new_number in random_numbers:
			random_offset = random.uniform(-tolerance, tolerance)
			new_number = round(original_float + random_offset,2)
		random_numbers.append(new_number)
	
	return random_numbers


def get_questions():
	mean = ["What is the mean of all values in the time-series?", "What is the average of all the values in the time-series?", "Could you calculate the time-series' mean by averaging all the values?", "What's the arithmetic mean of all the values in the time-series?", "Please find the mean value of the entire time-series by summing and dividing by the count.", "Calculate the time-series' average by adding up all the values and dividing by the total count."]

	median = ["What is the median of the time-series?", "What is the median value of the time-series data?", "Calculate the median of the time-series by finding the middle value.", "Could you determine the middle value, or the median, of the time-series?", "Please find the median by sorting the time-series and identifying the middle value.", "Find the time-series' median by arranging the values in ascending order and selecting the middle value."]

	variance = ["What is the variance of the time-series?", "What is the variance of the time-series, which measures the spread of data points?", "Calculate the variance of the time-series to assess how data points deviate from the mean.", "Could you determine the time-series' variance, representing the data's dispersion?", "Please find the variance by computing the average of the squared differences from the mean.", "Find the time-series' variance to understand the variability of data points around the mean."]

	std = ["What is the standard deviation of the time-series?", "What is the standard deviation, a measure of data dispersion, for the time-series?", "Calculate the standard deviation to assess how much data points deviate from the mean in the time-series.", "Could you determine the standard deviation of the time-series, which indicates the spread of data?", "Please find the standard deviation by taking the square root of the variance.", "Find the time-series' standard deviation to understand the average distance between data points and the mean."]

	# Not including Kurtosis and Skew!!!!
	skew = ["What is the skewness of the time-series?", "Calculate the skewness of the time-series to understand the asymmetry of its data distribution.", "What is the skewness, which measures the degree of asymmetry in the time-series data?", "Could you determine the time-series' skewness, indicating whether the data is skewed to the left or right?", "Please find the skewness by assessing the third standardized moment of the data.", "Find the skewness of the time-series to assess whether the data is positively or negatively skewed."]

	kurtosis = ["What is the kurtosis of the time-series?", "Calculate the kurtosis of the time-series to understand the shape of its data distribution.", "What is the kurtosis, which measures the degree of tailedness or peakedness in the time-series data?", "Could you determine the time-series' kurtosis, indicating whether the data has heavy or light tails?", "Please find the kurtosis by assessing the fourth standardized moment of the data.", "Find the kurtosis of the time-series to assess whether it has fat tails (leptokurtic) or thin tails (platykurtic)."]

	ques_arr = [mean, median, variance, std]
	return ques_arr

def prep_mcq(time_series):
	mcq_list = []
	stats = get_stats(time_series)
	questions = get_questions()

	for i in range(len(stats)):
		for ques in questions[i]:
			options = get_options(stats[i])
			cor = options[0]
			np.random.shuffle(options)
			cor_ind = np.where(options == cor)[0][0]
			opt_str = [str(k) for k in options]
			mcq_list.append([ques, opt_str, cor_ind])
	return mcq_list

def mcq_mike(data, filename):
	list_qs = []
	count = 0
	new_data_list = []
	for i in range(len(data)):
		tt_data = data[i]['series']
		mcq_list = prep_mcq(tt_data)
		for mcq in mcq_list:
			data[i]['question'] = mcq[0]
			data[i]['options'] = mcq[1]
			data[i]['answer_index'] = int(mcq[2])
			data[i]['qid'] = count
			new_data_list.append(data[i].copy())
			count += 1

	with open(filename, 'w') as f:
		for d in new_data_list:
			json.dump(d, f)
			f.write('\n')

def data_loader_mike(data_train, data_test = ''):
	train = []
	test = []
	with open(data_train) as f:
		for line in f:
			train.append(json.loads(line))

	with open(data_test) as f:
		for line in f:
			test.append(json.loads(line))
	return train, test

'''
This will load the dataset of time-series examples and create statistical MCQs.
Returns:
	1. A train.json file for training data MCQs.
	2. A test.json file for testing data MCQs.
'''
def main():
	# Data Loading
	data_train = sys.argv[1]
	data_test = sys.argv[2]
	train, test = data_loader_mike(data_loc, data_test)
	mcq_mike(train, 'train.json')
	mcq_mike(test, 'val.json')

if __name__=="__main__": 
	main() 
