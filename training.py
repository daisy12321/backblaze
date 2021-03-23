import os
import numpy as np
import pandas as pd
os.environ["IAI_JULIA"] = "/usr/local/bin/julia"
from interpretableai import iai

def get_X(df, drop_age, drop_eng):
	# Only keep SMART features in X
	cols_to_drop = [col for col in df.columns if 'smart' not in col]
	X = df.drop(cols_to_drop, axis=1)

	# Remove SMART features correlated with age
	if drop_age:
		inds = [4, 9, 12, 192, 193, 240, 241, 242]
		cols_age = [col for col in X.columns for i in inds if ('_' + str(i) + '_') in col]
		X = X.drop(cols_age, axis=1)

	# Remove engineered features
	if drop_eng:
		cols_engineered = [col for col in X.columns if ('raw_' in col) or ('normalized_' in col)]
		X = X.drop(cols_engineered, axis=1)

	# Return clean DataFrame
	return(X)

def train_OST(df, file_name, drop_age=True, drop_eng=True):
	# Extract data fields to fit Optimal Survival Tree
	X = get_X(df, drop_age=drop_age, drop_eng=drop_eng)
	died = df['failed']
	times = df['remaining_useful_life']

	# Fit Optimal Survival Tree
	grid = iai.GridSearch(iai.OptimalTreeSurvivor(random_seed=1,
	                                              missingdatamode='separate_class',
	                                              criterion='localfulllikelihood',
	                                              minbucket=int(0.01 * len(X))),
	                      max_depth=range(2, 5))
	grid.fit(X, died.tolist(), times, validation_criterion='harrell_c_statistic')

	# Write output models
	grid.get_learner().write_html('models/' + file_name + '.html')
	grid.write_json('models/' + file_name + '.json')

def train_OCT(df, file_name, drop_age=False, drop_eng=True):
	# Separate machines which failed and never failed
	machines_failures = list(df[df['failure'] == 1]['serial_number'].unique())
	machines_no_failures = list(df['serial_number'].drop_duplicates() \
                                	.replace(machines_failures, np.NaN).dropna())

	# Split machines between training and testing
	np.random.seed(1)
	machines_train = list(np.random.choice(machines_failures, int(len(machines_failures) * 0.8),
	                                       replace=False)) + \
	                 list(np.random.choice(machines_no_failures, int(len(machines_no_failures) * 0.8),
	                                       replace=False))
	machines_test = list(set(machines_failures + machines_no_failures).difference(set(machines_train)))

	# Save training and testing machines
	pd.Series(machines_train).to_csv('machines_train_dataset_3_2020_08_12.csv', index=False)
	pd.Series(machines_test).to_csv('machines_test_dataset_3_2020_08_12.csv', index=False)

	# Set pair (machine, date) as multi-index
	df = df.sort_values(['serial_number', 'datetime'])
	df = df.set_index(['serial_number', 'datetime'])

	# Split data between training and testing
	df_train = df.loc[machines_train]
	df_test = df.loc[machines_test]

	# Extract data fields to fit Optimal Classification Tree
	X_train = get_X(df_train, drop_age=drop_age, drop_eng=drop_eng)
	X_test = get_X(df_test, drop_age=drop_age, drop_eng=drop_eng)
	y_train = df_train['failure_next_30'].apply(lambda x: 'Failure' if x else 'Operational')
	y_test = df_test['failure_next_30'].apply(lambda x: 'Failure' if x else 'Operational')

	print(X_train.shape)
	print(X_test.shape)
	# Fit Optimal Classification Tree
	grid = iai.GridSearch(iai.OptimalTreeClassifier(random_seed=1,
	                                                missingdatamode='separate_class',
	                                                # criterion='misclassification',
	                                                criterion='gini',
	                                                minbucket=int(0.005 * len(X_train))),
	                          max_depth=range(2, 5))
	grid.fit_cv(X_train, y_train)

	# Write output models
	grid.get_learner().write_html('models/' + file_name + '.html')
	grid.write_json('models/' + file_name + '.json')



# ----- Model #1 -----
# OST on Dataset #1 - Survival dataset Q1 2017 - Q1 2020 (3 years and 1 quarter)

# Import Dataset #1
# df_failures = pd.read_csv('data/iteration_2/df_failures_2020_06_26.csv')
df_failures = pd.read_csv('dataset_1.csv')
df_failures['datetime'] = pd.to_datetime(df_failures['datetime'])

# Resample data
df_failures_rs = df_failures.sample(50000, random_state=1)

# Train OST
train_OST(df_failures_rs, '1_OST_17_20', drop_age=True, drop_eng=True)

# Clear variables
del df_failures
del df_failures_rs



# ----- Model #2 -----
# OST on Dataset #2 - Survival dataset Q2 2019 - Q1 2020 (1 year)

# Import Dataset #2
# df = pd.read_csv('data/iteration_2/df_OST_19_20_no_eng_2020_08_04.csv')
df = pd.read_csv('dataset_2.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Resample data
df_rs = df.sample(50000, random_state=1)

# Train OST
train_OST(df_rs, '2_OST_19_20', drop_age=True, drop_eng=True)

# Clear variables
del df
del df_rs



# ----- Model #3 -----
# OCT on Dataset #3 - Classification dataset Q2 2019 - Q1 2020 (1 year)

# Import Dataset #3
df = pd.read_csv('dataset_3.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# # Resample data
# df_rs = df[df['failure_next_30']] \
# 			.append(df[df['failure_next_30'] == False] \
#             	.sample(100000 - len(df[df['failure_next_30']])))

# Train OCT
train_OCT(df, '3_OCT_19_20', drop_age=True, drop_eng=True)

# Clear variables
del df
del df_rs



# ----- Model #4 -----
# OST on Dataset #4 - Survival dataset Q1 2020 (1 quarter)

# Import Dataset #4
df = pd.read_csv('dataset_4.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Train OST
train_OST(df, '4_OST_Q1_20_no_age', drop_age=True, drop_eng=True)
train_OST(df, '4_OST_Q1_20_age', drop_age=False, drop_eng=True)


