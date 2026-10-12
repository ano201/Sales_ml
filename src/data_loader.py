import pandas as pd

def load_and_split_data(file_path):
	df = pd.read_csv(file_path)
	
	# sort data by date id date column exist
	if 'date' in df.columns:
		df['date'] = pd.to_datetime(df['date'])
		df = df.sort_values('date').reset_index(drop=True)
		
	#time based split
	train_data = df.iloc[:-180]
	test_data = df.iloc[-180:]
	
	# seperating features and target
	numeric_cols = df.select_dtypes(include=['number']).columns
	target_col = numeric_cols[-1]
	feature_cols = [col for col in numeric_cols if col != target_col]
	
	X_train = train_data[feature_cols]
	Y_train = train_data[target_col]
	
	X_test = test_data[feature_cols]
	Y_test = test_data[target_col]
	
	return X_train, Y_train, X_test, Y_test
