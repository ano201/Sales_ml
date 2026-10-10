import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(file_path):
	# Load dataset and parse date
	df = pd.read_csv(file_path)
	df['date'] = pd.to_datetime(df['date'])

	print('Basic Statistical summery')
	print("Dataset Shape:", df.shape)
	print('\n', df.describe())

	print('Skewness and Kurtosis')
	print(f"Sales skewness: {df['sales'].skew():.4f}")
	print(f"Sales kurtosis: {df['sales'].kurtosis():.4f}")
	
	print("Outlier detection usin IQR")
	q1 = df['sales'].quantile(0.25)
	q3 = df['sales'].quantile(0.75)
	iqr = q3-q1
	l_bound = q1 - 1.5 * iqr
	u_bound = q3 + 1.5 * iqr
	outlier = df[(df['sales'] < l_bound) | (df['sales'] > u_bound)]
	
	print('Total outliers found', len(outlier))
	print(f"Outlier liits -> Lower: {l_bound}, Upper: {u_bound}")

	print('Generating and saving Visual Plots')
	
	# Sales Trend / Scatter_like line plot
	plt.figure(figsize=(12, 6))
	daily_sales = df.groupby('date')['sales'].sum().reset_index()
	plt.plot(daily_sales['date'], daily_sales['sales'], alpha=0.6, color='b', label='Daily Total Sales')
	plt.title('daily Sales Trend Over Time')
	plt.xlabel('Date')
	plt.ylabel('Sales')
	plt.savefig('data/processed/sales_trend_scatter.png')
	plt.close()

	# outlier Boxplot
	plt.figure(figsize=(8, 4))
	sns.boxplot(x=df['sales'], color='orange')
	plt.title('Sales outliers boxplot')
	plt.savefig('data/processed/sales_boxplot.png')
	plt.close()

	# Monthly sales Pattern
	df['month'] = df['date'].dt.month
	monthly_avg = df.groupby('month')['sales'].mean()
	plt.figure(figsize=(8, 4))
	monthly_avg.plot(kind='bar', color='green')
	plt.title('Avarage sales by month (Seasonality check)')
	plt.xlabel('Month')
	plt.ylabel('Average Sales')
	plt.savefig('data/processed/monthly_sessionality.png')
	plt.close()

	print("All plots saved successfully")


if __name__ == '__main__':
	run_eda('data/raw/train.csv')
