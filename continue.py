import pandas as pd
import numpy as np
from makeHistoricalData import makeHistoricalData
import multiprocessing as mp
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import itertools
import shelve
import time
import sys
import os
import zipfile


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')


from numpy.random import seed
seed(84156)
tf.random.set_seed(102487)

# mkdir for saving the results in
# Path("Results").mkdir(parents=True, exist_ok=True)

r = int(sys.argv[1])
H = 12
target = 'death'
test_size = 1
end_date_var = 7 - int(sys.argv[2])
point = int(sys.argv[2])
comparison_criteria = 'MAE'

default_model_type = 14

types = [
	[8, 16, 16], 
	[8, 16, 32], 
	[8, 32, 32], 
	[8, 32, 64], 
	[8, 64, 64], 
	[8, 16, 32, 16], 
	[8, 16, 32, 32], 
	[8, 32, 64, 32], 
	[8, 32, 128, 64],
	[8, 64, 128, 32], 
	[8, 16, 128, 128, 64], 
	[8, 16, 128, 256, 64], 
	[8, 32, 256, 256, 128], 
	[8, 64, 256, 128, 64], 
	[8, 256, 256, 128, 128], # default_model_type
	[8, 16, 32, 128, 64, 32], 
	[8, 16, 64, 256, 128, 64], 
	[8, 32, 128, 256, 128, 64], 
	[8, 64, 128, 256, 256, 128], 
	[8, 64, 256, 256, 128, 128]
]

dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]

lr_list = [
	0.000005, 0.000006, 0.000007, 0.000008, 0.000009, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 
	0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]


def mean_absolute_percentage_error(y_true, y_pred):
	return np.mean((abs(y_true - y_pred)/y_true)*100)


def normalize(X_train, y_train, X_val, y_val, X_test, y_test):
	scaler = preprocessing.StandardScaler()
	
	X_train = X_train.values
	X_val = X_val.values
	X_test = X_test.values
	y_train = y_train.values
	y_val = y_val.values
	y_test = y_test.values
	
	scaler = scaler.fit(X_train)
	
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)
	X_test = scaler.transform(X_test)
		
	scaler = scaler.fit(y_train.reshape(-1, 1))

	y_train = scaler.transform(y_train.reshape(-1, 1))
	y_val = scaler.transform(y_val.reshape(-1, 1))
	# y_test = scaler.transform(y_test.reshape(-1, 1))
	
	X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
	X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
	X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

	y_train = y_train.reshape((y_train.shape[0]), )
	y_val = y_val.reshape((y_val.shape[0]), )
	y_test = y_test.reshape((y_test.shape[0]), )

	return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c):
	train_X = X_train.iloc[:, 0:c].copy()
	train_y = y_train.copy()

	val_X = X_val.iloc[:, 0:c].copy()
	val_y = y_val.copy()

	test_X = X_test.iloc[:, 0:c].copy()
	test_y = y_test.copy()

	return train_X, val_X, test_X, train_y, val_y, test_y


################################################ my new activation function :))		( activation = sin(x)/cosh(x) )
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def custom1(x):
	return 2*K.sin(x) / (K.exp(x)+K.exp(-1*x))

get_custom_objects().update({'custom1': Activation(custom1)})
################################################


def get_model(n, model_type, dropout_value, lr_rate):
	model = Sequential()
	model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, n)))
	model.add(Activation(custom1))
	for i in types[model_type][1:len(types[model_type])-1]:
		model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
		model.add(Activation(custom1))
	model.add(LSTM(types[model_type][-1]))
	model.add(Activation(custom1))
	model.add(Dense(1))
	model.add(Activation(custom1))

	model.compile (
		loss=tf.keras.losses.MeanAbsoluteError(),
		optimizer=keras.optimizers.Adam(lr_rate),
		metrics=[tf.keras.metrics.MeanAbsoluteError()]
	)
	
	return model


def model_on_val(data, h, end_date, model_type, dropout_value, lr_rate):
	dataset = data.copy()

	results = []

	nrows = len(dataset.index)
	dataset.drop(dataset.index[nrows-r: nrows-1], 0, inplace=True)

	########################################### split data
	train = dataset.head(len(dataset.index)-2)
	val = dataset[len(dataset.index)-2:len(dataset.index)-1]
	test = dataset.tail(1)
	
	X_train = train.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_train = train['Target']

	X_val = val.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_val = val['Target']

	X_test = test.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_test = test['Target']
	########################################### end of split data

	limit = len(X_train.columns)

	for c in range(1, limit+1):	# step size is for going through covariates

		print("\n**********************************************************")
		print("STEP 1")
		print("h="+str(h)+", c="+str(c)+", end_date="+str(end_date)+", nrows="+str(nrows))
		print(X_train.shape, X_val.shape, X_test.shape)
		print(y_train.shape, y_val.shape, y_test.shape)
		print("**********************************************************\n")
		
		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = normalize(train_X, train_y, val_X, val_y, test_X, test_y)
		
		# es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

		model = get_model(train_X.shape[2], model_type, dropout_value, lr_rate)

		model.fit (
			train_X, train_y,
			epochs=1200,
			batch_size=4,
			validation_data=(val_X, val_y),
			verbose=0,
			# callbacks=[es], 
			shuffle=False
		)

		y_pred = model.predict(val_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
		
		y_pred = y_pred.reshape(-1)

		mae = mean_absolute_error(y_val.values.reshape(-1), y_pred)

		result_dict = {"h": h, "c": c, "model_type": model_type, "dropout": dropout_value, "lr": lr_rate, 
		"actual": float(y_val.values.reshape(-1)), "pred": float(y_pred), "MAE": mae}

		results.append(result_dict.copy())

		K.clear_session()

	return results


def model_on_val_fixed_param(data, h, c, end_date, model_type, dropout_value, lr_rate):
	dataset = data.copy()

	results = []

	nrows = len(dataset.index)
	dataset.drop(dataset.index[nrows-r: nrows-1], 0, inplace=True)

	########################################### split data
	train = dataset.head(len(dataset.index)-2)
	val = dataset[len(dataset.index)-2:len(dataset.index)-1]
	test = dataset.tail(1)
	
	X_train = train.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_train = train['Target']

	X_val = val.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_val = val['Target']

	X_test = test.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_test = test['Target']
	########################################### end of split data

	print("\n**********************************************************")
	print("STEP 2")
	print("h="+str(h)+", c="+str(c)+", end_date="+str(6-end_date)+", nrows="+str(nrows))
	print("model_type="+str(model_type)+", dropout="+str(dropout_value)+", lr_rate="+str(lr_rate))
	print(X_train.shape, X_val.shape, X_test.shape)
	print(y_train.shape, y_val.shape, y_test.shape)
	print("**********************************************************\n")
	
	train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y, scaler = normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	# es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

	model = get_model(train_X.shape[2], model_type, dropout_value, lr_rate)

	model.fit (
		train_X, train_y,
		epochs=1200,
		batch_size=4,
		validation_data=(val_X, val_y),
		verbose=0,
		# callbacks=[es], 
		shuffle=False
	)

	y_pred = model.predict(val_X)
	y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
	
	y_pred = y_pred.reshape(-1)

	mae = mean_absolute_error(y_val.values.reshape(-1), y_pred)

	result_dict = {"h": h, "c": c, "model_type": model_type, "dropout": dropout_value, "lr": lr_rate, 
	"actual": float(y_val.values.reshape(-1)), "pred": float(y_pred), "MAE": mae}

	results.append(result_dict.copy())

	K.clear_session()

	return results


def model_on_test(data, h, c, end_date, model_type, dropout_value, lr_rate):
	dataset = data.copy()

	nrows = len(dataset.index)
	dataset.drop(dataset.index[nrows-r: nrows-1], 0, inplace=True)

	########################################### split data
	train = dataset.head(len(dataset.index)-2)
	val = dataset[len(dataset.index)-2:len(dataset.index)-1]
	test = dataset.tail(1)
	
	X_train = train.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_train = train['Target']

	X_val = val.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_val = val['Target']

	X_test = test.drop(['date of day t', 'Target', 'county_fips'], axis=1)
	y_test = test['Target']
	########################################### end of split data
	
	train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y, scaler = normalize(train_X, train_y, val_X, val_y, test_X, test_y)

	# es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

	model = get_model(train_X.shape[2], model_type, dropout_value, lr_rate)

	model.fit (
		train_X, train_y,
		epochs=1200,
		batch_size=4,
		# validation_split=0.15, 
		validation_data=(val_X, val_y),
		verbose=0,
		# callbacks=[es], 
		shuffle=False
	)

	y_pred = model.predict(test_X)
	y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
	
	y_pred = y_pred.reshape(-1)

	K.clear_session()

	return y_pred


def get_best_historical_parameters(results_dataframe):	# each element is a tuple of (h, c, lr_rate, MAE)
	best_score, best_h, best_c = 0, 0, 0

	results_dataframe[comparison_criteria] = pd.to_numeric(results_dataframe[comparison_criteria])
	index_of_min_score = results_dataframe[comparison_criteria].idxmin()

	best_h = results_dataframe.iloc[index_of_min_score]['h']
	best_c = results_dataframe.iloc[index_of_min_score]['c']
	best_score = results_dataframe.iloc[index_of_min_score][comparison_criteria]

	return tuple((best_h, best_c, best_score))


def get_best_model_parameters(results_dataframe):	# each element is a tuple of (h, c, lr_rate, MAE)
	best_score, best_model_type, best_dropout, best_lr_rate = 0, 0, 0, 0

	results_dataframe[comparison_criteria] = pd.to_numeric(results_dataframe[comparison_criteria])
	index_of_min_score = results_dataframe[comparison_criteria].idxmin()

	best_model_type = results_dataframe.iloc[index_of_min_score]['model_type']
	best_dropout = results_dataframe.iloc[index_of_min_score]['dropout']
	best_lr_rate = results_dataframe.iloc[index_of_min_score]['lr']
	best_score = results_dataframe.iloc[index_of_min_score][comparison_criteria]

	return tuple((best_model_type, best_dropout, best_lr_rate, best_score))


def send_mail(mail_subject, mail_body, path, file_name):
	fromaddr = "lstm.covid19.server@gmail.com"
	toaddr = "m1998naderi@gmail.com"

	msg = MIMEMultipart()

	msg['From'] = fromaddr
	msg['To'] = toaddr

	msg['Subject'] = mail_subject

	body = mail_body
	msg.attach(MIMEText(body, 'plain'))

	filename = file_name
	# filepath = path+"/"+file_name
	filepath = file_name
	attachment = open(filepath, "rb")

	p = MIMEBase('application', 'octet-stream')
	p.set_payload((attachment).read())

	encoders.encode_base64(p)

	p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

	msg.attach(p)

	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.starttls()
	s.login(fromaddr, "4%f{h=W%m'f85cC7")

	text = msg.as_string()

	s.sendmail(fromaddr, toaddr, text)
	s.quit()


def zipdir(path, ziph):
	# ziph is zipfile handle
	for root, dirs, files in os.walk(path):
		for file in files:
			ziph.write(os.path.join(root, file))


def main():
	print("r=%s, point=%s" % (r, point))
	time.sleep(1)

	tmp_df = pd.read_csv('Results/default_model_results_'+str(point)+'.csv')
	best_h, best_c, _ = get_best_historical_parameters(tmp_df)
	print("best_h = " + str(best_h) + ", best_c = " + str(best_c) + "\n\n")
	time.sleep(2)

	start_time = time.time()

	data_list = []

	print("Started to make historical data")
	
	# making historical data with all 'h' values and storing the dataframes in a list
	with mp.Pool(mp.cpu_count()) as pool:
		data_list = pool.starmap(makeHistoricalData, [(h, r, test_size, target, 'mrmr', 'country', 'weeklyaverage', '', [], 'country', end_date_var) 
		for h in range(1, H+1)])

	print("Making historical data is done.")

	# ########################################################### choosing best 'h' and best 'c' for the point
	# with mp.Pool(mp.cpu_count()) as pool:
	# 	default_model_results = pool.starmap(model_on_val, [(data_list[h], h+1, end_date_var, 14, 0.2, 0.000009) for h in range(0, H)])

	# tmp = list(itertools.chain.from_iterable(default_model_results))
	# default_model_results_df = pd.DataFrame(tmp)

	# default_model_results_df.to_csv('Results/default_model_results_'+str(point)+'.csv', index=False)

	# tmp_df = pd.read_csv('Results/default_model_results_'+str(point)+'.csv')

	# best_h, best_c, _ = get_best_historical_parameters(tmp_df)		

	# print("best_h = " + str(best_h) + "best_c = " + str(best_c) + "\n\n")

	# with open('Results/best_h_and_best_c.txt', 'w') as f:
	# 	f.write("point = %s\n" % point)
	# 	f.write("end_date = %s\n" % end_date_var)
	# 	f.write("best_h = %s\n" % int(best_h))
	# 	f.write("best_c = %s\n" % int(best_c))
	# ###########################################################

	# # save the entire session until now
	# shelve_filename = 'Results/step1.out'
	# s = shelve.open(shelve_filename, 'n')  # 'n' for new
	# for key in dir():
	# 	try:
	# 		s[key] = locals()[key]
	# 	except:
	# 		print('ERROR shelving: {0}'.format(key))
	# s.close()

	# ###########################################################
	# zipf_name = "Results_Step1_r%s_p%s.zip" % (r, point)
	# zipf = zipfile.ZipFile(zipf_name, 'w', zipfile.ZIP_DEFLATED)
	# zipdir('Results/', zipf)
	# zipf.close()

	# # sending mail
	# mail_subject = "lstm model - STEP 1 - r%s - p%s"  % (r, point)
	# mail_body = "Finished STEP 1"
	# send_mail(mail_subject, mail_body, "", zipf_name)
	###########################################################

	########################################################### choosing best model type and dropout value and learning rate for the point
	best_model_type = -1
	best_dropout_value = -1.0
	best_lr_rate = -1.0

	h_tmp = int(best_h)
	c_tmp = int(best_c)
	data_tmp = data_list[h_tmp-1]

	with mp.Pool(mp.cpu_count()) as pool:
		model_results = pool.starmap(model_on_val_fixed_param, [(data_tmp, h_tmp, c_tmp, end_date_var, model_type, dropout, lr_rate) 
		for model_type in range(len(types)) for dropout in dropout_list for lr_rate in lr_list])
	
	tmp = list(itertools.chain.from_iterable(model_results))
	model_results_df = pd.DataFrame(tmp)

	model_results_df.to_csv('Results/other_models_results_'+str(point)+'.csv', index=False)

	tmp_df = pd.read_csv('Results/other_models_results_'+str(point)+'.csv')

	best_model_type, best_dropout_value, best_lr_rate, _ = get_best_model_parameters(tmp_df)
	best_model_type = int(best_model_type)

	with open('Results/best_parameters.txt', 'w') as f:
		f.write("point = %s\n" % point)
		f.write("end_date = %s\n" % end_date_var)
		f.write("best_model_type = %s\n" % best_model_type)
		f.write("dropout = %s\n" % best_dropout_value)
		f.write("lr_rate = %s\n" % best_lr_rate)
	###########################################################

	# save the entire session until now
	shelve_filename = 'Results/step2.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	###########################################################
	zipf_name = "Results_Step2_r%s_p%s.zip" % (r, point)
	zipf = zipfile.ZipFile(zipf_name, 'w', zipfile.ZIP_DEFLATED)
	zipdir('Results/', zipf)
	zipf.close()

	# # sending mail
	# mail_subject = "lstm model - STEP 2 - r%s - p%s"  % (r, point)
	# mail_body = "Finished STEP 2"
	# send_mail(mail_subject, mail_body, "", zipf_name)
	###########################################################

	########################################################### making prediction on the test point with the best parameters which are obtained
	h_tmp = int(best_h)
	c_tmp = int(best_c)
	model_type_tmp = best_model_type
	dropout_tmp = best_dropout_value
	lr_rate_tmp = best_lr_rate

	final_pred = model_on_test(data_list[h_tmp-1], h_tmp, c_tmp, end_date_var, model_type_tmp, dropout_tmp, lr_rate_tmp)
	###########################################################
	
	########################################################### saving final results
	tmp_dataset = data_list[0]
	y_real = float(tmp_dataset.tail(1)['Target'].values.reshape(-1))

	with open('Results/final_result.txt', 'w') as f:
		f.write("point = %s\n" % point)
		f.write("end_date = %s\n" % end_date_var)
		f.write("actual = %s\n" % y_real)
		f.write("pred = %s\n" % final_pred)

	df = pd.DataFrame()
	df['point'] = list([point])
	df['actual'] = list([y_real])
	df['pred'] = list([final_pred])
	df.to_csv('Results/final_result.csv', index = False)
	###########################################################

	# save the entire session until now
	shelve_filename = 'Results/step3.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	###########################################################
	zipf_name = "Results_Step3_r%s_p%s.zip" % (r, point)
	zipf = zipfile.ZipFile(zipf_name, 'w', zipfile.ZIP_DEFLATED)
	zipdir('Results/', zipf)
	zipf.close()

	end_time = time.time()

	# # sending mail
	# mail_subject = "lstm model - STEP 3 - r%s - p%s"  % (r, point)
	# mail_body = "Finished STEP 3\nThe code was ran in --- %s --- seconds" % (end_time - start_time)
	# send_mail(mail_subject, mail_body, "", zipf_name)
	###########################################################

	print("\nThe code was ran in --- %s --- seconds" % (end_time - start_time))


if __name__ == "__main__":
	main()
