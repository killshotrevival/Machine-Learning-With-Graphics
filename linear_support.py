import support as sp 

import pandas  
import numpy as np  
#import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

import threading


random_color = sp.rand_color()

font_color = (0,191,255)
text_font = 'freesansbold.ttf'

window_data = {'window_prop':{'height': 700,
								'width' : 1380,
								'name'  : 'Linear Regression'
								},
				'background_color':(250,250,250),
				'data_set': {
							'X':pandas.DataFrame(),
							'y': pandas.DataFrame()	
								},
				'data_set_size_rec_prop' : {
											'msg'  :'m - ',
											'button_x': 50,
											'button_y': 30,
											'button_width' : 180,
											'button_height' : 50,
											'inactive_color' : (240,240,240),
											'textfont' : text_font,
											'text_size' : 30,
											'inactive_text_color' : font_color
												},
				'features_rec_prop' : {
											'msg'  :'n - ',
											'button_x': 260,
											'button_y': 30,
											'button_width' : 180,
											'button_height' : 50,
											'inactive_color' : (240,240,240),
											'textfont' : text_font,
											'text_size' : 30,
											'inactive_text_color' : font_color
												},
				'algo_rec_prop' : {
											'msg'  :'',
											'button_x': 470,
											'button_y': 30,
											'button_width' : 180,
											'button_height' : 50,
											'inactive_color' : (240,240,240),
											'textfont' : text_font,
											'text_size' : 30,
											'inactive_text_color' : font_color,
											'pressed' : False
												},
				'learning_b_prop' : {
											'msg'  :'Learn',
											'button_x': 680,
											'button_y': 5,
											'button_width' : 100,
											'button_height' : 100,
											'inactive_color' : (220,20,60),
											'textfont' : text_font,
											'text_size' : 30,
											'inactive_text_color' : (240,240,240),
											'pressed' : False
												},
		

				'regularize_para_p_prop' : {
											'outer_rect_prop':{
														'x':810,
														'y': 25,
														'width':220,
														'height':50,
														'foreground_color':(240,240,240),
														'background_color': font_color

															},
											'lambda_text':'Lambda',
											'lambda_text_prop':(850,53,30,font_color),       #(width, height, size, color))
											'inner_rect_prop': {

																'msg'  :'',
																'button_x': 895,
																'button_y': 29,
																'button_width' : 125,
																'button_height' : 40,
																'inactive_color' : (220,220,220),
																'textfont' : text_font,
																'text_size' : 30,
																'inactive_text_color' : (0,0,0),
																'pressed' : False
															}
											},
						'graph_needed':{	'outer_rect_prop':{
														'x':1060,
														'y': 25,
														'width':200,
														'height':50,
														'foreground_color':(240,240,240),
														'background_color': font_color

															},
											'text':{
													'text_msg':'Graph',
													'text_prop':(1190,50,30,font_color)
											},
											'check_box':{
														'x':1080,
														'y':32,
														'width':30,
														'height':30,
														'inactive_color':(220,220,220),
														'active_color': font_color,
														'state':'inactive'
											}
										},
						'progress_bar': {
										'text':{
												'text_msg': 'Progress',
												'text_prop': (340, 130, 30, font_color),	
													},
										'progress_bar_prop':{
															'x':90,
															'y':150,
															'width':500,
															'height':30,
															'foreground_color':(0,255,0),
															'background_color': (245,245,245)
																},
										'percent':{
													'percent_msg':'',
													'percent_prop': (640,165,30, font_color)
														}
										},
						'total_cost':{
									'outer_rect':{
												'x':80,
												'y':210,
												'width':520,
												'height':40,
												'foreground_color':(240,230,230)	
													},
									'inner':{
											'inner_msg':'Total Cost =  ',
											'inner_prop': (350,230,30, font_color)
													}	
										},
						'time': {
								'outer_rect':{
											'x':80,
											'y':280,
											'width':520,
											'height':40,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Time Elapsed :',
									'inner_prop': (350,300,30, font_color)
										}
								},
						'mean_abs': {

								'outer_rect':{
											'x':80,
											'y':350,
											'width':520,
											'height':40,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Mean Abs Error :',
									'inner_prop': (330,370,30, font_color)
										
										}	
										},

						'mean_sqr': {

								'outer_rect':{
											'x':80,
											'y':420,
											'width':520,
											'height':40,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Mean Sqr Error :',
									'inner_prop': (330,440,30, font_color)
										
										}	
										},

						'root_mean_sqr': {

								'outer_rect':{
											'x':80,
											'y':490,
											'width':520,
											'height':40,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Root Mean Sqr Error :',
									'inner_prop': (330,510,30, font_color)
										
										}	
										},

						'predicted_input_path': {	'text':{
															'text_msg':'Predict',
															'text_prop':(1080, 130, 30, font_color)
															},
													'outer_rect':{
																	'x':830,
																	'y':157,
																	'width':500,
																	'height':50,
																	'foreground_color':(245,245,245),
																	'background_color': (100,100,100)
																},
													'inner_rect_prop':{
																'msg'  :'Predict input file path here',
																'button_x': 840,
																'button_y': 160,
																'button_width' : 480,
																'button_height' : 35,
																'inactive_color' : (255,255,255),
																'textfont' : text_font,
																'text_size' : 21,
																'inactive_text_color' : font_color,
																'pressed' : False
																	}
												},
						'outputfile_name': {
													'outer_rect':{
																	'x':830,
																	'y':237,
																	'width':500,
																	'height':50,
																	'foreground_color':(245,245,245),
																	'background_color': (100,100,100)
																},

													'inner_rect_prop':{
																'msg'  :'outputfile name here',
																'button_x': 840,
																'button_y': 240,
																'button_width' : 480,
																'button_height' : 35,
																'inactive_color' : (255,255,255),
																'textfont' : text_font,
																'text_size' : 21,
																'inactive_text_color' : font_color,
																'pressed' : False
																	}
											},

						'predict_b_prop':{
											'msg'  :'Predict',
											'button_x': 1000,
											'button_y': 310,
											'button_width' : 160,
											'button_height' : 50,
											'inactive_color' : (220,20,60),
											'textfont' : text_font,
											'text_size' : 30,
											'inactive_text_color' : (240,240,240),
											'pressed' : False	
										},
						'status_bar':{
										'outer_rect':{
											'x':0,
											'y':660,
											'width':1540,
											'height':40,
											'foreground_color':(150,150,150)
											},
										'inner':{
											'inner_msg': 'Status:',
											'inner_prop': (60,680,30, (20, 20, 20)),
											'inner_msg1': '',
											'inner_prop1': (720,680,25, (0, 0, 0))	
											}
									},
					'forward_button':{
									'msg'  :'F',
									'button_x': 1355,
									'button_y': 300,
									'button_width' : 25,
									'button_height' : 100,
									'inactive_color' : (10,10,10),
									'textfont' : text_font,
									'text_size' : 20,
									'inactive_text_color' : (200,200,200),
									'pressed' : False
										},
					'Computation_status': '',
					'compute_prop':'', 
					'fun_completed': False
}


class thread(threading.Thread):
	def __init__(self, name, fun):
		threading.Thread.__init__(self)
		self.name = name
		self.fun = fun
	def run(self):
		self.fun()



def compute():
	window_data['status_bar']['inner']['inner_msg1'] = ''
	window_data['Computation_status'] = ''
	try:
		#print('alpha', window_data['regularize_para_p_prop']['inner_rect_prop']['msg'])
		alpha = float(window_data['regularize_para_p_prop']['inner_rect_prop']['msg'])
		if window_data['algo_rec_prop']['msg'] == 'Ordinary':
			regressor = LinearRegression()
		elif window_data['algo_rec_prop']['msg'] == 'Ridge':
			alpha = float(window_data['regularize_para_p_prop']['inner_rect_prop']['msg'])
			regressor = Ridge(alpha =alpha)
		elif window_data['algo_rec_prop']['msg'] == 'Lasso':
			alpha = float(window_data['regularize_para_p_prop']['inner_rect_prop']['msg'])
			regressor = Lasso(alpha =alpha)

		X_train, X_test, y_train, y_test = train_test_split(window_data['data_set']['X'], window_data['data_set']['y'], test_size=0.2, random_state=0)

		#regressor = LinearRegression()  
		prop = regressor.fit(X_train, y_train)

		# coeff_df = pandas.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
		#print('Coefficients', regressor.coef_)

		# y_pred = regressor.predict(X_test)
		y_pred = prop.predict(X_test)
		diff = (y_pred-y_test)
		window_data['total_cost']['inner']['inner_msg'] = 'Total Cost =  '+str(round(diff.sum(),2))
		window_data['mean_abs']['inner']['inner_msg'] = 'Mean Absolute Error =  '+str(round(metrics.mean_absolute_error(y_test, y_pred),2))
		window_data['mean_sqr']['inner']['inner_msg'] = 'Mean Squared Error =  '+str(round(metrics.mean_squared_error(y_test, y_pred),2))
		window_data['root_mean_sqr']['inner']['inner_msg'] = 'Root Mean Squared Error =  '+str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))

		# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
		# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
		# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
		window_data['status_bar']['inner']['inner_msg1'] = 'Mission Accomplished, whats next..'
		window_data['Computation_status'] = 'Done'
		window_data['compute_prop'] = prop
		#print(window_data['compute_prop'])
		# return prop
		#print('exiting preproces')
		# try:
		# except:
		# 	window_data['status_bar']['inner']['inner_msg1'] = 'An error occured while saving the plot'
		window_data['fun_completed'] = True

	except ValueError:
		window_data['status_bar']['inner']['inner_msg1'] = 'An error occured while computing the data, please change the value in lambda(float)'


def predict_fun():
	window_data['status_bar']['inner']['inner_msg1'] = ''
	try:
		data_set = pandas.read_csv(window_data['predicted_input_path']['inner_rect_prop']['msg'])
	except:
		window_data['status_bar']['inner']['inner_msg1'] = 'Error While loading the data'
		return
	try:
		x = data_set
		X_final_data = pandas.get_dummies(x)
		X_final_data = X_final_data.fillna(X_final_data.mean())

		l = window_data['compute_prop']

		y_pred = l.predict(X_final_data)
			#print(y_pred)
			#rint('type', type(y_pred))
	except:
		window_data['status_bar']['inner']['inner_msg1'] = 'Error while computing the data, please reexamine the data and its shape'
		return

	try:
		data_set_final_predicted = pandas.concat([data_set, pandas.DataFrame(y_pred)], axis=1)

		if 'csv' in window_data['outputfile_name']['inner_rect_prop']['msg']:
			data_set_final_predicted.to_csv(window_data['outputfile_name']['inner_rect_prop']['msg'])
		else:
			data_set_final_predicted.to_csv(window_data['outputfile_name']['inner_rect_prop']['msg']+'.csv', index=False
				)
	except:
		window_data['status_bar']['inner']['inner_msg1'] = 'Error While saving the data'

	window_data['status_bar']['inner']['inner_msg1'] = 'File Uploaded in the directory'
	window_data['fun_completed'] = True

