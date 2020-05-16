import support as sp 

import pandas  
import numpy as np  
import matplotlib.pyplot as plt  
#import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import threading


random_color = sp.rand_color()

font_color = (0,191,255)
text_font = 'freesansbold.ttf'

window_data = {'window_prop':{'height': 700,
								'width' : 1540,
								'name'  : 'Linear Regression'
								},
				'main_window_b_prop' : {
						'msg'  :'Linear Regression',
						'button_x': 50,
						'button_y': 150,
						'button_width' : 310,
						'button_height' : 50,
						'inactive_color' : (245,245,245),
						'textfont' : 'modernno20',
						'text_size' : 30,
						'inactive_text_color' : font_color,
						'pressed' : False

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
				'learning_rate_p_prop' : {
											'outer_rect_prop':{
														'x':810,
														'y': 25,
														'width':200,
														'height':50,
														'foreground_color':(240,240,240),
														'background_color': font_color

															},
											'alpha_text':'Alpha',
											'alpha_text_prop':(840,53,30,font_color),       #(width, height, size, color))
											'inner_rect_prop': {
																'msg'  :'',
																'button_x': 875,
																'button_y': 27,
																'button_width' : 125,
																'button_height' : 40,
																'inactive_color' : (220,220,220),
																'textfont' : text_font,
																'text_size' : 30,
																'inactive_text_color' : (0,0,0),
																'pressed' : False
															}
											},

				'regularize_para_p_prop' : {
											'outer_rect_prop':{
														'x':1040,
														'y': 25,
														'width':220,
														'height':50,
														'foreground_color':(240,240,240),
														'background_color': font_color

															},
											'lambda_text':'Lambda',
											'lambda_text_prop':(1080,53,30,font_color),       #(width, height, size, color))
											'inner_rect_prop': {

																'msg'  :'',
																'button_x': 1125,
																'button_y': 27,
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
														'x':1290,
														'y': 25,
														'width':200,
														'height':50,
														'foreground_color':(240,240,240),
														'background_color': font_color

															},
											'text':{
													'text_msg':'Graph',
													'text_prop':(1390,50,30,font_color)
											},
											'check_box':{
														'x':1300,
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
												'text_prop': (320, 130, 30, font_color),	
													},
										'progress_bar_prop':{
															'x':70,
															'y':150,
															'width':500,
															'height':30,
															'foreground_color':(0,255,0),
															'background_color': (245,245,245)
																},
										'percent':{
													'percent_msg':'',
													'percent_prop': (620,170,30, font_color)
														}
										},
						'total_cost':{
									'outer_rect':{
												'x':60,
												'y':210,
												'width':520,
												'height':40,
												'foreground_color':(240,230,230)	
													},
									'inner':{
											'inner_msg':'Total Cost =  ',
											'inner_prop': (320,230,30, font_color)
													}	
										},
						'time': {
								'outer_rect':{
											'x':60,
											'y':280,
											'width':520,
											'height':40,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Time Elapsed :',
									'inner_prop': (320,300,30, font_color)
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

						'mean_abs': {

								'outer_rect':{
											'x':830,
											'y':337,
											'width':500,
											'height':50,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Mean Abs Error :',
									'inner_prop': (1080,362,30, font_color)
										
										}	
										},

						'mean_sqr': {

								'outer_rect':{
											'x':830,
											'y':417,
											'width':500,
											'height':50,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Mean Sqr Error :',
									'inner_prop': (1080,442,30, font_color)
										
										}	
										},

						'root_mean_sqr': {

								'outer_rect':{
											'x':830,
											'y':497,
											'width':500,
											'height':50,
											'foreground_color':(240,230,230)
											},
								'inner':{
									'inner_msg': 'Root Mean Sqr Error :',
									'inner_prop': (1080,522,30, font_color)
										
										}	
										},
						'predict_b_prop':{
											'msg'  :'Predict',
											'button_x': 1350,
											'button_y': 187,
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
											'inner_msg1': 'dfjbkdfdjsfbd Error',
											'inner_prop1': (770,680,25, (0, 0, 0))	
											}



									}
}



# def compute():
# 	X_train, X_test, y_train, y_test = train_test_split(window_data['data_set']['X'], window_data['data_set']['y'], test_size=0.2, random_state=0)


class thread(threading.Thread):
	def __init__(self, name, fun):
		threading.Thread.__init__(self)
		self.name = name
		self.fun = fun
	def run(self):
		self.fun()