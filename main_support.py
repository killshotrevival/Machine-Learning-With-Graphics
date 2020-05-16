import pandas
import support as sp
import linear,logistic,nn,svm

random_color = sp.rand_color()

window_data = {
				'window_prop':{'height': 700,
								'width' : 800,
								'name'  : 'Machine Learning With Graphics'
								},
				'background_color':(250,250,250),
				'title' : 'MLG',
				'title_prop' : (400,50,50, random_color.genrate()),       #(width, height, size, color)
				'desc' : 'Machine Learning With Graphics',
				'desc_prop' : (400, 90, 15, random_color.genrate()),      #(width, height, size, color)
				'error_msg' : {
								'Error' : 'Error Occured Please load the data in the required format only',
								'No Error' : 'Data loaded sucessfully, proceed with algo selection'
				},
				'error_msg_prop' : (400, 670, 20, (161,161,161)),
				'format': 'Format for loading the database',
				'format_prop': (400, 450, 25, random_color.genrate()),
				'data_path_prop': (360, 635, 25, (0,0,0) ),                         #(width, height, size, color)
				'linear_regression_b_prop': linear.window_data['main_window_b_prop'],
				'logistic_regression_b_prop': logistic.window_data['main_window_b_prop'], 
				'neural_network_b_prop': nn.window_data['main_window_b_prop'],
				'svm_b_prop': svm.window_data['main_window_b_prop'],
				'path_box_prop' : {
										'x':50,
										'y':610,
										'width':600,
										'height':50,
										'color':(225, 225, 225),
										'text_color':(0,0,0),
										'text_size':20
									},                        							
				'paste_b_prop': {
									'msg'  :'Load',
									'button_x': 660,
									'button_y': 610,
									'button_width' : 100,
									'button_height' : 50,
									'inactive_color' : (192,192,192),
									'textfont' : 'modernno20',
									'text_size' : 30,
									'inactive_text_color' : (0,0,0),
									'pressed' : False

								},
				'database_processed': {'X': pandas.DataFrame(),
										'y': pandas.DataFrame()
									}
}