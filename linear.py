import pygame
import support as sp
import pyautogui
import pandas
import pyperclip
from linear_support import *
import time
	
screen_height = pyautogui.size()[1]
screen_width = pyautogui.size()[0]

pygame.init()
clock = pygame.time.Clock()
random_color = sp.rand_color()



def linear_window_loop(display_main):
	#overall variables
	close_window = False
	algo_selected = 'GD'
	pressed = True

	#execution variables
	q = 0
	val = 0
	total_cost_value = 0.0
	total_cost_updated = False
	start_clock = 0.0
	end_clock = 0.0
	time_updated = False
	mean_abs_updated = False
	mean_sqr_updated = False
	root_mean_sqr_updated = False
	input_active = ''
	freeze = False

	trained = False

	while not close_window:
		q+=1
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit
				quit()
			elif event.type == pygame.MOUSEBUTTONUP:
				pressed = True
			elif event.type == pygame.KEYDOWN:
				if input_active:
					if input_active == 'alpha':
						temp = window_data['learning_rate_p_prop']['inner_rect_prop']['msg'] 
					elif input_active == 'lambda':
						temp = window_data['regularize_para_p_prop']['inner_rect_prop']['msg'] 
					elif input_active == 'predict_path':
						temp = window_data['predicted_input_path']['inner_rect_prop']['msg']
					elif input_active == 'file_name':
						temp = window_data['outputfile_name']['inner_rect_prop']['msg']

					if event.key == pygame.K_RETURN:
						input_active = ''
					elif event.key == pygame.K_BACKSPACE:
						temp = temp[:-1]
					elif event.key == pygame.K_v:
						mods = pygame.key.get_mods()

						if mods & pygame.KMOD_CTRL:
							copied = True
							temp = pyperclip.paste()
						else:
							temp+=event.unicode
					else:
						temp+=event.unicode

				if input_active == 'alpha':
					window_data['learning_rate_p_prop']['inner_rect_prop']['msg'] = temp
				elif input_active == 'lambda':
					window_data['regularize_para_p_prop']['inner_rect_prop']['msg'] = temp
				elif input_active == 'predict_path':
					window_data['predicted_input_path']['inner_rect_prop']['msg'] = temp
				elif input_active == 'file_name':
					window_data['outputfile_name']['inner_rect_prop']['msg'] = temp

		if not freeze:
			display_main.fill(window_data['background_color'])

		#Main Tools

		#DataSet Size		
		window_data['data_set_size_rec_prop']['msg'] = ' m- ' +str(window_data['data_set']['X'].shape[0])	
		data_set_size_rec = sp.button(window_data['data_set_size_rec_prop'])
		data_set_size_rec.draw(display_main, clickable = False, dont_change = True)	

		#Features Count
		window_data['features_rec_prop']['msg'] = 'n- '+str(window_data['data_set']['X'].shape[1])
		features_rec = sp.button(window_data['features_rec_prop'])
		features_rec.draw(display_main, clickable = False, dont_change = True)	

		#Optimizing Algorithm
		if algo_selected == 'GD' :
			window_data['algo_rec_prop']['msg'] = 'Gradient Descent'
		elif algo_selected == 'NE' :
			window_data['algo_rec_prop']['msg'] = 'Normal Equation'

		algo_rec = sp.button(window_data['algo_rec_prop'])
		window_data['algo_rec_prop']['pressed'] = algo_rec.draw(display_main, clickable = True, freeze = freeze)
		if window_data['algo_rec_prop']['pressed'] and pressed and not freeze:
			if algo_selected == 'NE':
				algo_selected = 'GD'
			elif algo_selected == 'GD':
				algo_selected = 'NE'
			pressed = False

		#learing button
		learning_b = sp.button(window_data['learning_b_prop'], shape='circle')
		window_data['learning_b_prop']['pressed'] = learning_b.draw(display_main, clickable = True, freeze = freeze)
		if window_data['learning_b_prop']['pressed']:
			start_clock = time.time()

			#make a thread for the final function

			end_clock = time.time()
			time_updated = False
			total_cost_updated = False
			window_data['time']['inner']['inner_msg'] = 'Time Elapsed : '
			window_data['total_cost']['inner']['inner_msg'] = 'Total Cost :  '
			
		#learning rate Prompt
		learning_rate_ = sp.rect_new(window_data['learning_rate_p_prop']['outer_rect_prop'], type = 'shadow')
		learning_rate_box = learning_rate_.draw(display_main)

		alpha_desc = sp.message(window_data['learning_rate_p_prop']['alpha_text'], window_data['learning_rate_p_prop']['alpha_text_prop'], text_font)
		alpha_desc_box = alpha_desc.message_display(display_main)

		learning_rate_b = sp.button(window_data['learning_rate_p_prop']['inner_rect_prop'])
		window_data['learning_rate_p_prop']['inner_rect_prop']['pressed'] = learning_rate_b.draw(display_main, clickable = True, dont_change = True)

		if window_data['learning_rate_p_prop']['inner_rect_prop']['pressed']:
			input_active = 'alpha'
			window_data['learning_rate_p_prop']['inner_rect_prop']['msg'] = ''

		#Regularize Parameter Prompt
		Regularize_rate_ = sp.rect_new(window_data['regularize_para_p_prop']['outer_rect_prop'], type = 'shadow')
		Regularize_rate_box = Regularize_rate_.draw(display_main)

		lambda_desc = sp.message(window_data['regularize_para_p_prop']['lambda_text'], window_data['regularize_para_p_prop']['lambda_text_prop'],text_font )
		lambda_desc_box = lambda_desc.message_display(display_main)

		Regularize_rate_b = sp.button(window_data['regularize_para_p_prop']['inner_rect_prop'])
		window_data['regularize_para_p_prop']['inner_rect_prop']['pressed'] = Regularize_rate_b.draw(display_main, clickable = True, dont_change = True)

		if window_data['regularize_para_p_prop']['inner_rect_prop']['pressed']:
			input_active = 'lambda'
			window_data['regularize_para_p_prop']['inner_rect_prop']['msg'] = ''

		#Graph Needed
		graph_ = sp.rect_new(window_data['graph_needed']['outer_rect_prop'], type = 'shadow')
		graph_box = graph_.draw(display_main)

		graph_text = sp.message(window_data['graph_needed']['text']['text_msg'], window_data['graph_needed']['text']['text_prop'],text_font )
		graph_text_box = graph_text.message_display(display_main)

		graph_check = sp.check_box(window_data['graph_needed']['check_box'])
		state = graph_check.draw(display_main)
		if state and pressed:
			if window_data['graph_needed']['check_box']['state'] == 'inactive':
				window_data['graph_needed']['check_box']['state'] = 'active'

			elif window_data['graph_needed']['check_box']['state'] == 'active':
				window_data['graph_needed']['check_box']['state'] = 'inactive'
			pressed = False
		

###############################################################################################################################################################################################

		#Learning Process Fileds

		#Progress Bar
		progress_desc = sp.message(window_data['progress_bar']['text']['text_msg'], window_data['progress_bar']['text']['text_prop'])
		progress_box = progress_desc.message_display(display_main)

		progress_bar_ = sp.rect_new(window_data['progress_bar']['progress_bar_prop'], type = 'progress')
		if q%60==0:
			val+=1
		progress_bar_box = progress_bar_.draw(display_main, val)

		window_data['progress_bar']['percent']['percent_msg'] = str(val)+'%'
		progress_percent_desc = sp.message(window_data['progress_bar']['percent']['percent_msg'],window_data['progress_bar']['percent']['percent_prop'])
		progress_percent_box = progress_percent_desc.message_display(display_main)

		#Total Cost
		total_cost_ = sp.rect_new(window_data['total_cost']['outer_rect'], type = 'normal')
		total_cost_box = total_cost_.draw(display_main)

		if not total_cost_updated:
			window_data['total_cost']['inner']['inner_msg'] = window_data['total_cost']['inner']['inner_msg']+str(total_cost_value)
			total_cost_updated = True
		total_cost_desc = sp.message(window_data['total_cost']['inner']['inner_msg'], window_data['total_cost']['inner']['inner_prop'])
		total_cost_box = total_cost_desc.message_display(display_main)

		#time elapsed
		time_ = sp.rect_new(window_data['time']['outer_rect'], type = 'normal')
		time_box = time_.draw(display_main)

		if not time_updated:
			window_data['time']['inner']['inner_msg'] = window_data['time']['inner']['inner_msg']+ str(round(end_clock-start_clock,2))+' seconds'
			time_updated = True
		time_msg = sp.message(window_data['time']['inner']['inner_msg'],window_data['time']['inner']['inner_prop'] )
		time_msg_box = time_msg.message_display(display_main)


###############################################################################################################################################################################################


		#Predicted Fields

		#Input File Path 
		input_path_ = sp.message(window_data['predicted_input_path']['text']['text_msg'],window_data['predicted_input_path']['text']['text_prop'])
		input_path_box = input_path_.message_display(display_main)


		p_input_ = sp.rect_new(window_data['predicted_input_path']['outer_rect'], type = 'normal')
		p_input_box = p_input_.draw(display_main)

		input_path_inner = sp.button(window_data['predicted_input_path']['inner_rect_prop'])
		window_data['predicted_input_path']['inner_rect_prop']['pressed'] = input_path_inner.draw(display_main, clickable = True, dont_change = True)

		if window_data['predicted_input_path']['inner_rect_prop']['pressed']:
			input_active = 'predict_path'
			window_data['predicted_input_path']['inner_rect_prop']['msg'] = ''

		#Name of Outupt file
		o_input_ = sp.rect_new(window_data['outputfile_name']['outer_rect'], type = 'normal')
		o_input_box = o_input_.draw(display_main)

		outputfile_inner = sp.button(window_data['outputfile_name']['inner_rect_prop'])
		window_data['outputfile_name']['inner_rect_prop']['pressed'] = outputfile_inner.draw(display_main, clickable = True, dont_change = True)

		if window_data['outputfile_name']['inner_rect_prop']['pressed']:
			input_active = 'file_name'
			window_data['outputfile_name']['inner_rect_prop']['msg'] = ''

		#Mean Absolute Error
		mean_abs_ = sp.rect_new(window_data['mean_abs']['outer_rect'], type = 'normal')
		mean_abs_box = mean_abs_.draw(display_main)

		if not time_updated:
			window_data['mean_abs']['inner']['inner_msg'] = window_data['mean_abs']['inner']['inner_msg']+ str(round(end_clock-start_clock,2))
			mean_abs_updated = True
		mean_abs_msg = sp.message(window_data['mean_abs']['inner']['inner_msg'],window_data['mean_abs']['inner']['inner_prop'] )
		mean_abs_msg_box = mean_abs_msg.message_display(display_main)

		#Mean Square Error
		mean_sqr_ = sp.rect_new(window_data['mean_sqr']['outer_rect'], type = 'normal')
		mean_sqr_box = mean_sqr_.draw(display_main)

		if not time_updated:
			window_data['mean_sqr']['inner']['inner_msg'] = window_data['mean_sqr']['inner']['inner_msg']+ str(round(end_clock-start_clock,2))
			mean_sqr_updated = True
		mean_sqr_msg = sp.message(window_data['mean_sqr']['inner']['inner_msg'],window_data['mean_sqr']['inner']['inner_prop'] )
		mean_sqr_msg_box = mean_sqr_msg.message_display(display_main)

		#Root Mean Square Error
		root_mean_sqr_ = sp.rect_new(window_data['root_mean_sqr']['outer_rect'], type = 'normal')
		root_mean_sqr_box = root_mean_sqr_.draw(display_main)

		if not time_updated:
			window_data['root_mean_sqr']['inner']['inner_msg'] = window_data['root_mean_sqr']['inner']['inner_msg']+ str(round(end_clock-start_clock,2))
			root_mean_sqr_updated = True
		root_mean_sqr_msg = sp.message(window_data['root_mean_sqr']['inner']['inner_msg'],window_data['root_mean_sqr']['inner']['inner_prop'] )
		root_mean_sqr_msg_box = root_mean_sqr_msg.message_display(display_main)

		#Predict Button
		predict_b = sp.button(window_data['predict_b_prop'], shape='rect')
		window_data['predict_b_prop']['pressed'] = predict_b.draw(display_main, clickable = True, dont_change =False)



###############################################################################################################################################################################################

		#status Bar
		status_bar_ = sp.rect_new(window_data['status_bar']['outer_rect'], type = 'normal')
		mean_sqr_box = status_bar_.draw(display_main)

		status_bar_msg = sp.message(window_data['status_bar']['inner']['inner_msg'],window_data['status_bar']['inner']['inner_prop'] )
		status_bar_msg_box = status_bar_msg.message_display(display_main)

		status_msg = sp.message(window_data['status_bar']['inner']['inner_msg1'],window_data['status_bar']['inner']['inner_prop1'] )
		status_msg_box = status_msg.message_display(display_main)


###############################################################################################################################################################################################
		
		#print('input_active = ', input_active)
		pygame.display.update()
		clock.tick(60)


def linear_window_start(data_set=None):
	if data_set is not None:
		window_data['data_set']['X'] = data_set['X']
		window_data['data_set']['y'] = data_set['y']
	#print('Data set', window_data['data_set'])
	linear_window = sp._window(window_data['window_prop'])
	display_main = linear_window.create_window()
	linear_window_loop(display_main)


linear_window_start()
#C:\Users\91981\Desktop\ML and DL Projects\NNG\winequality-red.csv
