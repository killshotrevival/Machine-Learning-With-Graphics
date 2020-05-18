import pygame
import support as sp
import pyautogui
import pandas
import pyperclip
from nn_support import *
import time
import threading


screen_height = pyautogui.size()[1]
screen_width = pyautogui.size()[0]

pygame.init()
clock = pygame.time.Clock()
random_color = sp.rand_color()



def neural_window_loop(display_main):
	#overall variables
	close_window = False
	activation_selected = 'relu'
	solver_selected = 'lbfgs'
	pressed = True

	#execution variables
	q = 0
	val = 0
	start_clock = 0.0
	end_clock = 0.0
	time_updated = False
	score_updated = False
	input_active = ''
	freeze = False
	

	while not close_window:
		q+=1
		#print('threads', threading.enumerate())
		if threading.activeCount()==1 and freeze:
			freeze = False
			end_clock = time.time()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit
				quit()
			elif event.type == pygame.MOUSEBUTTONUP:
				pressed = True
			elif event.type == pygame.KEYDOWN:
				if input_active:
					if input_active == 'lambda':
						temp = window_data['regularize_para_p_prop']['inner_rect_prop']['msg']
					elif input_active == 'epocs_count':
						temp = window_data['epocs_para_p_prop']['inner_rect_prop']['msg'] 
					elif input_active == 'predict_path':
						temp = window_data['predicted_input_path']['inner_rect_prop']['msg']
					elif input_active == 'file_name':
						temp = window_data['outputfile_name']['inner_rect_prop']['msg']
					elif input_active == 'hidden_layer':
						temp = window_data['hidden_layers_prop']['inner_rect_prop']['msg']
					elif input_active == 'alpha':
						temp = window_data['alpha_para_p_prop']['inner_rect_prop']['msg']

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

				if input_active == 'lambda':
					window_data['regularize_para_p_prop']['inner_rect_prop']['msg'] = temp
				elif input_active == 'epocs_count':
					window_data['epocs_para_p_prop']['inner_rect_prop']['msg'] = temp
				elif input_active == 'predict_path':
					window_data['predicted_input_path']['inner_rect_prop']['msg'] = temp
				elif input_active == 'file_name':
					window_data['outputfile_name']['inner_rect_prop']['msg'] = temp
				elif input_active == 'hidden_layer':
					window_data['hidden_layers_prop']['inner_rect_prop']['msg'] = temp
				elif input_active == 'alpha':
					window_data['alpha_para_p_prop']['inner_rect_prop']['msg'] = temp
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

		#Class Count
		window_data['classes_rec_prop']['msg'] = 'Classes - '+str(len(set(window_data['data_set']['y'])))
		features_rec = sp.button(window_data['classes_rec_prop'])
		features_rec.draw(display_main, clickable = False, dont_change = True)	


		#learing button
		learning_b = sp.button(window_data['learning_b_prop'], shape='circle')
		window_data['learning_b_prop']['pressed'] = learning_b.draw(display_main, clickable = True, freeze = freeze)
		if window_data['learning_b_prop']['pressed'] and not freeze and pressed:
			freeze = True
			window_data['fun_completed'] = False
			start_clock = time.time()
			val = 0

			#make a thread for the final function
			compute_thread = thread(name = 'Compute_thread', fun = compute )
			#threads.append(compute_thread)
			compute_thread.setDaemon(True)
			compute_thread.start()

			time_updated = False
			total_cost_updated = False
			window_data['time']['inner']['inner_msg'] = 'Time Elapsed : '
			window_data['score']['inner']['inner_msg'] = 'Score :  '

		#Activation
		if activation_selected == 'identity' :
			window_data['activation_rec_prop']['msg'] = 'identity'
		elif activation_selected == 'logistic' :
			window_data['activation_rec_prop']['msg'] = 'logistic'
		elif activation_selected == 'tanh' :
			window_data['activation_rec_prop']['msg'] = 'tanh'
		elif activation_selected == 'relu' :
			window_data['activation_rec_prop']['msg'] = 'relu'

		algo_rec = sp.button(window_data['activation_rec_prop'])
		window_data['activation_rec_prop']['pressed'] = algo_rec.draw(display_main, clickable = True, freeze = freeze)
		if window_data['activation_rec_prop']['pressed'] and pressed and not freeze:
			if activation_selected == 'identity':
				activation_selected = 'logistic'
			elif activation_selected == 'logistic':
				activation_selected = 'tanh'
			elif activation_selected == 'tanh':
				activation_selected = 'relu'
			elif activation_selected == 'relu':
				activation_selected = 'identity'
			pressed = False

		#Hidden layer Props
		hidden_layer = sp.rect_new(window_data['hidden_layers_prop']['outer_rect_prop'], type = 'shadow')
		hidden_layer_box = hidden_layer.draw(display_main)

		hidden_desc = sp.message(window_data['hidden_layers_prop']['hidden_text'], window_data['hidden_layers_prop']['hidden_text_prop'],text_font )
		hidden_desc_box = hidden_desc.message_display(display_main)

		hidden_layer_b = sp.button(window_data['hidden_layers_prop']['inner_rect_prop'])
		window_data['hidden_layers_prop']['inner_rect_prop']['pressed'] = hidden_layer_b.draw(display_main, clickable = True, dont_change = True)

		if window_data['hidden_layers_prop']['inner_rect_prop']['pressed'] and not freeze:
			input_active = 'hidden_layer'
			window_data['hidden_layers_prop']['inner_rect_prop']['msg'] = ''
		

		# Regularize Parameter Prompt
		Regularize_rate_ = sp.rect_new(window_data['regularize_para_p_prop']['outer_rect_prop'], type = 'shadow')
		Regularize_rate_box = Regularize_rate_.draw(display_main)

		lambda_desc = sp.message(window_data['regularize_para_p_prop']['lambda_text'], window_data['regularize_para_p_prop']['lambda_text_prop'],text_font )
		lambda_desc_box = lambda_desc.message_display(display_main)

		Regularize_rate_b = sp.button(window_data['regularize_para_p_prop']['inner_rect_prop'])
		window_data['regularize_para_p_prop']['inner_rect_prop']['pressed'] = Regularize_rate_b.draw(display_main, clickable = True, dont_change = True)

		if window_data['regularize_para_p_prop']['inner_rect_prop']['pressed'] and not freeze:
			input_active = 'lambda'
			window_data['regularize_para_p_prop']['inner_rect_prop']['msg'] = ''

		# #Epocs Count
		epocs_count = sp.rect_new(window_data['epocs_para_p_prop']['outer_rect_prop'], type = 'shadow')
		epocs_count_box = epocs_count.draw(display_main)

		epocs_desc = sp.message(window_data['epocs_para_p_prop']['epocs_text'], window_data['epocs_para_p_prop']['epocs_text_prop'],text_font )
		epocs_desc_box = epocs_desc.message_display(display_main)

		epocs_count_b = sp.button(window_data['epocs_para_p_prop']['inner_rect_prop'])
		window_data['epocs_para_p_prop']['inner_rect_prop']['pressed'] = epocs_count_b.draw(display_main, clickable = True, dont_change = True)

		if window_data['epocs_para_p_prop']['inner_rect_prop']['pressed'] and not freeze:
			input_active = 'epocs_count'
			window_data['epocs_para_p_prop']['inner_rect_prop']['msg'] = ''

		#solver
		if solver_selected == 'lbfgs' :
			window_data['solver_rec_prop']['msg'] = 'lbfgs'
		elif solver_selected == 'sgd' :
			window_data['solver_rec_prop']['msg'] = 'sgd'
		elif solver_selected == 'adam' :
			window_data['solver_rec_prop']['msg'] = 'adam'

		algo_rec = sp.button(window_data['solver_rec_prop'])
		window_data['solver_rec_prop']['pressed'] = algo_rec.draw(display_main, clickable = True, freeze = freeze)
		if window_data['solver_rec_prop']['pressed'] and pressed and not freeze:
			if solver_selected == 'lbfgs':
				solver_selected = 'sgd'
			elif solver_selected == 'sgd':
				solver_selected = 'adam'
			elif solver_selected == 'adam':
				solver_selected = 'lbfgs'
			pressed = False


		#Learning Rate __init__
		alpha_count = sp.rect_new(window_data['alpha_para_p_prop']['outer_rect_prop'], type = 'shadow')
		alpha_count_box = alpha_count.draw(display_main)

		alpha_desc = sp.message(window_data['alpha_para_p_prop']['alpha_text'], window_data['alpha_para_p_prop']['alpha_text_prop'],text_font )
		alpha_desc_box = alpha_desc.message_display(display_main)

		alpha_count_b = sp.button(window_data['alpha_para_p_prop']['inner_rect_prop'])
		window_data['alpha_para_p_prop']['inner_rect_prop']['pressed'] = alpha_count_b.draw(display_main, clickable = True, dont_change = True)

		if window_data['alpha_para_p_prop']['inner_rect_prop']['pressed'] and not freeze:
			input_active = 'alpha'
			window_data['alpha_para_p_prop']['inner_rect_prop']['msg'] = ''
		

###############################################################################################################################################################################################

		#Learning Process Fileds

		#Progress Bar
		progress_desc = sp.message(window_data['progress_bar']['text']['text_msg'], window_data['progress_bar']['text']['text_prop'])
		progress_box = progress_desc.message_display(display_main)

		progress_bar_ = sp.rect_new(window_data['progress_bar']['progress_bar_prop'], type = 'progress')
		if q%40==0 and val<98 and freeze:
			val+=1
		if window_data['fun_completed']:
			val = 100
		progress_bar_box = progress_bar_.draw(display_main, val)

		pygame.draw.rect(display_main, (10,10,10),(600, 145, 55,50))
		window_data['progress_bar']['percent']['percent_msg'] = str(val)+'%'
		progress_percent_desc = sp.message(window_data['progress_bar']['percent']['percent_msg'],window_data['progress_bar']['percent']['percent_prop'])
		progress_percent_box = progress_percent_desc.message_display(display_main)

		#Score
		score_ = sp.rect_new(window_data['score']['outer_rect'], type = 'normal')
		score_box = score_.draw(display_main)

		if not score_updated:
			window_data['score']['inner']['inner_msg'] = window_data['score']['inner']['inner_msg']+ str(round(end_clock-start_clock,2))
			score_updated = True
		score_msg = sp.message(window_data['score']['inner']['inner_msg'],window_data['score']['inner']['inner_prop'] )
		score_msg_box = score_msg.message_display(display_main)


		#time elapsed
		time_ = sp.rect_new(window_data['time']['outer_rect'], type = 'normal')
		time_box = time_.draw(display_main)

		if not time_updated and not freeze:
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

		if window_data['predicted_input_path']['inner_rect_prop']['pressed'] and not freeze:
			if window_data['Computation_status'] == 'Done':
				input_active = 'predict_path'
				window_data['predicted_input_path']['inner_rect_prop']['msg'] = ''
			else:
				window_data['status_bar']['inner']['inner_msg1'] = 'Let me learn the data first, then only I can make some prediction XD'


		#Name of Outupt file
		o_input_ = sp.rect_new(window_data['outputfile_name']['outer_rect'], type = 'normal')
		o_input_box = o_input_.draw(display_main)

		outputfile_inner = sp.button(window_data['outputfile_name']['inner_rect_prop'])
		window_data['outputfile_name']['inner_rect_prop']['pressed'] = outputfile_inner.draw(display_main, clickable = True, dont_change = True)

		if window_data['outputfile_name']['inner_rect_prop']['pressed'] and not freeze:
			if window_data['Computation_status'] == 'Done':
				input_active = 'file_name'
				window_data['outputfile_name']['inner_rect_prop']['msg'] = ''
			else:
				window_data['status_bar']['inner']['inner_msg1'] = 'Let me learn the data first, then only I can make some prediction XD'
		
		#Predict Button
		predict_b = sp.button(window_data['predict_b_prop'], shape='rect')
		window_data['predict_b_prop']['pressed'] = predict_b.draw(display_main, clickable = True, dont_change =False, freeze = freeze)

		if window_data['predict_b_prop']['pressed'] and not freeze :
			if window_data['Computation_status'] == 'Done':
				predict_thread = thread(name = 'predict', fun = predict_fun)
				predict_thread.start()

				window_data['fun_completed'] = False
				val = 0
				freeze = True
			else:
				window_data['status_bar']['inner']['inner_msg1'] = 'Let me learn the data first, then only I can make some prediction XD'
		




###############################################################################################################################################################################################

		#status Bar
		status_bar_ = sp.rect_new(window_data['status_bar']['outer_rect'], type = 'normal')
		mean_sqr_box = status_bar_.draw(display_main)

		status_bar_msg = sp.message(window_data['status_bar']['inner']['inner_msg'],window_data['status_bar']['inner']['inner_prop'] )
		status_bar_msg_box = status_bar_msg.message_display(display_main)

		status_msg = sp.message(window_data['status_bar']['inner']['inner_msg1'],window_data['status_bar']['inner']['inner_prop1'] )
		status_msg_box = status_msg.message_display(display_main)


###############################################################################################################################################################################################


		pygame.display.update()
		clock.tick(60)


def neural_window_start(data_set=None):
	if data_set is not None:
		window_data['data_set']['X'] = data_set['X']
		window_data['data_set']['y'] = data_set['y']
	#print('Data set', window_data['data_set']) 
	neural_window = sp._window(window_data['window_prop'])
	display_main = neural_window.create_window()
	neural_window_loop(display_main)


#neural_window_start()
#C:\Users\91981\Desktop\ML and DL Projects\LoanPrediction\train.csv

