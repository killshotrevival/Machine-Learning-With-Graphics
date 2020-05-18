import pygame
import support as sp
import linear,logistic,nn
import pyperclip
import pandas
#from pandas import ExcelWriter
from main_support import *

pygame.init()
clock = pygame.time.Clock()
random_color = sp.rand_color()
#writer1 = ExcelWriter(r'C:\Users\91981\Desktop\ML and DL Projects\Loan Prediction\oooohlll.xlsx')
#writer2 = ExcelWriter(r'C:\Users\91981\Desktop\ML and DL Projects\Loan Prediction\hellooowindow.xlsx')

def main_window_loop(display_main):
	database_path = 'File Address Here..'
	typing = False
	data_loaded = False
	display_error = False
	error_occuried = False
	close_window = False

	while not close_window:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit
				quit()

			if event.type == pygame.MOUSEBUTTONDOWN:
				mouse = pygame.mouse.get_pos()
				if window_data['path_box_prop']['x']+window_data['path_box_prop']['width']>mouse[0]>window_data['path_box_prop']['x'] and window_data['path_box_prop']['y']+window_data['path_box_prop']['height']>mouse[1]>window_data['path_box_prop']['y']:
					#if not copied:
					database_path = ''
					typing = True

			if event.type == pygame.KEYDOWN:
				if typing:
					if event.key == pygame.K_RETURN:
						typing = False
						if not data_loaded:
							error_status = load_data_main(database_path)
							if error_status:
								error_occuried = True
								data_loaded = False
							else:
								error_occuried = False
								data_loaded = True		
						display_error = True

					elif event.key ==pygame.K_BACKSPACE:
						if database_path:
							database_path = database_path[:-1]

					elif event.key == pygame.K_v:

						mods = pygame.key.get_mods()

						if mods & pygame.KMOD_CTRL:
							copied = True
							database_path = pyperclip.paste()
						else:
							database_path+=event.unicode
					else:
						database_path+=event.unicode


		display_main.fill(window_data['background_color'])

		# message.__init__ = text, prop , color, font='freesansbold.ttf'
		title = sp.message(window_data['title'],window_data['title_prop'])
		title_desc = sp.message(window_data['desc'], window_data['desc_prop'], 'lucidahandwriting')
		title_box = title.message_display(display_main)
		title_desc_box = title_desc.message_display(display_main)


		#buttons
		#1)Linear Regression
		linear_regression_b = sp.button(window_data['linear_regression_b_prop'])
		window_data['linear_regression_b_prop']['pressed'] = linear_regression_b.draw(display_main)
		if window_data['linear_regression_b_prop']['pressed']:
			#quit()
			if data_loaded == True:
				start_linear_loop()

		#2)Logistic Regression
		logistic_regression_b = sp.button(window_data['logistic_regression_b_prop'])
		window_data['logistic_regression_b_prop']['pressed'] = logistic_regression_b.draw(display_main)
		if window_data['logistic_regression_b_prop']['pressed']:
			#quit()
			if data_loaded == True:
				start_logistic_loop()


		#3)Neural Networks
		neural_network_b = sp.button(window_data['neural_network_b_prop'])
		window_data['neural_network_b_prop']['pressed'] = neural_network_b.draw(display_main)
		if window_data['neural_network_b_prop']['pressed']:
			#quit()
			if data_loaded == True:
				start_neural_loop()


		#Database
		format = sp.message(window_data['format'], window_data['format_prop'])
		format_box = format.message_display(display_main)

		image = pygame.image.load(r'C:\\Users\91981\\Desktop\\ML and DL Projects\NNG\\format.png')

		display_main.blit(image, (130,470))

		#Path of the database
		pygame.draw.rect(display_main,window_data['path_box_prop']['color'],(window_data['path_box_prop']['x'], window_data['path_box_prop']['y'], window_data['path_box_prop']['width'], window_data['path_box_prop']['height']) )

		#paste button
		paste_b = sp.button(window_data['paste_b_prop'])
		window_data['paste_b_prop']['pressed'] = paste_b.draw(display_main)
		if window_data['paste_b_prop']['pressed']:
			if not data_loaded:
				error_status = load_data_main(database_path)
				if error_status:
					error_occuried = True
					data_loaded = False
				else:
					error_occuried = False
					data_loaded = True		
			display_error = True

		data_path = sp.message(database_path, window_data['data_path_prop'])
		data_path_box = data_path.message_display(display_main)

		if error_occuried:
			error_m = window_data['error_msg']['Error']
		else:
			error_m = window_data['error_msg']['No Error']
		if display_error:
			error_msg = sp.message(error_m, window_data['error_msg_prop'])
			error_msg_box = error_msg.message_display(display_main)

		pygame.display.update()
		clock.tick(60)

def load_data_main(path):
	data = sp.load_data(path)
	error_status = data.load()
	if error_status=='Error':
		return True
	else:
		#data_processed = data.pre_process()
		#data_processed.to_excel(writer1, 'Sheet1', index=True)
		#writer1.save()
		window_data['database_processed']['X'], window_data['database_processed']['y'] = data.pre_process()
		return False

def start_linear_loop():
	linear.linear_window_start(window_data['database_processed'])

def start_logistic_loop():
	logistic.logistic_window_start(window_data['database_processed'])

def start_neural_loop():
	nn.neural_window_start(window_data['database_processed'])
	 


def main_window_start():
	main_window = sp._window(window_data['window_prop'])
	display_main = main_window.create_window()
	main_window_loop(display_main)

main_window_start()
