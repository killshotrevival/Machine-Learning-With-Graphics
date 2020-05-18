import pygame
import random
import pandas
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer

# import os
# os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10,30)

pygame.init()
enc = OneHotEncoder()
label_bi = LabelBinarizer()


class rand_color:
	def genrate(self):
		return tuple(random.sample(range(255),3))

print
class _window:
	def __init__(self,window_data):
		self.window_width = window_data['width']
		self.window_height = window_data['height']
		self.window_name = window_data['name']

	def create_window(self):
		pygame.display.set_caption(self.window_name)
		return pygame.display.set_mode((self.window_width,self.window_height))


class message:
	def __init__(self,text, prop, font='freesansbold.ttf', trippy = False):
		self.text = text
		self.size = prop[2]
		self.x = prop[1]
		self.y = prop[0]
		self.color = prop[3]
		self.font = font
		self.trippy = trippy


	def text_objects(self):
		text_font = pygame.font.SysFont(self.font,self.size)
		if self.trippy:
			random_color = rand_color()
			text_surface = text_font.render(self.text, True, random_color.genrate())
		else:
			text_surface = text_font.render(self.text, True, self.color)
		return text_surface, text_surface.get_rect()


	def message_display(self, gameDisplay):
		text_surface,text_rect = self.text_objects()
		text_rect.center = (self.y,self.x)
		gameDisplay.blit(text_surface,text_rect)
		return text_rect


class button:
	def __init__(self,prop, shape = 'rect'):
		self.msg = prop['msg']
		self.x = prop['button_x']
		self.y = prop['button_y']
		self.width = prop['button_width']
		self.height = prop['button_height']
		self.inactive_color = prop['inactive_color']
		self.active_color = prop['inactive_text_color']
		#self.action = action
		self.text_font = prop['textfont']
		self.text_size = prop['text_size']
		self.inactive_text_color = prop['inactive_text_color']
		self.active_text_color = prop['inactive_color']
		self.button_shape = shape


	def draw(self,gameDisplay, clickable = True, dont_change = False, freeze = False):
		mouse = pygame.mouse.get_pos()
		click = pygame.mouse.get_pressed()
		if freeze == True:
			dont_change = True
		if not dont_change:
			if self.button_shape == 'rect':
				if self.x+self.width>mouse[0]>self.x and self.y+self.height>mouse[1]>self.y:
					pygame.draw.rect(gameDisplay,self.active_color,(self.x, self.y, self.width, self.height))
					prop = [int((self.x+(self.width/2))), int((self.y+(self.height/2))), self.text_size, self.active_text_color]
					msg = message(self.msg, prop, self.text_font)
					msg.message_display(gameDisplay)
					if click[0]==1:
						if clickable:
							return True
				else:
					pygame.draw.rect(gameDisplay,self.inactive_color,(self.x, self.y, self.width, self.height))

					#(width, height, size, color)
					prop = [int((self.x+(self.width/2))), int((self.y+(self.height/2))), self.text_size, self.inactive_text_color]
					msg = message(self.msg,prop,self.text_font)
					msg.message_display(gameDisplay)

			elif self.button_shape == 'circle':
				r = int(self.width/2)
				if self.x+self.width>mouse[0]>self.x and self.y+self.height>mouse[1]>self.y:
					pygame.draw.circle(gameDisplay,self.active_color,(self.x+r, self.y+r), r)
					prop = [self.x+r, self.y+r, self.text_size, self.active_text_color] #(width, height, size, color)
					msg = message(self.msg, prop, self.text_font)
					msg.message_display(gameDisplay)
					if click[0]==1:
						if clickable:
							return True
				else:
					pygame.draw.circle(gameDisplay,self.inactive_color,(self.x+r, self.y+r), r)
					prop = [self.x+r, self.y+r, self.text_size, self.inactive_text_color] #(width, height, size, color)
					msg = message(self.msg,prop,self.text_font)
					msg.message_display(gameDisplay)
		else:
			if self.button_shape == 'rect':
				pygame.draw.rect(gameDisplay,self.inactive_color,(self.x, self.y, self.width, self.height))

				#(width, height, size, color)
				prop = [int((self.x+(self.width/2))), int((self.y+(self.height/2))), self.text_size, self.inactive_text_color]
				msg = message(self.msg,prop,self.text_font)
				msg.message_display(gameDisplay)


			elif self.button_shape == 'circle':
				r = int(self.width/2)
				pygame.draw.circle(gameDisplay,self.inactive_color,(self.x+r, self.y+r), r)
				prop = [self.x+r, self.y+r, self.text_size, self.inactive_text_color] #(width, height, size, color)
				msg = message(self.msg,prop,self.text_font)
				msg.message_display(gameDisplay)
			if self.x+self.width>mouse[0]>self.x and self.y+self.height>mouse[1]>self.y:
				if click[0]==1:
					#print('Coming inside')
					if clickable:
						return True


class load_data:
	def __init__(self, path):
		self.path = path

	def load(self):
		try:
			self.data_set = pandas.read_csv(self.path)
		except:
			return 'Error'
		return 'No Error'

	def pre_process(self):

		#preproces
		#print('INside preproces')
		x = self.data_set.iloc[:,:-1]
		self.y_final_data = self.data_set.iloc[:,-1]
		self.X_final_data = pandas.get_dummies(x)
		self.X_final_data = self.X_final_data.fillna(self.X_final_data.mean())
		#print('exiting preproces')
		return self.X_final_data, self.y_final_data

class rect_new:
	def __init__(self,prop, type = 'normal'):
		self.x = prop['x']
		self.y = prop['y']
		self.width = prop['width']
		self.height = prop['height']
		self.fcolor = prop['foreground_color']
		self.type = type
		if self.type == 'shadow' or self.type == 'progress' :
			self.bcolor = prop['background_color']

	def draw(self,gameDisplay,progress = 0):
		if self.type == 'shadow':
			pygame.draw.rect(gameDisplay, self.bcolor, (self.x, self.y, self.width, self.height))
			pygame.draw.rect(gameDisplay, self.fcolor, (self.x-5, self.y-5, self.width, self.height))
		elif self.type == 'normal':
			pygame.draw.rect(gameDisplay, self.fcolor, (self.x, self.y, self.width, self.height))
		elif self.type == 'progress':
			prg_width = (self.width/100)*progress
			pygame.draw.rect(gameDisplay, self.bcolor, (self.x-10, self.y-5, self.width+20, self.height+10))
			pygame.draw.rect(gameDisplay, self.fcolor, (self.x, self.y, prg_width, self.height))

class check_box:
	def __init__(self, prop):
		self.x = prop['x']
		self.y = prop['y']
		self.width = prop['width']
		self.height = prop['height']
		self.state = prop['state']
		self.active_color = prop['active_color']
		self.inactive_color = prop['inactive_color']

	def draw(self,gameDisplay):
		mouse = pygame.mouse.get_pos()
		click = pygame.mouse.get_pressed()

		if self.state == 'inactive':
			pygame.draw.rect(gameDisplay,self.inactive_color, (self.x, self.y, self.width, self.height))

		elif self.state == 'active':
			pygame.draw.rect(gameDisplay,self.active_color, (self.x, self.y, self.width, self.height))


		if self.x+self.width>mouse[0]>self.x and self.y+self.height>mouse[1]>self.y:
			if click[0]==1:
				return True
			else:
				return False
		else:
			return False
