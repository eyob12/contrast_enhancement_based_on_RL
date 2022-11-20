"""Designing GUI with tkinter"""
import tkinter as tk
from tkinter import messagebox as tkMessageBox, filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gLobal as gl

EPISODES = 50

class GUI(tk.Tk, object):
	"""Models gui environment"""
	def __init__(self, N=10):
		super().__init__()
		self.option_add('*Font', 'Times')
		self.title('Reinforcement Learning Project')
		self.rate: int = None
		self.flag: bool = False
		self.imsel: bool = False
		self.skip: bool = False
		self.log: bool = False
		self.notify = tk.StringVar(value='Deteriorated Image')
		self.perform = tk.StringVar(value="Go to test")
		self.policy_sel = tk.StringVar(value="Use pre-trained model") 
		self.mode = True #for training
		self.mod = True # train the model
		self.build_gui()
		self.rates: List[int] = []
		self.reward: List[float] = []
		self.punishment: List[float] = [] #punishments
		self.last_n: List[float] = []
		self.rpr: List[float] = [] # history of reward to punishment ratio
		self.running_ave: List[float] = []
		self.total_ave: List[float] = []
		self._N = N
		self.itr = 0

	def build_gui(self):
		"""builds the GUI"""
		#create window object

		#declare all window objects here
		self.space = tk.Label(self, text = '    ')
		self.space.grid(row=0, column=12, rowspan=1)

		self.text1 = tk.Label(self, textvariable = self.notify, borderwidth=2,relief="groove")
		self.text1.grid(row=1, column=2)

		self.text2 = tk.Label(self, text = 'Modified Image', borderwidth=2, relief="groove")
		self.text2.grid(row=1, column=20, columnspan=6)

		self.space = tk.Label(self, text = ' =~> ')
		self.space.grid(row=1, column=12, columnspan=2)

		self.space = tk.Label(self, text = '    ')
		self.space.grid(row=2, column=12, rowspan=2)

		self.space = tk.Label(self, text = '   ')
		self.space.grid(row=15, column=12, rowspan=2)

		self.btn1 = tk.Button(self, text='Much Better', width=20, pady=5,
			command=self.feedback1, borderwidth=2,relief="raised", bg='green')
		self.btn1.grid(row=17, column=25 )

		self.btn2 = tk.Button(self, text='Slightly Better', width=20, pady=5,
			command=self.feedback2,borderwidth=2,relief='raised',bg='#9EF703')
		self.btn2.grid(row=18, column=25 )

		self.btn3 = tk. Button(self, text='Same', width=20, pady=5, 
			command=self.feedback3, borderwidth=2, relief="raised")
		self.btn3.grid(row=19, column=25)

		self.btn4 = tk.Button(self, text='Slightly Worse', width=20, pady=5, 
			command=self.feedback4, borderwidth=2, relief="raised", bg='#E7720C')
		self.btn4.grid(row=20, column=25 )

		self.btn5 = tk.Button(self, text='Much Worse', width=20, pady=5, 
			command=self.feedback5, borderwidth=2, relief="raised", bg='red')
		self.btn5.grid(row=21, column=25 )


		self.btn6 = tk.Button(self, text='Seelct Image', width=17, pady=5,
			command=self.select, borderwidth=2, relief='raised', bg='#20bebe', fg='white')
		self.btn6.grid(row=17, column=1)

		self.train_v_test = tk.Button(self, textvariable=self.perform, width=17, pady=5,
			command=self.session, borderwidth=2, relief='raised', bg='#20bebe', fg='white')
		self.train_v_test.grid(row=18, column=1)

		self.btn7 = tk.Button(self, textvariable=self.policy_sel, width=17, pady=5,
			command=self.use_model, borderwidth=2, relief='raised', bg='#20bebe', fg='white')
		self.btn7.grid(row=19, column=1)

		self.pbtn = tk.Button(self, text='show performance', width=17, pady=5,
			command=self.show_performance,borderwidth=2, relief='raised', bg='#20bebe', fg='white')
		self.pbtn.grid(row=20, column=1)

		self.btn8 = tk.Button(self, text='Quit', width=17, pady=5,
			command=self.leave, borderwidth=2, relief='raised', bg='red', fg='white')
		self.btn8.grid(row=21, column=1)

		self.btn9 = tk.Button(self, text='Satisfied', width=17, pady=5,
			command=self.satisfy, borderwidth=2, relief='raised', bg='green', fg='white')
		self.btn9.grid(row=22, column=1)

		self.btn9 = tk.Button(self, text='Save', width=17, pady=5,
			command=self.save, borderwidth=2, relief='raised', bg='green', fg='white')
		self.btn9.grid(row=22, column=25)
		

	def display_images(self, imd=None, imt=None):
		"""displays two images side by sid"""
		# im1 = [ImageTk.PhotoImage(Image.open(imd_name)), ImageTk.PhotoImage(Image.open(imt_name))]
		self.imdL = tk.Label(self, image=imd,borderwidth=4, relief="sunken")
		self.imdL.grid(row=4, column=0, rowspan=11, columnspan=12)

		self.imtL = tk.Label(self, image=imt,borderwidth=4, relief="sunken")
		self.imtL.grid(row=4, column=14, rowspan=11, columnspan=12)

	def session(self):
		if self.mode:
			self.perform.set("Go to training")
			self.mode = False
		else:
			self.perform.set("Go to test")
			self.mode = True
	def use_model(self):
		if self.mod:
			fname = filedialog.askopenfilename(initialdir="/home/panther/Desktop/", title="Select file",
				filetypes=(("csv files", "*.csv"), ("all files","*.*")))
			gl.q_table = pd.read_csv(fname, header=0)
			print(gl.q_table)
			self.mod = False
			self.policy_sel.set("Save Q-table")
		else:
			gl.q_table.to_csv('model.csv', index=False)
			self.mod = True
			self.policy_sel.set("Use pre-trained model")


	
	def select(self):
		self.imsel = True
		self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
			filetypes=(("jpeg files", "*.jpg"), ("all files","*.*")))

	def feedback1(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 5

	def feedback2(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 4
		
	def feedback3(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 3
		
	def feedback4(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 2
		#gui.destroy()

	def feedback5(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 1
	def satisfy(self):
		"""Go to the next image"""
		self.flag = True
		self.rate = 4
		self.skip = True
	def save(self):
		self.flag = True
		self.rate = 4
		self.log = True

	def leave(self):
		if tkMessageBox.askyesno("Quit", "Are you sure you want to quit?"):
			self.destroy()

	def show_performance(self):
		plt.clf()
		plt.subplot(221)
		plt.title(f"Reward Histogram")
		plt.hist(self.rates,5,histtype='bar', alpha=0.5,facecolor='b')

		plt.subplot(222)
		plt.plot(range(self.itr), self.rpr, color='r', label='R/P')
		plt.legend()

		plt.subplot(223)
		plt.plot(range(self.itr),self.running_ave, color='g', label='RA')
		plt.legend()
		plt.subplot(224)
		plt.plot(range(self.itr), self.total_ave, label='TA')
		plt.legend(loc='best')
		plt.show(block=False)
		#self.update()


if __name__ == '__main__':
	import cv2
	import time

	gui = GUI()
	# gui.mainloop()
	im = cv2.imread('lena.jpg') #read an image here
	#convert to PIL image format
	imc = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(imc)#reverse im_np = np.asarray(im_pil)

	#imd = Image.open('lena.png') 
	#resized = im_pil.resize((400,300), Image.ANTIALIAS)
	im = ImageTk.PhotoImage(im_pil) #(imd)
	im1 = [im,im]

	for i in range(50):
		gui.display_images(im1[0],im1[1])
		time.sleep(0.1)
		gui.update()
	
