from typing import Tuple, List,Dict, Callable
import cv2
import time
import numpy as np
from random import choices
import matplotlib.pyplot as plt 
from PIL import ImageTk, Image
from apppub import GUI
import gLobal as gl

EPISODES = 50
L = 256
# http://www.dip.ee.uct.ac.za/imageproc/stdimages/greyscale/
class QL(object):
	"""Modelling Qlearning for contrast enhancement"""
	def __init__(self,states, actions, step_size: float=0.1,
			gamma: float=0.9,e_greedy: float=0.1) -> None:
		self.states: int = states
		self.actions: List = actions
		self.alpha: float = step_size #learning rate
		self.gamma: float = gamma
		self.epsilon: float = e_greedy
		self.tou: float =  None
		self.iter: int = 0
		
		gl.initialize(self.states, self.actions)

		#gl.q_table.to_csv('model.csv', index=False)
		
	def learn(self, s, a, r, s_n) -> None:
		"""Implements Q-learning"""
		q_sa = gl.q_table.iloc[s,a]
		q_target = r + self.gamma*np.max(gl.q_table.iloc[s_n, :])

		#update Q table i.e. q(s,a)
		gl.q_table.iloc[s,a] += self.alpha*(q_target - q_sa)

	def select_action(self, state: int, tou_0: float=1) -> int:
		"""selects an action using softmax action selection"""

		self.iter += 1
		self.tou = tou_0/np.sqrt(self.iter) # t = t0/sqrt(iter)
		state_action = gl.q_table.iloc[state,:]
		#selection an action greedly
		action = int(np.random.choice(state_action[state_action==
					np.max(state_action)].index))

		if np.random.random() < self.epsilon:
			"""explore new action using boltzmann distribution
			ignoring the best action"""
			#keep track of the favorite action's index
			index = self.actions.index(action) 
			count = 0
			prob = []
			for n in state_action.values:
				if count == index:
					count += 1
					continue
				prob.append(np.exp(n/self.tou))
				count += 1
			#remove the best action
			choose_action = self.actions[:]
			prob = prob/np.sum(prob)
			choose_action.remove(action)
			action = choices(choose_action,weights=prob)[0]
			#print(action)

		return action

		
class RCA(object):
	"""Modelling Reinforced contrast Adaptation"""
	def __init__(self, gm = 2):
		self.height: int = 0
		self.width: int =0
		self.imd = None # modified or deteriorated gray image
		self.imt = None # transformed image
		self.pdf: List = None
		self.phi_0z: List = None
		self.phi_zL: List = None
		self.phi_ = lambda b1,b2: np.sum(self.pdf[b1:b2+1])
		#transformation
		self.tf = (lambda phi_l, phi_r: (phi_l*phi_r)**gm *(phi_l**gm + 
					phi_r**gm)**(-2))

	def modify_image(self, gray_im):
		"""returns a modiefied image (g')"""

		new_min, new_max = np.random.choice(range(256), 2, replace=False)
		if new_min > new_max:
			new_min, new_max = new_max, new_min

		g_min, g_max = np.min(gray_im), np.max(gray_im)
		self.height, self.width = gray_im.shape 
		print(f"row: {self.height}, col: {self.width}, new_min: {new_min}, new_max: {new_max}")
		mod_im = np.zeros((self.height, self.width))
		for i in range(self.height):
			for j in range(self.width):
				mod_im[i,j] = (new_min + 
					round((gray_im[i][j] - g_min)*(new_max - new_min)/(g_max - g_min)))

		self.imd =  np.uint8(mod_im)

	def find_state(self, gray_im):
		"""determine the state of the image"""

		# find g_min, g_max
		g_min, g_max = np.min(gray_im), np.max(gray_im)

		#cv2.calcHist(im, channel, mask, histsize(BINS), range)
		hist = cv2.calcHist([gray_im], [0], None, [256],
					[0,256])

		#calculate gh_max
		gh_max = int(np.where(hist == np.max(hist))[0])
		#find the state of the image
		#plot the histogram
		# plt.plot(hist, color='b')
		# plt.title('Histogram of gray image')
		# plt.show()

		if self.state(g_max) == 0:
			status = self.state(gh_max)
		elif self.state(g_min) == 0:
			if self.state(g_max) == 1:
				status = 2 + self.state(gh_max)
			elif self.state(g_max) == 2:
				status = 5
		elif self.state(g_min) == 1:
			if self.state(g_max) == 1:
				status = 5 + self.state(gh_max)
			elif self.state(g_max) == 2:
				status = 8
		elif self.state(g_min) == 2:
			status = 9

		return status

	def state(self, val):
		if val < (L-1)/3:
			st = 0 # low
		elif val < 2*(L-1)/3:
			st = 1 #medium
		else:
			st = 2 #high
		return st
	def calc_hist(self):
		hist = np.zeros((1,256))
		hist = hist[0]
		for i in range(self.height):
			for j in range(self.width):
				hist[self.imd[i,j]] += 1
		return hist

	def ret_px_xpx(self, pdf, low, high):
	
		px = np.sum(pdf[low:high+1])
		xpx = 0

		for i in range(low, high+1):
			xpx +=  (i/255)*pdf[i]

		return px,xpx

	def calc_kbar(self, b):
		ybar = 0.5
		hist = self.calc_hist()/(self.height*self.width)
		b = 255*b
		t = round(b)
		b1,a1 = self.ret_px_xpx(hist, t, 255)
		_,c1 = self.ret_px_xpx(hist, 0, t)
		_,e1 = self.ret_px_xpx(hist, 0, 255)

		num = (1-b)*ybar - a1 + b*b1
		den = c1 + b*b1 - b*e1
		kbar = b*(num/den)
		#print(f"kbar: {kbar}")
		return kbar


	def transform_image(self, a):
		"""Receives the deteriorated image and modifies it"""
		if a == 0:
			do: Callable = self.func1(1/40)
		elif a == 1:
			do: Callable = self.func2(1/40)
		elif a == 2:
			do: Callable = self.sigm_ref
		elif a == 3:
			do: Callable = self.sigmoid
		elif a == 4:
			x_min, x_max = self.imd.min(), self.imd.max()
			do: Callable = self.lin_stretch(x_max, x_min)
		elif a == 5:
			do: Callable = self.adaptive_one()
		elif a == 6:
			do: Callable = self.adaptive_improved()
		else:
			print("<!> Error: Unsupported action")

		# transform the image according to each action
		mod_im = np.zeros((self.height, self.width))
		for i in range(self.height):
			for j in range(self.width):
				mod_im[i,j] = round(do(self.imd[i,j]))
		self.imt = np.uint8(mod_im)

		 

	def sigmoid(self,x): #for action 4
		"""x: intensity of a pixel divided by 255"""
		if x > 0 and x < 254:
			return 255*(1/ (1 + np.exp(-5*(2*(x/255) - 1))))
		return x

	def sigm_ref(self,x): #for action 3
		if x > 1 and x < 254:
			return 255*(1 - np.log(255/x -1)/5)/2
		return x
			

	def func1(self,a): # for action 1😁👌
		b = 255/np.log(a*255 + 1)
		return (lambda x: b*np.log(a*x + 1))

	def func2(self,a): #for action 2
		b = 255/np.log(a*255 + 1)
		return (lambda x: (np.exp(x/b) -1)/a)

	def lin_stretch(self, x_min, x_max):
		return (lambda x: 255*(x - x_min)/(x_max - x_min)) 

	def adaptive_one(self):
		b,_ = cv2.threshold(self.imd, 0, 255, cv2.THRESH_BINARY +
				cv2.THRESH_OTSU,)
		b = b/255
		x_min, x_max = self.imd.min(),self.imd.max()
		mu = 0.2

		kbar = self.calc_kbar(b)
		k_low = (1- mu)*b
		k_high = mu + k_low

		if kbar > k_high:
			k = k_high
		elif kbar <= k_high and kbar>= k_low:
			k = kbar
		else:
			k = k_low
		#print(f"k: {k}")
		
		return (lambda x: 0 if x <= x_min 
			else (((k/(b - x_min/255))*(x - x_min)) if x > x_min and x <= b 
				else((255*(k + ((1-k)/(x_max/255 - b))*(x/255 - b))) 
					if x > b and x < x_max else 1) ))


	def adaptive_improved(self):
		self.pdf = self.calc_hist()/(self.height*self.width)
		self.phi_0z = [self.phi_(0,0)]
		for z in range(1,256):
			self.phi_0z.append(self.phi_0z[z-1] + self.phi_(z,z))
		#self.phi_0z = [self.phi_(0,z) for z in range(0,256)]
		temp = self.phi_0z[255]
		self.phi_zL = [temp]
		for z in range(1,256):
			self.phi_zL.append(temp - self.phi_0z[z-1])
		#self.phi_zL = [self.phi_(z,255) for z in range(0, 256)]

		k = np.sum([self.tf(self.phi_0z[z],self.phi_zL[z]) 
			for z in range(0,256)])**(-1)
		return (lambda x: 255*k*np.sum([self.tf(self.phi_0z[z], self.phi_zL[z]) 
				for z in range(0, int(round(x)))]) )


class Statistics(object):
	"""Keeps track of rates(MOS), +ve and -ve rewards,
	last N rewards and total average"""
	def __init__(self, N = 10):
		self.rates: List[int] = []
		self.reward: List[float] = []
		self.punishment: List[float] = [] #punishments
		self.last_n: List[float] = []
		self.rpr: List[float] = [] # history of reward to punishment ratio
		self.running_ave: List[float] = []
		self.total_ave: List[float] = []
		self._N = N 


def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0: 
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass



	
	

if __name__ == '__main__':
	actions = list(range(7))
	ql = QL(10, actions)
	rca = RCA()
	gui = GUI()
	#gui = Statistics()
	#print(rca.func2(255/255))
	count = 0
	last_n = []
	 
	#load all the images
	color = 0
	im_set = []
	im_set.append(cv2.imread("images/cameraman.tif",color))
	im_set.append(cv2.imread("images/woman_blonde.tif", color))
	im_set.append(cv2.imread("images/pirate.tif", color))
	im_set.append(cv2.imread("images/mandril_gray.tif", color))
	im_set.append(cv2.imread("images/lena_color_512.tif", color))
	im_set.append(cv2.imread("images/walkbridge.tif", color))
	im_set.append(cv2.imread("images/livingroom.tif", color))
	im_set.append(cv2.imread("images/download (4).jpg", color))
	im_set.append(cv2.imread("images/OIP (2).jpg", color))

	im_set.append(cv2.imread("images/peppers_gray.tif", color))
	im_set.append(cv2.imread("images/OIP (4).jpg", color))
	im_set.append(cv2.imread("images/OIP (5).jpg", color))
	im_set.append(cv2.imread("images/OIP (6).jpg", color))
	

	im_set = np.array(im_set, dtype=object) # change to numpy array
	test_index = np.random.choice(range(len(im_set)),3, replace=False)
	#choose images for training
	training_index = list(range(len(im_set)))

	for i in test_index:
		training_index.remove(i) 

	#test_index = t #[t,np.array([4])] #lena and one of the first four images
	#training_index = np.array(training_index)
	test_index = np.append(test_index, 0)
	print(test_index)
	def show_results():

		im_d = cv2.cvtColor(rca.imd, cv2.COLOR_BGR2RGB)
		im_dp = Image.fromarray(im_d) #pillow image format
		im_t = cv2.cvtColor(rca.imt, cv2.COLOR_BGR2RGB)
		im_tp = Image.fromarray(im_t) #pillow image format

		# h,w = rca.imd.shape
		# if h > 512 or w > 512:
		im_dp = im_dp.resize((400,400), Image.ANTIALIAS)
		im_tp = im_tp.resize((400,400), Image.ANTIALIAS)
		# else:
		# 	im_dp = im_dp.resize((h,w), Image.ANTIALIAS)
		# 	im_tp = im_tp.resize((h,w), Image.ANTIALIAS)
		imd,imt = ImageTk.PhotoImage(im_dp), ImageTk.PhotoImage(im_tp)
		
		#waiting for feedback from the user
		while not gui.flag:
			gui.display_images(imd, imt)
			time.sleep(0.1)
			gui.update()
	
	for itr in range(EPISODES):
		gui.flag = False #button waiting

		#1. Select an original image from a set of images
		idx =  np.random.choice(training_index,1)[0]	
		print("index: "+str(idx))

		#2. Modify the original image (deteriorating)
		rca.modify_image(im_set[idx])
		
		# = cv2.calcHist([im_set[idx][0]], [0], None, [256],
		#			[0,256])
		# plt.plot(hist, color='g')
		# plt.title('original image histogram')
		# plt.show()

		#3. determine state of the deteriorated image (S)
		s = rca.find_state(rca.imd)

		#4. transform the deteriorated image using the selected action
		a = ql.select_action(s) #select an action
		print(f'action: {a}')
		rca.transform_image(a)

		#5. determine state of the transformed image (St+1)
		s_n = rca.find_state(rca.imt)

		#6. get reward from the user(comparing imd and imt)

		#...Show the user the modified image and transformed image...
		show_results()
			
		#map the rates [1,2,..,5] to [-0.4, -0.2, 0 , 0.2,0.4]
		gui.rates.append(gui.rate)
		r = - 0.4 + (gui.rate - 1)/5 
		if r > 0:
			gui.reward.append(r)
		elif r < 0: 
			gui.punishment.append(r)
		if count < 10:
			last_n.append(r)
			count += 1
		else:
			last_n.pop(0)
			last_n.append(r)
		print("last n: ",last_n)
		gui.running_ave.append(np.sum(last_n)/gui._N)
		gui.rpr.append(len(gui.reward)/(len(gui.punishment) + 0.05))
		gui.total_ave.append((np.sum(gui.reward) +
			np.sum(gui.punishment)) / (itr + 1))
		#6. update Q-table
		ql.learn(s,a,r,s_n)
		gui.itr += 1
		#............go to test
		if not gui.mode:
			break

	#test the algorithm
	print("Testing or validating the algorithim...")

	gui.notify.set("Original Image")
	for idx in test_index:
		gui.flag = False
		if not gui.imsel:
			print("index: " + str([idx]))
			rca.imd = np.uint8(im_set[idx])
		else:
			gui.imsel = False
			rca.imd = np.uint8(cv2.imread(gui.filename, color))
			#print("filename: ", gui.filename)
		rca.height, rca.width = rca.imd.shape
		s = rca.find_state(rca.imd)
		state_action = gl.q_table.iloc[s,:]

		#select an action according to the Q-policy
		a = np.random.choice(state_action[state_action==
					np.max(state_action)].index)
		#a = ql.select_action(s)
		rca.transform_image(int(a))
		show_results()
		if gui.mode:
			...

	print("\n*********************end*********************")
	
	
	print("\n<Q-table>:")
	print(gl.q_table)



