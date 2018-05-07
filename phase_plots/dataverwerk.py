
#import sys
#from random import randint
#from graphics import *
import numpy as np
#import datetime

import matplotlib.pyplot as plt
#import tqdm



phase = np.load('phaseplots\phase_dataeb5fdeae.npy' )

colorlist = np.linspace(0, 255, len(phase), dtype='uint8')

X = np.linspace(0, 255, len(phase), dtype='uint8')
plt.close()
plt.plot(X, phase)
plt.show()