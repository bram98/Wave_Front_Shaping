
#import sys
#from random import randint
#from graphics import *
import numpy as np
#import datetime
from glob import glob
import os

import matplotlib.pyplot as plt
#import tqdm


#files = glob("phaseplots\phase*.npy")
files = glob("enhancementdata/*.npy")
files.sort(key=os.path.getmtime)
strfiles = []

for fle in files:
    strfiles.append(str(fle))
#phase = np.load('phaseplots\phase_data590159de.npy' )
print(strfiles[-1])
phase = np.load(strfiles[-1])

colorlist = np.linspace(0, 255, len(phase), dtype='uint8')

X = np.linspace(0, 255, len(phase), dtype='uint8')
plt.close()
plt.plot(X, phase)
plt.show()