# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:30:06 2018

@author: pbaxter
"""

#import sys
#from random import randint
#from graphics import *
import time
import numpy as np

import pypylon
import slmpy 
import matplotlib.pyplot as plt
import uuid
#import tqdm

import winsound

#CAMERA
available_cameras = pypylon.factory.find_devices()
cam = pypylon.factory.create_device(available_cameras[-1])# Grep the first one and create a camera for it
#print('Camera info of camera object:', cam.device_info)# We can still get information of the camera back
cam.open()# Open camera and grep some images
for key in cam.properties.keys():
    try:
        value = cam.properties[key]
    except IOError:
        value = '<NOT READABLE>'
    
height, width = cam.properties['Height'], cam.properties['Width']
total_pixels_camera = (height*width) # number of pixels of the camera    
cam.properties['DefectPixelCorrectionMode']='Off'
cam.properties['ContrastEnhancement']=0.0
cam.properties['GainAuto']='Off' 
cam.properties['Gain'] = 0
cam.properties['ExposureAuto']='Off'
cam.properties['BlackLevel']=0.0
cam.properties['ExposureTime']=6./60*10**6 #1/60 s in us = 16667, 816652.0


#SLM
slm = slmpy.SLMdisplay(isImageLock = True)
resY, resX = slm.getSize()
#print '%i %i'%(resX,resY)
#print resX,resY
#print cam.properties['Width']
#print cam.properties['Height']


#%% Define active area of SLM.
active_radius = 300 #250 # 
opt_size = 10 #optimization area = (2*opt_size)^2

# Define the WFS configuration (no. of 'superpixels') for pre-optimization
xsp = 5 #number of superpixels in x direction
ysp = xsp #number of superpixels in y direction
xssp = (resX/xsp) #number of pixels in one superpixel in the x direction
yssp = (resY/ysp) #number of pixels in one superpixel in the y direction

# Number of pixels for optimization
xsp2 = 10   #number of superpixels in x direction
ysp2 = xsp2 #number of superpixels in y direction
xssp2 = (resX/xsp2) #number of pixels in one superpixel in the x direction
yssp2 = (resY/ysp2) #number of pixels in one superpixel in the y direction

sleep_time=.1
#colours ( 0 = black, 255 = white )
#Allard said to start with 5
# Set the greyvalues for which we measure:
#colorlist = np.arange(0,256,102,dtype='uint8') 
#colorlist = [0,51,102,153,204,255]#,102,153,204,255]
colorlist = [0,100,200]

# Images for SLM
testIMG = np.zeros((resX,resY),dtype=np.uint8) # SLM test pattern
bestIMG = np.zeros((resX,resY),dtype=np.uint8) # SLM pattern after pre-opt
optIMG = np.zeros((resX,resY),dtype=np.uint8) # SLM pattern after optimization

#save intensity for each SuperPixel to plot them
intensitiesSP = np.zeros((xsp,ysp,len(colorlist)),dtype='float64')#array of xsp x ysp x 9 (number of colours now)
intensitybefore, intensityafter = 0.0, 0.0

#use this to start the program with the last saved result
use_preload = True

def av_intensity(times=1): #1600x1200, READ CAMERA (pixel in the center)
    global opt_size
    plt.close('all')
    ccd1=list(cam.grab_images(1))
    ccd_feedback=sum(sum(ccd1[0][480-opt_size*times:480+opt_size*times,640-opt_size*times:640+opt_size*times]*1.0))/(times*times)
    if np.max(ccd1[0][480-opt_size:480+opt_size,640-opt_size:640+opt_size])>253:
        print 'At least one pixel in saturation!'
    return ccd_feedback

def inradius(x,y,r):
    if np.sqrt((x-resX/2+20)**2 + (y-resY/2+40)**2)<r:
        return 1
    else:
        return 0
    
def cc(): #I'm tired of typing cam.close()
    cam.close()
    
def newDrawRec(x,y,dx,dy,img,color): #change pixels on img and return 2D-array
    img2=img*1 # Forces new memory allocation
    img2[x:x+dx,y:y+dy]=color
    return img2

def write_array(img, array): #writes array to SLM per superpixel
    xsp, ysp = np.shape(array)
    xssp = (resX/xsp) #number of pixels in one superpixel in the x direction
    yssp = (resY/ysp) #number of pixels in one superpixel in the y direction
    
    for j in range(ysp):
        for i in range(xsp):
            if inradius(i*xssp+xsp/2,j*yssp+yssp/2,active_radius):
                img = newDrawRec(i*xssp,j*yssp,xssp,yssp,img,array[i,j])
    slm.updateArray(img)
    return img
    
def random_pattern(): #initialises SLM to have random values
    global bestIMG
    rand_array = np.random.randint(256, size=(50, 50)).astype('uint8')
    bestIMG = write_array(bestIMG, rand_array)
    
def pre_opt():#pre-optimization    
    global testIMG
    global bestIMG
    global intensitiesSP
    for j in range(ysp):
        for i in range(xsp):
            phase=[]
            if inradius(i*xssp+xssp/2,j*yssp+yssp/2,active_radius):
                for k in range(len(colorlist)):
                    testIMG = newDrawRec(i*xssp,j*yssp,xssp,yssp,testIMG,colorlist[k])
                    slm.updateArray(testIMG)
                    time.sleep(sleep_time) #sleep
                    intensity=av_intensity()
                    phase.append(intensity)
                    intensitiesSP[i,j,k]=intensity
                g = phase.index(max(phase))
                bestPH=colorlist[g]
                    
                bestIMG = newDrawRec(i*xssp,j*yssp,xssp,yssp,bestIMG,bestPH)
                slm.updateArray(bestIMG)
                testIMG = bestIMG
        
def opt():#optimization    
    global testIMG
    global bestIMG
    global optIMG
    global intensitiesSP
    for j in range(ysp2):
        for i in range(xsp2):
            phase=[]
            if inradius(i*xssp2+xssp2/2,j*yssp2+yssp2/2,active_radius):
                for k in range(len(colorlist)):
                    testIMG = newDrawRec(i*xssp2,j*yssp2,xssp2,yssp2,bestIMG,colorlist[k])
                    slm.updateArray(testIMG)
                    time.sleep(sleep_time) #sleep 
                    intensity=av_intensity()
                    phase.append(intensity)
                g = phase.index(max(phase))
                bestPH=colorlist[g]
                # Now we update our high resolution optimization pattern:
                optIMG = newDrawRec(i*xssp2,j*yssp2,xssp2,yssp2,optIMG,bestPH)
                slm.updateArray(bestIMG)

#%% Gradient method
                
def grad_step_init():
    global curX
    global newX
    global deltaX
    global xlog
    global ylog
    
    for j in range(ysp2):
        for i in range(xsp2):
            if inradius(i*xssp2+xssp2/2,j*yssp2+yssp2/2,active_radius):
    
    
    newX = np.uint8(newX + .5)
    for i in range(dimensie-1):
        y0[i] = Y[curX[i]]
    np.append(xlog, newX)
    np.append(ylog, Y[newX])
    for i in range(dimensie-1):
        y1[i] = Y[np.uint8(newX[i])]
        deltaX[i]= learningRate*(y1[i] + y0[i])*np.sign(deltaX[i])
    newX = curX + deltaX
    #curX = np.uint8(newX + .5)
    xlog = np.append(xlog, curX)
    ylog = np.append(ylog, Y[curX])
    
def grad_opt():
    global testIMG
    global bestIMG
    global optIMG
    global intensitiesSP
   
#%% options (pre)opt with(out) output
                
def pre_opt_with_output():
    intensitybefore = av_intensity(20) # Measure initial intensity
    before = time.time()
    pre_opt()#pre-optimization
    deltaT = time.time() - before
    
    intensityafter = av_intensity() #check intensity in focus after 
    #
    ratio = 1.0*intensityafter/intensitybefore
    print 'Pre-optimization finished'
    print('Ratio between intensity before and intensity after is %f \n') %ratio
    print('delta T is %f minutes\n')%(deltaT/60.)

def opt_with_output(soortopt = opt()):
    global colorlist
    print 'Starting optimization'
    
    intensitybefore = av_intensity(20) 
    before = time.time()
    colorlist = [0,51,102,153,204,255]#,102,153,204,255]
    soortopt
    slm.updateArray(optIMG)
    
    time.sleep(sleep_time)
    print 'Optimization finished'
    deltaT_opt = time.time() - before
    print('Opt. time %f minutes\n')%(deltaT_opt/60.)
    
    intensityafter2 = av_intensity() #check intensity in focus after 
    ratio = 1.0*intensityafter2/intensitybefore
    print('Ratio between intensity before and intensity after is %f \n') %ratio
    
def blink():
    show_many = 500
    sleep = 2.
    for n in range(show_many):
        slm.updateArray(bestIMG*0)
        time.sleep(sleep)
    #    slm.updateArray(bestIMG)
    #    time.sleep(sleep)
        slm.updateArray(optIMG)
        time.sleep(sleep)

#%% Actual program:
# Take image of initial pattern:
for Initial_Pattern in cam.grab_images(1):
    None

#either use a preloaded image or run a full optimization
if use_preload:
    optIMG = plt.imread("FrankEnBram/opt_preload/current_preload.bmp", format="bmp" )
    bestIMG = optIMG
    slm.updateArray(optIMG)
    time.sleep(sleep_time)
else:
    random_pattern()

    pre_opt_with_output()
    
    opt_with_output()

#save current optIMG so you can use it next time
plt.imsave("FrankEnBram/opt_preload/current_preload.bmp", optIMG)

#blink()
#%% Make Phase-Intensity Plot to identify the nessecery number of phases to test.
amountofphases=255
colorlist = np.linspace(0, 255, amountofphases, dtype='uint8')
phase=[]
xSamplePoint, ySamplePoint = int(xsp/2)-3, int(ysp/2)-3
for k in range(len(colorlist)):
    testIMG = newDrawRec(xSamplePoint*xssp2,ySamplePoint*yssp2,xssp2,yssp2,bestIMG,colorlist[k])
    slm.updateArray(testIMG)
    time.sleep(sleep_time) #sleep 
    intensity=av_intensity(1)
    phase.append(intensity)
slm.updateArray(bestIMG)

np.save('FrankEnBram\phase_data' + str(uuid.uuid4())[:8], phase)

#plt.plot(subphase)
#plt.show()

cam.close()
winsound.Beep(523, 100)
winsound.Beep(466, 150)
winsound.Beep(622, 300)

slm.updateArray(optIMG)
