import time
#import timeit
import numpy as np
import RPi.GPIO as GPIO


Tr = 11 # Pin number of input terminal of ultrasonic module
Ec = 8 # Pin number of output terminal of ultrasonic module
GPIO.setmode(GPIO.BCM)
GPIO.setup(Tr, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Ec, GPIO.IN)


def get_dist(min_dist=0.02, max_dist=0.5, half_sound_speed=171.5, repeat=3, omit=2, repeat_time=0.1, return_sem=False):
	"""
	return distance in meter
	"""
	res = []
	for ii in range(repeat+omit):
		GPIO.output(Tr, GPIO.HIGH) # Set the input end of the module to high level and emit an initial sound wave
		time.sleep(0.00001)
		GPIO.output(Tr, GPIO.LOW)
		
		while not GPIO.input(Ec): pass # When the module no longer receives the initial sound wave
		t1 = time.time() # Note the time when the initial sound wave is emitted
		while GPIO.input(Ec): pass # When the module receives the return sound wave
		t2 = time.time() # Note the time when the return sound wave is captured
		
		dt = t2-t1
		dist = dt*half_sound_speed # Calculate distance
		if ii>=omit and dist>=min_dist and dist<=max_dist:
			res.append(dist)
		if repeat_time-dt>0:
			time.sleep(repeat_time-dt)
	if len(res)==0:
		dist = np.nan
		dist_sem = np.nan
	elif len(res)==1:
		dist = res[0]
		dist_sem = np.nan
	else:
		dist = sum(res)/len(res)
		if return_sem:
			dist_sem = np.std(res, ddof=0) / np.sqrt(len(res))
	if return_sem:
		return dist, dist_sem
	else:
		return dist


if __name__=='__main__':
	dists = []
	for i in range(10):
		dist = get_dist()
		print(dist)
		dists.append(dist)
		time.sleep(1)
	
	import matplotlib.pyplot as plt
	plt.plot(dists)
	plt.show()
    
