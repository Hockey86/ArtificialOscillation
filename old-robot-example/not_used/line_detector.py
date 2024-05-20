import RPi.GPIO as GPIO


# Hunt module output pin
line_pin_right = 19
line_pin_middle = 16
line_pin_left = 20


def setup():
	'''
	Initialize your GPIO port related to the line patrol module
	'''
	GPIO.setwarnings(False)
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(line_pin_right,GPIO.IN)
	GPIO.setup(line_pin_middle,GPIO.IN)
	GPIO.setup(line_pin_left,GPIO.IN)


def get_status():
	'''
	Read the values of three infrared sensor phototransistors (0 is no line detected, 1 is line detected)
	This routine takes the black line on white as an example
	'''
	left = GPIO.input(line_pin_left)
	middle = GPIO.input(line_pin_middle)
	right = GPIO.input(line_pin_right)
	return left, middle, right


if __name__=='__main__':
	import time
	setup()
	for i in range(10):
		print(get_status())
		time.sleep(1)
	
