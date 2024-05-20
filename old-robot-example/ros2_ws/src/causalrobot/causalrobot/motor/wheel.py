import RPi.GPIO as GPIO
import time


class WheelMotor:
    def __init__(self, side):
        if side=='left':
            self.Motor_EN    = 4
            self.Motor_Pin1  = 26
            self.Motor_Pin2  = 21
        else:
            self.Motor_EN    = 17
            self.Motor_Pin1  = 27
            self.Motor_Pin2  = 18
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.Motor_EN, GPIO.OUT)
        GPIO.setup(self.Motor_Pin1, GPIO.OUT)
        GPIO.setup(self.Motor_Pin2, GPIO.OUT)

        self.stop_motor()
        self.pwm = GPIO.PWM(self.Motor_EN, 100)

    def stop_motor(self):
        GPIO.output(self.Motor_Pin1, GPIO.LOW)
        GPIO.output(self.Motor_Pin2, GPIO.LOW)
        GPIO.output(self.Motor_EN, GPIO.LOW)

    def run_motor(self, speed):
        if speed == 0: # stop
            self.stop_motor()
        elif 100>=speed>=40:
            GPIO.output(self.Motor_Pin1, GPIO.LOW)
            GPIO.output(self.Motor_Pin2, GPIO.HIGH)
            self.pwm.start(0)
            self.pwm.ChangeDutyCycle(speed)
        elif -100<=speed<=-40:
            GPIO.output(self.Motor_Pin1, GPIO.HIGH)
            GPIO.output(self.Motor_Pin2, GPIO.LOW)
            self.pwm.start(0)
            self.pwm.ChangeDutyCycle(-speed)
        else:
            print('Speed must be between 40 to 100 or 0.')

if __name__=='__main__':
    print('Create wheel...', end='', flush=True)
    whl_l = WheelMotor('left')
    whl_r = WheelMotor('right')
    print('OK', flush=True)
    
    print('Move wheel...', end='', flush=True)
    whl_l.run_motor(40)
    whl_r.run_motor(40)
    time.sleep(3)
    print('OK', flush=True)

    print('Stop wheel...', end='', flush=True)
    whl_l.stop_motor()
    whl_r.stop_motor()
    GPIO.cleanup()  # Release resource
    print('OK', flush=True)
