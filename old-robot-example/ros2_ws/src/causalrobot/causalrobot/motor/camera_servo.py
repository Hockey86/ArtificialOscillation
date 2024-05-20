import Adafruit_PCA9685 # Import the library used to communicate with PCA9685
import time


class CameraServo:
    def __init__(self, servo_id=0):
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(50)
        self.servo_id = servo_id
        self.min_pos = 560
        self.max_pos = 100
        self.neutral_pos = (self.min_pos+self.max_pos)//2
        self.min_angle = -90
        self.max_angle = 90
        self.neutral_angle = (self.min_angle+self.max_angle)/2.
        self.min_allowed_angle = -15
        self.max_allowed_angle = 30

        # move to init angle
        self.pwm.set_pwm(self.servo_id, 0, self.neutral_pos)
        self.pos = self.neutral_pos

    def move_to_angle(self, angle, speed='slow'):
        """
        angle is in unit of degree
        speed is in unit of degree/second
        """
        assert angle<=self.max_allowed_angle and angle>=self.min_allowed_angle
        assert speed in ['slow', 'fast']
        if speed=='slow':
            incr = 2
        elif speed=='fast':
            incr = 4

        pos = self.angle2pos(angle)
        if pos>self.pos:
            pos_range = range(self.pos+1,pos,incr)
        elif pos<self.pos:
            pos_range = range(self.pos-1,pos,-incr)
        else:
            pos_range = []
        for x in pos_range:
            self.pwm.set_pwm(self.servo_id, 0, x)
            time.sleep(0.01)
        self.pwm.set_pwm(self.servo_id, 0, pos)
        self.pos = pos

    def pos2angle(self, pos):
        angle = (pos-self.min_pos)/(self.max_pos-self.min_pos)*(self.max_angle-self.min_angle)+self.min_angle
        return angle

    def angle2pos(self, angle):
        pos = (angle-self.min_angle)/(self.max_angle-self.min_angle)*(self.max_pos-self.min_pos)+self.min_pos
        pos = int(round(pos))
        return pos


if __name__=='__main__':
    print('Create servo...', end='', flush=True)
    cs = CameraServo()
    print('OK', flush=True)

    print('Move servo...', end='', flush=True)
    cs.move_to_angle(30)
    cs.move_to_angle(-15)
    time.sleep(2)
    cs.move_to_angle(30, speed='fast')
    cs.move_to_angle(-15, speed='fast')
    time.sleep(2)
    print('OK', flush=True)

    print('Move servo to neutral...', end='', flush=True)
    cs.move_to_angle(0)
    print('OK', flush=True)

