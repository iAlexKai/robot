import RPi.GPIO as GPIO
import time

Light = 36
Lifter_Button = 38
Lifter_Up = 40
lifting_time = 10


def init_GPIO_and_lift_up():
    GPIO.setmode(GPIO.BOARD);
    
    GPIO.setup(Lifter_Button, GPIO.OUT)
    GPIO.setup(Lifter_Up, GPIO.OUT)

    GPIO.output(Lifter_Button, GPIO.HIGH)
    GPIO.output(Lifter_Up, GPIO.HIGH)
    
    time.sleep(1)
    print("Finish initializing")

def lifter_up():
    print("1")
    GPIO.output(Lifter_Button,GPIO.LOW)
 #   GPIO.output(Lifter_Up,GPIO.HIGH)
    time.sleep(lifting_time)
    GPIO.output(Lifter_Button,GPIO.HIGH)
    GPIO.output(Lifter_Up,GPIO.HIGH)
    print(2)

#    GPIO.output(Lifter_Up,GPIO.LOW)
#    time.sleep(3)
#    GPIO.output(Lifter_Up,GPIO.HIGH)
#    time.sleep(1)
#    GPIO.output(Lifter_Up,GPIO.LOW)
#    time.sleep(3)
#    GPIO.output(Lifter_Up,GPIO.HIGH)
#    print("3")

def lifter_close():
    print(3)
    start_time = time.time()
    GPIO.output(Lifter_Up, GPIO.LOW)
    time.sleep(lifting_time)
    print(4)
    GPIO.output(Lifter_Up,GPIO.HIGH)
    GPIO.output(Lifter_Button,GPIO.HIGH)



def light():
    GPIO.setmode(GPIO.BOARD);
    GPIO.setup(Light, GPIO.OUT)
    start_time = time.time()
    while (time.time() - start_time < 5):
        GPIO.output(Light, GPIO.LOW)
    GPIO.output(Light, GPIO.HIGH)
        



if __name__ == "__main__":
    init_GPIO_and_lift_up()
#    lifter_up()
#    time.sleep(5)
    lifter_close()   
    light()


