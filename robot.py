from detector import loop_and_detect
from utils.camera import open_cam_usb, Camera_for_Robot
from utils.visualization import open_window, show_fps, record_time, show_runtime
from utils.engine import BBoxVisualization
from utils.yolo_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3

import numpy as np
import RPi.GPIO
import serial
import cv2
import sys
import time

# The following flag ise used to control whether to use a GStreamer
# pipeline to open USB webcam source.  If set to False, we just open
# the webcam using cv2.VideoCapture(index) machinery. i.e. relying
# on cv2's built-in function to capture images from the webcam.
USB_GSTREAMER = True

# global variables for peripherals and GPIO
Buzzer_Pin = 0
LED_RED = 1
LED_GREEN = 2
Lifter_Button = 27
Lifter_Up = 28
Lifter_Down = 29
FOUND_CARS_PORT = 40

# global variables for video
VIDEO_ID = 0
DETECT_ID = 0
SAVE_VIDEO = 0

# to be modified global variables
TOTAL_RUN_TIME = 30
DETECT_TIME = 120
StraghtValue = 130

# current x value for control direction
currentX = 130
GEAR = 60

# variables for frame cut
cutWidth = 0
cutHeight = 300

# car direction states
directionStates = ("left", "right", "straight", "stop")
curCarState = "straight"

# open the serial
serial_agent = serial.Serial("/dev/ttyUSB0", 9600)


def set_config():
    global TOTAL_RUN_TIME
    global DETECT_TIME
    global StraghtValue
    global currentX

    if len(sys.argv) == 1:
        currentX = StraghtValue
    else:
        assert len(sys.argv) == 4
        TOTAL_RUN_TIME = int(sys.argv[1])
        DETECT_TIME = int(sys.argv[2])
        StraghtValue = int(sys.argv[3])
        currentX = StraghtValue

    return StraghtValue


def init_GPIO_and_lift_up():
    GPIO.setmode(GPIO.BOARD);
    GPIO.setup(LED_RED, GPIO.OUT);
    GPIO.setup(LED_GREEN, GPIO.OUT);
    GPIO.setup(Buzzer_Pin, GPIO.OUT);
    GPIO.setup(Lifter_Button, GPIO.OUT);
    GPIO.setup(Lifter_Up, GPIO.OUT);
    GPIO.setup(Lifter_Down, GPIO.OUT);

    GPIO.output(Buzzer_Pin, GPIO.HIGH);
    GPIO.output(LED_RED, GPIO.HIGH);
    GPIO.output(LED_GREEN, GPIO.HIGH);
    GPIO.output(Lifter_Button, GPIO.HIGH);
    GPIO.output(Lifter_Up, GPIO.HIGH);
    GPIO.output(Lifter_Down, GPIO.HIGH);

    GPIO.setup(FOUND_CARS_PORT, GPIO.IN);
    time.sleep(15)


def lifter_up():
    pass


def lifter_down():
    pass


def go_straight():
    global currentX
    currentX = StraghtValue
    outputCommand = bytes("$AP0:{}X254Y127A127B!".format(str(currentX)), encoding='utf8')
    serial_agent.write(outputCommand)


# the larger the X, the left it turns
def turn_left():
    global currentX
    if currentX < StraghtValue:
        currentX = StraghtValue + GEAR
    elif currentX < StraghtValue + GEAR * 1.5:
        currentX = currentX + GEAR
    print("Turning Left with value {} larger than straight".format(currentX - StraghtValue))

    outputCommand =  bytes("$AP0:{}X254Y127A127B!".format(str(currentX)), encoding='utf-8')
    serial_agent.write(outputCommand)


# the smaller the X, the right it turns
def turn_right():
    global currentX
    if currentX > StraghtValue:
        currentX = StraghtValue - GEAR
    elif currentX > StraghtValue - GEAR * 1.5:
        currentX = currentX - GEAR

    print("Turning right with value {} less than straight".format(currentX - StraghtValue))

    outputCommand =  bytes("$AP0:{}X254Y127A127B!".format(str(currentX)), encoding='utf-8')
    serial_agent.write(outputCommand)


def stop_car():
    outputCommand = bytes("$AP0:127X127Y127A127B!", encoding='utf8')
    for i in range(3):
        serial_agent.write(outputCommand)
        time.sleep(0.05)
    print("The car should be stopped and start to detect motion")


def get_edges(img, blur_size, canny_lth, canny_hth):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 1)
    edges = cv2.Canny(blur, canny_lth, canny_hth)
    return edges


def roi_mask(img, corner_points):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, corner_points, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color, thickness=1):
    # import pdb
    # pdb.set_trace()
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_transform(img, rho, theta, threshold, min_line_len, max_line_gap):
    # 统计概率霍夫直线变换
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    # 新建一副空白画布
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(drawing, lines, color=[0, 0, 255])     # 画出直线检测结果
    return drawing, lines


def filter_lanes(lines):

    left_lines, right_lines = [], []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    if len(left_lines) == 0 or len(right_lines) == 0:
        return None, None

    left_lane = right_lane = None
    down_most_y = -1
    for line in left_lines:
        if line[0][0] <= 30 and down_most_y < line[0][1]:
            left_lane = line
            down_most_y = line[0][1]

    down_most_y = -1
    for line in right_lines:
        if line[0][2] >= 600 and down_most_y < line[0][3]:
            right_lane = line
            down_most_y = line[0][3]

    return left_lane, right_lane


def search_line(img, ):
    # variables for gassian and canny
    blur_size = 5
    canny_lth = 30
    canny_hth = 45

    # variables for hough
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 20

    edges = get_edges(img=img, blur_size=blur_size, canny_lth=canny_lth, canny_hth=canny_hth)

#    cv2.imshow("edges", edges)
#    cv2.waitKey(0)

    rows, cols = edges.shape
    # points = np.array([[(0, rows), (15, 250), (630, 250), (cols, rows)]])
    points = np.array([[(0, rows), (0, cutHeight), (cols, cutHeight), (cols, rows)]])
    roi_edges = roi_mask(edges, points)
#    cv2.imshow("roi_edges", roi_edges)
#    cv2.waitKey(0)

    drawing, lines = hough_transform(img=roi_edges, rho=rho, theta=theta,
                                     threshold=threshold, min_line_len=min_line_len, max_line_gap=max_line_gap)
    
    left_lane, right_lane = filter_lanes(lines)
    diffFromCentre = 0

    if left_lane is not None and right_lane is not None:
        draw_lines(drawing, [left_lane, right_lane], color=[0, 0, 255])  # 画出直线检测结果
        leftK = (left_lane[0][1] - left_lane[0][3]) / (left_lane[0][0] - left_lane[0][2])
        leftB = left_lane[0][1] - leftK * left_lane[0][0]
        rightK = (right_lane[0][1] - right_lane[0][3]) / (right_lane[0][0] - right_lane[0][2])
        rightB = right_lane[0][1] - rightK * right_lane[0][0]
        # print(leftK, leftB, rightK, rightB)

        middleLineLeftX = (cutHeight - leftB) / leftK
        middleLineRightX = (cutHeight - rightB) / rightK
        middleX = (middleLineLeftX + middleLineRightX) / 2
        diffFromCentre = cols / 2 - middleX
    else:
        print("At least one lane not detected")
    cv2.imshow("lines", drawing)
    cv2.waitKey(1)

    return left_lane is not None and right_lane is not None, diffFromCentre


def run_straight():
    capture_road = open_cam_usb(VIDEO_ID, width=640, height=480)
    if not capture_road.isOpened():
        print("Capture road opens failure")
        return
    if SAVE_VIDEO == 1:
        outputPath = "./run_log/1.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoOut = cv2.VideoWriter(outputPath, fourcc, 20.0, (640, 480))

    global curCarState
    # capture_road = cv2.imread('imgs/1.jpg')
    # time initalize
    startTime = int(time.time())

    while int(time.time()) - startTime < TOTAL_RUN_TIME:
        ret, frame = capture_road.read()
        if not ret:
            print("Capture road reads failure")
            return

        # frame = capture_road

        # diff threshold for turn
        frameWidth = frame.shape[0]
        turnValue = frameWidth

        lineFound, diffFromCentre = search_line(frame)
       
        if not lineFound:
            go_straight()
            print("Some lane not found, keep straight")
            if curCarState is not "straight":
                curCarState = "straight"
                assert curCarState in directionStates
        elif diffFromCentre > frameWidth or diffFromCentre < -frameWidth:
            go_straight()
            print("inValid state, stay straight")
            if curCarState is not "straight":
                curCarState = "straight"
                assert curCarState in directionStates
        elif diffFromCentre > turnValue:
            turn_left()
            if curCarState is not "left":
                curCarState = "left"
                assert curCarState in directionStates
        elif diffFromCentre < -turnValue:
            turn_right()
            if curCarState is not "right":
                curCarState = "right"
                assert curCarState in directionStates
        else:
            go_straight()
            print("Stay staight")
            if curCarState is not "straight":
                curCarState = "straight"
                assert curCarState in directionStates

    stop_car()
    capture_road.release()


def detect_and_alarm():
    WINDOW_NAME = "Robot_YOLOv3_Detector_with_TensorRT"
    MODEL = "yolov3-tiny-416"
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    RUN_TIME = False
    DETECT_TIME = 30

    cam = Camera_for_Robot(video_dev=DETECT_ID, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
    cam.open()

    if not cam.is_opened:
        print("Capture road opens failure")
        return
    cam.start()

    open_window(WINDOW_NAME, IMAGE_WIDTH, IMAGE_HEIGHT,
                'TensorRT YOLOv3 Detector')
    cls_dict = get_cls_dict('coco')
    yolo_dim = int(MODEL.split('-')[-1])  # 416 or 608
    trt_yolov3 = TrtYOLOv3(model=MODEL, input_shape=(yolo_dim, yolo_dim))
    vis = BBoxVisualization(cls_dict)
    
    loop_and_detect(cam, RUN_TIME, trt_yolov3, conf_th=0.3, vis=vis, window_name=WINDOW_NAME, total_time=DETECT_TIME)
    cam.stop()
    cam.release()
    cv2.destroyAllWindows()
    print("Detection finishes successfully")

def stop_and_close():
    serial_agent.close()
    pass


def main():
    #currentX = set_config()
    # init_GPIO_and_lift_up()

    #run_straight()
    detect_and_alarm()
    #stop_and_close()


if __name__ == "__main__":
    main()
