

'''
    0         pan left(pan++)           300   320         pan right(pan--)               640
  0 +------------------------------------+-----+-----+------------------------------------+
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
    |                                    |     |     |          tilt up(tilt--)           | 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
225 +------------------------------------+-----+-----+------------------------------------+ 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    |  
240 +------------------------------------+-----+-----+------------------------------------+ 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    |  
255 +------------------------------------+-----+-----+------------------------------------+ 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
    |                                    |     |     |          tilt down(tilt++)         | 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    |  
480 +------------------------------------+-----+-----+------------------------------------+ 
'''
import cv2
import numpy as np
import serial
import time
margin_x = 40
margin_y = 70
x = y = 0; w = h = 1;
sp  = serial.Serial('COM3', 115200, timeout=0.125)


cam = cv2.VideoCapture(1)       # 2nd camera
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cam.set(cv2.CAP_PROP_FRAME_FPS, 10)


def up():
    sp.write(b'w')
def down():
    sp.write(b's')   
def left():
    sp.write(b'a')
def right():
    sp.write(b'd')


if not cam.isOpened():
    print("Could not open camera")
    exit()


while cam.isOpened():
    time.sleep(0.01)
    status, frame = cam.read() #  read camera frame
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # Convert from BGR to HSV


    # define range of blue color in HSV
    lower_blue = np.array([100,100,120])          # range of blue
    upper_blue = np.array([150,255,255])


    lower_green = np.array([50, 150, 50])        # range of green
    upper_green = np.array([80, 255, 255])


    lower_red = np.array([150, 50, 50])        # range of red
    upper_red = np.array([180, 255, 255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)     # color range of blue
    mask1 = cv2.inRange(hsv, lower_green, upper_green)  # color range of green
    mask2 = cv2.inRange(hsv, lower_red, upper_red)      # color range of red


    # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(frame, frame, mask=mask)      # apply blue mask
    res = cv2.bitwise_and(frame, frame, mask=mask1)    # apply green mask
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)    # apply red mask
    
    gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)    
    _, bin = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    largest_area = 0    
    
    COLOR = (0, 255, 0)
    for cnt in contours:                # find largest blue object
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_contour = cnt
            
     # draw bounding box with green line
    if largest_contour is not None:
        #area = cv2.contourArea(cnt)
        if largest_area > 768:  # draw only larger than 500
            x, y, width, height = cv2.boundingRect(largest_contour)
            center_x = x + width//2
            center_y = y + height//2
            print("center: ( %s, %s )"%(center_x, center_y))
            '''---------------------------------------------------'''
            if center_x <= 320-margin_x:### need pan left ###########
                left()
            elif center_x > 320+margin_x: ### need pan right ########
                right()
            else:
                pass
            '''---------------------------------------------------'''
            if center_y <= 240-margin_y:### need tilt up ##########
                up()
            elif center_y > 240+margin_x: ### need tilt down ##########
                down()
            else: ########################### no need move tilt
                pass
            cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR, 2)
            time.sleep(0.05)   
    cv2.imshow("VideoFrame",frame)       # show original frame


    k = cv2.waitKey(5) & 0xFF
        
    if k == 27:
        break
   
        
cam.release()
cv2.destroyAllWindows()
