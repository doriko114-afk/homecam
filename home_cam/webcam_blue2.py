'''
    0         pan left                  300   320         pan right                     640
  0 +------------------------------------+-----+-----+------------------------------------+
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    | 
    |                                    |     |     |          tilt up                   | 
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

margin_x = 160
margin_y = 120

sp = serial.Serial('COM3', 115200, timeout=1)

def up():
   sp.write(b'w')

def down():
   sp.write(b's')

def left():
   sp.write(b'a')

def right():
   sp.write(b'd')

webcam = cv2.VideoCapture(1)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

margin_x = 40
margin_y = 30

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,100,120])
    upper_blue = np.array([150,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
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
        if largest_area > 500:  # draw only larger than 500
            x, y, width, height = cv2.boundingRect(largest_contour)       
            cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR, 2)
            center_x = x + width//2
            center_y = y + height//2            
            if center_x < 320 - margin_x:
                left()
            elif center_x > 320 + margin_x:
                right()
            else:
                pass           
            if center_y < 240 - margin_y:
                up()
            elif center_y > 240 + margin_y:
                down()
            else:
                pass
            print("center: ( %s, %s )"%(center_x, center_y))
            
            time.sleep(0.3)
    cv2.imshow("VideoFrame",frame)       # show original frame
    '''
    cv2.imshow('blue', res)           # show applied blue mask
    cv2.imwrite("blue.png", res)
    cv2.imshow('Green', res1)          # show applied green mask
    cv2.imwrite("green.png", res1)
    cv2.imshow('red', res2)          # show applied red mask
    cv2.imwrite("red.png", res2)
    '''
    k = cv2.waitKey(5) & 0xFF
        
    if k == 27:
        break
   
        
capture.release()
cv2.destroyAllWindows()
