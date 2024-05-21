#Install scipy in CMD
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

#Import Numpy
#Install Numpy in CMD
import numpy as np

#Install Imutills in CMD
import imutils

#Import OpenCV
import cv2

#Midpoint variable initialization
# Determine the midpoint of the object to be measured
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


#Enable Camera to Display Video in Realtime
cap = cv2.VideoCapture(0)


#Creating Conditions
#If the camera is active and the video has started, then run the program below
while (cap.read()):
        ref,frame = cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        orig = frame[:1080,0:1920]
       
        #Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)

        result_img = closing.copy()
        contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        count_object = 0

        
        #Convert Pixel Readout Value Into CM Units
        pixelsPerMetric = None

        #Create Looping Conditions
        #Initialize Variable cnt = counturs
        for cnt in contours:

            #Pembacaan Area Objek yang di Ukur
            area = cv2.contourArea(cnt)

            #If Area Less than 1000 and More than 12000 Pixels
            #Then Take Measurements
            if area < 1000 or area > 120000:
                continue

            #Calculates the bounding box of the Object's contours
            orig = frame.copy()
            box = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

            
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

            
            (xl, x2, y1, y2) = box
            (a1, a2) = midpoint(xl, x2)
            (b1, b2) = midpoint(y2, y1)
            (c1, c2) = midpoint(xl, y2)
            (d1, d2) = midpoint(x2, y1)

            #Draw the midpoint on the object
            cv2.circle(orig, (int(a1), int(a2)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(b1), int(b2)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(c1), int(c2)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(d1), int(d2)), 0, (0, 255, 64), 5)

            #Draw a line at the midpoint
            cv2.line(orig, (int(a1), int(a2)), (int(b1), int(b2)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(c1), int(c2)), (int(d1), int(d2)),
                    (255, 0, 255), 2)

            #Calculates the Euclidean distance between midpoints
            wide_pixel = dist.euclidean((a1, a2), (b1, b2))
            long_pixel = dist.euclidean((c1, c2), (d1, d2))

            #If pixelsPerMetric pixels have not been initialized, then
            #Calculate as the ratio of pixels to the provided metric
            #In this case CM
            if pixelsPerMetric is None:
                pixelsPerMetric = wide_pixel
                pixelsPerMetric = long_pixel
            wide = wide_pixel
            long = long_pixel

#Describes the size of objects in the image
            cv2.putText(orig, "Breadth: {:.1f}CM".format(wide_pixel/25.5),(int(d1 + 10), int(d2)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            cv2.putText(orig, "Length: {:.1f}CM".format(long_pixel/25.5),(int(a1 - 15), int(a2 - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            #cv2.putText(orig,str(area),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
            count_object+=1

        #Displays the number of detected objects
        cv2.putText(orig, "No. of objects: {}".format(count_object),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  
        cv2.imshow('Camera',orig)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()