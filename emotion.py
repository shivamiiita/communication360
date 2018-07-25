
# coding: utf-8

# In[2]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import cv2
import numpy as np
from PIL import Image
import helper
import asyncio
# import matplotlib.pyplot as plt
# %matplotlib inline 


_key = '7a1f3b0255754754a0bd19e017c4f489'

headers = dict()
headers['Ocp-Apim-Subscription-Key'] = _key
headers['Content-Type'] = 'application/octet-stream'

json = None
params = None


# In[3]:

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Display frame    
    cv2.imshow('webcam', frame)
    
    # save frame to jpg
    img = Image.fromarray(frame)
    img.save('demo.jpg')
    
    # read back frame
    with open('demo.jpg', 'rb' ) as f:
        data = f.read()
    
    # send demo image to emotion api  
    result = helper.processRequest(json, data, headers, params)
    print(result)

    # display result
    if result is not None:
        # Load the original image from disk
        data8uint = np.fromstring(data, np.uint8 ) # Convert string to an unsigned int array
        img = cv2.cvtColor(cv2.imdecode( data8uint, cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
        print(img)
        helper.renderResultOnImage(result, img)
        frame2 = img
        img = Image.fromarray(frame2) 
        img.save('demo_conversion.jpg') 
        out.write(frame2)        
        cv2.imshow('emotion', frame2)
        count += 1
        print("count " + str(count))
    if count == 500:
        break
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
    # Stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

