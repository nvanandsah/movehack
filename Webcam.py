import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
import json
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
options = {
    'model': 'cfg/traffic.cfg',
    'load': 'bin/traffic.weights',
    'threshold': 0.40504555,
    'gpu': 1.0
}


tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
writer = None


#for webcam         capture = cv2.VideoCapture(0)
#for video file     capture = cv2.VideoCapture(ayz.mp4)

capture = cv2.VideoCapture("http://207.192.232.2:8000/mjpg/video.mjpg")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
videowriter = None
while True:
     stime = time.time()
     ret, frame = capture.read()
     if ret:
	results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
	    confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100,2)
	    xxx = '{:.0f}'.format(confidence * 100,2)
	    xxx = int(xxx)
	    frame = cv2.rectangle(frame, tl, br, color, 2)
	    frame = cv2.putText(
            	    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            if (xxx>60):
		crop_img = frame[(tl[1]):(tl[0]),(br[1]):(br[0])]
		cv2.imwrite("images-by-timeframe/"+label+str(datetime.datetime.now())+".jpg", crop_img) 
	cv2.imshow('frame', frame)
	print(results)
        f = open('DATA/kishan.txt', 'a')
        if (results!=[]):
		f.write("%s\n"%results)
                f.write("[{'frame':'"+str(datetime.datetime.now())+ "'}]")
        f.close()
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
cv2.destroyAllWindows()
capture.release()
writer.release()
