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

#http://207.192.232.2:8000/mjpg/video.mjpg
#http://46.186.121.222:83/GetData.cgi?CH=1
#http://200.54.42.126:8090/mjpg/video.mjpg
#http://88.2.219.238:9000/mjpg/2/video.mjpg
#http://88.2.219.238:9000/mjpg/4/video.mjpg
#http://210.148.107.78/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000
#http://91.214.20.212:80/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER
#http://153.156.230.207:8081/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000
#http://122.219.76.26:60001/SnapshotJPEG?Resolution=640x480&amp;Quality=Clarity&amp;COUNTER
#http://209.12.71.138/mjpg/video.mjpg
#http://209.12.71.132:80/mjpg/video.mjpg
#http://190.29.99.134:80/mjpg/video.mjpg
#http://138.118.33.201:80/mjpg/video.mjpg
#http://192.168.43.1:8080/videofeed
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
	    #xxx = '{:.0f}'.format(confidence * 100,2)
	    #xxx = int(xxx)
	    frame = cv2.rectangle(frame, tl, br, color, 2)
	    frame = cv2.putText(
            	    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            if (xxx>60):
		crop_img = frame[(tl[1]):(tl[0]),(br[1]):(br[0])]
		cv2.imwrite("images-by-timeframe/"+label+str(datetime.datetime.now())+".jpg", crop_img) 
	cv2.imshow('frame', frame)
	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	#fps = round(capture.get(cv2.CAP_PROP_FPS))
        #videoWriter = cv2.VideoWriter(
         #   'video12.avi', fourcc, fps, (418,418))
	#print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print(results)
        f = open('DATA/kishan.txt', 'a')
        #f.write("{")
        #i=-1
        #for jj in results:
         #   i+=1
          #  if i<(len(results)-2):
	if (results!=[]):
		f.write("%s\n"%results)
                f.write("[{'frame':'"+str(datetime.datetime.now())+ "'}]")
        f.close()
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
cv2.destroyAllWindows()
capture.release()
writer.release()
