from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from imutils.video import VideoStream
from imutils.video import FPS
import urllib.request
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from os import remove
from os import path
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'videos'
#leer carpeta
lista = []
salida = []
def devolverArchivos(carpeta, lista):
	for archivo in os.listdir(carpeta):
		lista.append(os.path.join(carpeta,archivo))
		if os.path.isdir(os.path.join(carpeta,archivo)):
			devolverArchivos(os.path.join(carpeta,archivo), lista)
def eliminar():
  for eliminar in lista:#eliminar archivos
    print(eliminar)
    if path.exists(eliminar):
      remove(eliminar)
#------------------------------------------------------------>>>>>>>>>>>>>
@app.route("/")
def upload_file():
	return render_template('index.html')


@app.route("/", methods=['POST'])
def uploader(listo = "Archivo subido exitosamente"):
  devolverArchivos("videos/", lista)
  eliminar()
  if request.method == 'POST':
    f = request.files['archivo']
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    devolverArchivos("videos/", lista)
    contar(lista)
  return render_template('index.html', listo=listo)

#----------------------------------------------------------->>>>>>>>>>>>
# variable de el archivo que este dentro de la carpeta
#---------------------------------------------------------->>>>>>>>

def contar(lista):
  CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
  print("[INFO] loading model...")
  net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
  print("[INFO] opening video file...")
  vs = cv2.VideoCapture(f"{lista[0]}")
  writer = None
  W = None
  H = None
  ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
  trackers = []
  trackableObjects = {}
  totalFrames = 0
  totalDown = 0
  totalUp = 0
  fps = FPS().start()
  while True:
	  frame = vs.read()
	  frame = frame[1]
	  if f"{lista[0]}" is not None and frame is None:
		  break
	  frame = imutils.resize(frame, width=500)
	  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	  if W is None or H is None:
		  (H, W) = frame.shape[:2]
	  # if we are supposed to be writing a video to disk, initialize
	  # the writer
	  if "static/output_02.mp4 " is not None and writer is None:
		  fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		  writer = cv2.VideoWriter("static/output_02.mp4", fourcc, 30,
			  (W, H), True)
	  status = "Waiting"
	  rects = []
	  if totalFrames % 30 == 0:
		  status = "Detecting"
		  trackers = []
		  blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		  net.setInput(blob)
		  detections = net.forward()
		  for i in np.arange(0, detections.shape[2]):
			  confidence = detections[0, 0, i, 2]
			  if confidence > 0.4:
				  idx = int(detections[0, 0, i, 1])
				  if CLASSES[idx] != "person":
					  continue
				  box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				  (startX, startY, endX, endY) = box.astype("int")
				  tracker = dlib.correlation_tracker()
				  rect = dlib.rectangle(startX, startY, endX, endY)
				  tracker.start_track(rgb, rect)
				  trackers.append(tracker)
	  else:
		  for tracker in trackers:
			  status = "Tracking"
			  tracker.update(rgb)
			  pos = tracker.get_position()
			  startX = int(pos.left())
			  startY = int(pos.top())
			  endX = int(pos.right())
			  endY = int(pos.bottom())
			  rects.append((startX, startY, endX, endY))
	  cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
	  objects = ct.update(rects)
	  for (objectID, centroid) in objects.items():
		  to = trackableObjects.get(objectID, None)
		  if to is None:
			  to = TrackableObject(objectID, centroid)
		  else:
			  y = [c[1] for c in to.centroids]
			  direction = centroid[1] - np.mean(y)
			  to.centroids.append(centroid)
			  if not to.counted:
				  if direction < 0 and centroid[1] < H // 2:
					  totalUp += 1
					  to.counted = True
				  elif direction > 0 and centroid[1] > H // 2:
					  totalDown += 1
					  to.counted = True
		  trackableObjects[objectID] = to
		  text = "ID {}".format(objectID)
		  cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	  info = [("Up", totalUp),("Down", totalDown),("Status", status),]
	  for (i, (k, v)) in enumerate(info):
		  text = "{}: {}".format(k, v)
		  cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	  if writer is not None:
		  writer.write(frame)
	  key = cv2.waitKey(1) & 0xFF
	  if key == ord("q"):
		  break
	  totalFrames += 1
	  fps.update()
  fps.stop()
  print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
  #url = "/videos/example_01.mp4"
  #file = "ejemplo.mp4"
  #r = urllib.request.urlopen(url)
  #f = open(file, "wb")
  #f.write(r.read())
  #f.close()
#---------------------------------------------------->>>>>
# mostrar resultado
#--------------------------------------------------->>>>>>
if __name__ == '__main__':
 app.run(debug=True)
 