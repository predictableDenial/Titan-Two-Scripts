import os
import cv2
import math
from gtuner import *
import gtuner
import numpy as np



#ADD MULTI PROCESSOR SUPPORT FOR DETECTED OBJECT LOOPS

ON_FRAME = 1 #SEND DATA EVERY x FRAME, 1 SENDS EVERY FRAME, 2 SENDS EVERY 2nd FRAME, ETC.
SHOW_FRAME_COUNT = True #Show frame count on screen or not

#COLORS FOR TEXT ON SCREEN
BBOX_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 255, 0)

#MODEL LOCATIONS AND SETTINGS
WEIGHT_LOCATION = "C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/yolov3.weights"
CFG_LOCATION = "C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/yolov3.cfg"
CLASSES_FILE = "yolov3.txt" #full path or path relative to working dir is fine
MODEL_SIZE = 608 #Lower will run faster, higher will be more accurate. Multiples of 32 only
MODEL_SCALE = 1/256 #idk
#REMEMBER TO SET OR NOT SET FP16 FOR CUDA TARGET ON LINE 40


#calculates distance of center screen to bbox based on angle
def yx_dist(x1, y1, x2,  y2):
	return  ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5

class GCVWorker:
	def __init__(self, width, height):
		os.chdir(os.path.dirname(__file__))
		
		self.frameCount = 0
		self.prevFrameDist = 0
		
		#read pre-trained model and config file
		self.net = cv2.dnn.readNet(WEIGHT_LOCATION, CFG_LOCATION)
		
		#set cuda backend
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
		
		#read class names from classes text file
		self.classes = None
		with open(CLASSES_FILE, 'r') as f:
			self.classes = [line.strip() for line in f.readlines()]
		
		#set opencv detection model
		self.model = cv2.dnn_DetectionModel(self.net)
		self.model.setInputParams(size=(MODEL_SIZE, MODEL_SIZE), scale=MODEL_SCALE)
	
	def __del__(self):
		del self.prevFrameDist
		del self.frameCount
		del self.net
		del self.classes
		del self.model
	
	def process(self, frame):
		gcvdata = None
		gcvdata = bytearray([0xFF])
		gcvdata[0] = 0
		
		self.frameCount += 1
		centerX = 0
		centerY = 0
		degs = 0
		dist = 0
		
		classes, scores, boxes = self.model.detect(frame, 0.5, 0.4)
		
		num_detections = len(boxes)
		lowestDist = 9999
		for iteration, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
			if self.classes[classid[0]] == 'person':
				checkDist = int((box[0] + (box[0] + box[2])) / 2) - 960
				if abs(checkDist) < lowestDist:
					lowestDist = abs(checkDist)
					centerX = (box[0] + (box[0] + box[2])) / 2
					centerY = (box[1] + (box[1] + (box[3] / 2.5))) / 2
					x = box[0]
					y = box[1]
					w = box[2]
					h = box[3]
				#if get_val(BUTTON_8):
					#cv2.putText(frame, "AIMBOT", (900, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.5, COLOR, 2)
					#if self.frameCount % 213 == 0:
						#cv2.imwrite('C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/playerImages/' + str(self.frameCount) + '.png', frame)
				label = "%s : %f" % ('player', score)
				cv2.rectangle(frame, box, BBOX_COLOR, 2)
				cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 2)
		
		if centerX != 0 and centerY != 0:
			if centerX + (w / 2) >= 960 and centerX - (w / 2) <= 960 and centerY + (h / 1.5) >= 540 and centerY - (h / 4.5) <= 540:
				cv2.putText(frame, "IN BBOX", (870, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) #Shows when aim is inside the bounding box
				gcvdata[0] = 1
			cv2.rectangle(frame, (960,540), (960,540), (255,255,255), 5)
			cv2.rectangle(frame, (int(centerX),int(centerY)), (int(centerX),int(centerY)), BBOX_COLOR, 10)
			px = int(centerX)
			py = int(centerY)
			dy = 540 - py
			dx = px - 960
			cv2.line(frame, (960, 540), (px, py), (255, 255, 255), 2)
			rads = math.atan2(dy,dx)
			degs = math.degrees(rads)
			degs = degs * -1
			cv2.putText(frame, "ANGLE: " + str(int(degs)) + " deg", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 2)
			dist = yx_dist(960, 540, px, py)
			cv2.putText(frame, "DISTANCE: " + str(int(dist)) + " pix", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 2)
			
		if SHOW_FRAME_COUNT:
			cv2.putText(frame, "FRAME: " + str(self.frameCount), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 2)
		
		if self.frameCount % ON_FRAME == 0:
			gcvdata.extend(int(degs*0x10000).to_bytes(4, byteorder='big', signed=True))
			gcvdata.extend(int(dist*0x10000).to_bytes(4, byteorder='big', signed=True))
			gcvdata.extend(int(centerX*0x10000).to_bytes(4, byteorder='big', signed=True))
			gcvdata.extend(int(centerY*0x10000).to_bytes(4, byteorder='big', signed=True))
		
		del centerX
		del centerY
		del degs
		del dist
		
		return (frame, gcvdata)
