import os
import cv2
import math
from gtuner import *
import gtuner
import time
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta

def runThread(t):
	print(f"running..")
	time.sleep(t)
	return 'done'


def hideSensitiveInfo(frame):
	cv2.rectangle(frame, (1680,1060), (1680+200,1060+10), (255,255,255), 10)
	cv2.rectangle(frame, (1540,100), (1540+320,100+50), (255,255,255), 60)
	cv2.rectangle(frame, (70,950), (70+130,950+60), (255,255,255), 80)

def yx_dist(x1, y1, x2,  y2):
	return  ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5

COLOR = (0, 0, 255)
hCOLOR = (0, 255, 255)

class PrimaryWeapon:
	name = [1330, 162]
	height = 20
	width = 43

class Attachment:
	slot1 = [1663, 301]
	height = 22
	width = 22

class GCVWorker:
	def __init__(self, width, height):
		os.chdir(os.path.dirname(__file__))
		self.gcvdata = bytearray([0xFF, 0xFF, 0xFF])
		
		#############ALL GUN DETECCTION STUFF
		self.inventoryIsOpen = False
		self.guns = ['M416', 'AKM', 'Beryl', 'G36C', 'AUG', 'Groza', 'QBZ', 'DP-28', 'M249', 'MG3', 'Vector', 'UMP45', 'Tommy', 'Bizon', 'MP5', 'VSS', 'Mk14', 'SCAR', 'M16', 'Mini', 'SKS', 'SLR', 'QBU']
		self.gunImages = []
		for gun in self.guns:
			self.gunImages.append(cv2.imread(gun + '.png'))
		self.scope2x = cv2.imread('2x.png')
		self.scope3x = cv2.imread('3x.png')
		self.scope4x = cv2.imread('4x.png')
		self.scope6x = cv2.imread('6x.png')
		self.inventory = cv2.imread('inventory.png')
		self.equippedWeapon = 0
		self.equippedScope = 0
		self.gunName = 'NONE'
		
		self.prevFrame = []
		self.prevSeconds = 0
		self.totalSeconds = 0
		self.newFrames = 0
		self.totalFrames = 0
		self.frameRate = 0
		
		##########
		self.frameCount = 0
		self.prevFrameDist = 0
		
		#read pre-trained model and config file
		self.net = cv2.dnn.readNet("C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/yolov4.weights", "C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/yolov4.cfg")
		
		#set cuda backend
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
		
		#read class names from classes text file
		self.classes = None
		with open("yolov4.txt", 'r') as f:
			self.classes = [line.strip() for line in f.readlines()]
		
		#set opencv detection model
		self.model = cv2.dnn_DetectionModel(self.net)
		self.model.setInputParams(size=(608, 608), scale=1/256)
	
	def __del__(self):
		del self.gcvdata
		
		del self.prevFrame
		del self.prevSeconds
		del self.totalSeconds
		del self.newFrames
		del self.totalFrames
		del self.frameRate
	
	def process(self, frame):
		gccvdata = None
		gcvdata = bytearray([0xFF, 0xFF, 0xFF])
		gcvdata[0] = self.gcvdata[0]
		gcvdata[1] = self.gcvdata[1]
		
		#take images for object training
		#if get_val(gtuner.BUTTON_8) and get_val(gtuner.BUTTON_5):
			#cv2.putText(frame, str(self.frameCount), (900, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.5, COLOR, 2)
			#if self.frameCount % 31 == 0:
				#cv2.putText(frame, str(self.frameCount), (900, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.5, COLOR, 2)
				#cv2.imwrite('C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/playerImages/' + str(self.frameCount) + '.png', frame)
		
		#cv2.imwrite('C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/inventory.png', frame)
		##########ALL GUN DETECTION STUFF
		#y, y, x, x
		
		inventoryX = 515
		inventoryY = 100
		inventorySizeX = 111
		inventorySizeY = 32
		inventoryCoords = [inventoryY, inventoryY + inventorySizeY, inventoryX, inventoryX + inventorySizeX]
		#capture = frame[inventoryCoords[0]:inventoryCoords[1], inventoryCoords[2]:inventoryCoords[3]]
		#cv2.imwrite('C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/inventory.png', capture)
		cv2.rectangle(frame, (inventoryCoords[2], inventoryCoords[0]), (inventoryCoords[3], inventoryCoords[1]), COLOR, 2)
		if cv2.norm(frame[inventoryCoords[0]:inventoryCoords[1], inventoryCoords[2]:inventoryCoords[3]], self.inventory[0:inventorySizeY, 0:inventorySizeX]) < 6666:#6666:
			self.inventoryIsOpen = True
		else:
			self.inventoryIsOpen = False
		if self.inventoryIsOpen:
			attach = Attachment()
			weap = PrimaryWeapon()
			weapon = frame[weap.name[1]:weap.name[1]+weap.height, weap.name[0]:weap.name[0]+weap.width]
			attachSlot1 = frame[attach.slot1[1]:attach.slot1[1]+attach.height, attach.slot1[0]:attach.slot1[0]+attach.width]
			
			#cv2.imwrite('C:/Users/m_bot/anaconda3/envs/gpu/Scripts/Titan Two/PUBG/QBU.png', weapon)
			#cv2.rectangle(frame, (attach.slot1[0], attach.slot1[1]), (attach.slot1[0]+attach.width, attach.slot1[1]+attach.height), COLOR, 2)
			
			x = 0
			self.gunName = 'NONE'
			self.equippedWeapon = x
			for gun in self.gunImages:
				x += 1
				if cv2.norm(weapon, self.gunImages[x-1][0:0+weap.height, 0:0+weap.width]) < 2222:
					self.gunName = self.guns[x - 1]
					self.equippedWeapon = x
			del x
			if cv2.norm(attachSlot1, self.scope2x[0:0+attach.height, 0:0+attach.width]) < 1600:
				self.equippedScope = 2
			elif cv2.norm(attachSlot1, self.scope3x[0:0+attach.height, 0:0+attach.width]) < 1600:
				self.equippedScope = 3
			elif cv2.norm(attachSlot1, self.scope4x[0:0+attach.height, 0:0+attach.width]) < 1600:
				self.equippedScope = 4
			elif cv2.norm(attachSlot1, self.scope6x[0:0+attach.height, 0:0+attach.width]) < 900:
				self.equippedScope = 6
			else:
				self.equippedScope = 1
			cv2.rectangle(frame, (900,40), (900+100,40+10), (255,255,255), 40)
			self.gcvdata[0] = self.equippedWeapon
			self.gcvdata[1] = self.equippedScope
			del attachSlot1
			del weapon
			del attach
			del weap
		hideSensitiveInfo(frame)
		frame = cv2.putText(frame, "Inventory Open: " + str(self.inventoryIsOpen), (750, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		frame = cv2.putText(frame, "Equipped Weapon: " + str(self.gunName), (1515, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		#frame = cv2.putText(frame, "Weapon Array #: " + str(self.equippedWeapon), (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		frame = cv2.putText(frame, "Equipped Scope: " + str(self.equippedScope), (1515, 152), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		#####################################
		#########FRRAME RATE
		#chheck if 1 second has elapsed
		now = datetime.now()
		seconds = int(now.strftime("%S"))
		if seconds != self.prevSeconds:
			self.totalSeconds += 1
			self.frameRate = self.totalFrames
			self.totalFrames = 0
		cv2.putText(frame, "ELAPSED " + str(self.totalSeconds), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 2)
		self.prevSeconds = seconds
		
		#compare current frame and previous frame
		newFrame = frame[810:1110, 490:590]
		if not np.array_equal(newFrame,self.prevFrame):
			self.totalFrames += 1
		
		
		cv2.putText(frame, "FRAMERATE: " + str(self.frameRate), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
		
		self.prevFrame = frame[810:1110, 490:590]
		###############################################
		#########PERSON DETECTION STUFF
		#print(__name__)
		if self.frameCount == 30:
			print(__name__)
			if __name__ == '__main__':
				with concurrent.futures.ProcessPoolExecutor() as executor:
					t1 = executor.submit(runThread, 2)
					print(t1.result())
		
		self.frameCount += 1
		
		x = 0
		y = 0
		w = 0
		h = 0
		centerX = 0
		centerY = 0
		
		#classes, scores, boxes = self.model.detect(frame, 0.5, 0.4)
		
		#num_detections = len(boxes)
		lowestDist = 9999
		#for iteration, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
			#if self.classes[classid[0]] == 'player':
				#checkDist = int((box[0] + (box[0] + box[2])) / 2) - 960
				#if abs(checkDist) < lowestDist:
					#lowestDist = abs(checkDist)
					#centerX = (box[0] + (box[0] + box[2])) / 2
					#centerY = (box[1] + (box[1] + (box[3] / 1.5))) / 2
					#x = box[0]
					#y = box[1]
					#w = box[2]
					#h = box[3]
				#label = "%s : %f" % (self.classes[classid[0]], score)
				#cv2.rectangle(frame, box, COLOR, 2)
				#cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
			#if self.classes[classid[0]] == 'head':
				#label = "%s : %f" % (self.classes[classid[0]], score)
				#cv2.rectangle(frame, box, hCOLOR, 2)
				#cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hCOLOR, 2)
		
		#TESTING
		if centerX != 0 and centerY != 0:
			if centerX + (w / 2) >= 960 and centerX - (w / 2) <= 960 and centerY + (h / 1.5) >= 540 and centerY - (h / 4.5) <= 540:
				cv2.putText(frame, "IN", (870, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
				gcvdata[2] = 1
			else:
				gcvdata[2] = 0
			cv2.rectangle(frame, (960,540), (960,540), (255,255,255), 5)
			px = int(centerX)
			py = int(centerY)
			dy = 540 - py
			dx = px - 960
			
			cv2.line(frame, (960, 540), (px, py), (255, 255, 255), 2)
			
			rads = math.atan2(dy,dx)
			degs = math.degrees(rads)
			degs = degs * -1
			cv2.putText(frame, str(int(degs)) + " deg", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR, 2)
			dist = yx_dist(960, 540, px, py)
			cv2.putText(frame, str(int(dist)) + " pix", (500, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR, 2)
			
			#cv2.putText(frame, str(self.frameCount), (500, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR, 2)
			if self.frameCount % 2 == 0:
				gcvdata.extend(int(degs*0x10000).to_bytes(4, byteorder='big', signed=True))
				gcvdata.extend(int(dist*0x10000).to_bytes(4, byteorder='big', signed=True))
				gcvdata.extend(int(centerX*0x10000).to_bytes(4, byteorder='big', signed=True))
				gcvdata.extend(int(centerY*0x10000).to_bytes(4, byteorder='big', signed=True))
			else:
				gcvdata.extend(int(0*0x10000).to_bytes(4, byteorder='big', signed=True))
				gcvdata.extend(int(0*0x10000).to_bytes(4, byteorder='big', signed=True))
				gcvdata.extend(int(centerX*0x10000).to_bytes(4, byteorder='big', signed=True))
				gcvdata.extend(int(centerY*0x10000).to_bytes(4, byteorder='big', signed=True))
			self.prevFrameDist = dist
		else:
			gcvdata[2] = 0
			gcvdata.extend(int(0*0x10000).to_bytes(4, byteorder='big', signed=True))
			gcvdata.extend(int(0*0x10000).to_bytes(4, byteorder='big', signed=True))
			gcvdata.extend(int(centerX*0x10000).to_bytes(4, byteorder='big', signed=True))
			gcvdata.extend(int(centerY*0x10000).to_bytes(4, byteorder='big', signed=True))
		
		
		del x
		del y
		del w
		del h
		del centerX
		del centerY
		
		return (frame, gcvdata)
