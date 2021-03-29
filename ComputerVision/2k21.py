import os
import cv2
import numpy as np
from gtuner import *
import gtuner

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
        self.gcvdata = bytearray([0xFF])
        self.gfxArrow = cv2.imread('greenArrow.png')
        self.gfxBar = cv2.imread('bar.png')
        self.oldPos = None
        self.Xdist = 0.0
    
    def __del__(self):
        del self.gcvdata
        del self.Xdist
    
    def process(self, frame):
        self.gcvdata = bytearray([0xFF])
        self.gcvdata[0] = 0 # default value, for doing nothing
        src = [0,0, self.gfxArrow.shape[1],self.gfxArrow.shape[0]] # arrow
        
        Amatch = 0
        if self.oldPos is not None:  # earch around last area first
            # xmin, ymin, xmax, ymax
            if self.oldPos[0] >= 50:
                search = [ self.oldPos[0]-40, self.oldPos[1]-20, self.oldPos[0]+40, self.oldPos[1]+20 ]
            else:
                print("IN1")
                search = [20, self.oldPos[1]-30, self.oldPos[0]+50, self.oldPos[1]+30 ]
            #cv2.rectangle(frame, (self.oldPos[0] - 40, self.oldPos[1] - 20), (self.oldPos[0] + 40, self.oldPos[1] + 20), (0, 255, 0), 2)
            Amatch, AbestX, AbestY = mySearchCVtmp(self.gfxArrow, frame, src , search, 0.7, 50, 100, 'Arrow: ', grey=False, mark=False, silent=False)
            self.oldPos = None
        
        cv2.rectangle(frame, (100, 150), (1750, 500), (255, 0, 0), 2)
        if get_val(STICK_1_Y) > 80:
            if Amatch == 0:
                search = [100,150, 1750, 500] #full area
                Amatch, AbestX, AbestY = mySearchCVtmp(self.gfxArrow, frame, src , search, 0.7, 50, 100, 'Arrow: ', grey=False, mark=False, silent=False)
        
        if Amatch:  # only when triangle is found search for bar
            self.oldPos = [AbestX,AbestY] # store old position to speed up next find
            src = [0,0, self.gfxBar.shape[1],self.gfxBar.shape[0]]
            #cv2.rectangle(frame, (AbestX - 60, AbestY + 10), (AbestX + 60, AbestY + 25), (0, 255, 0), 2)
            #bar = frame[AbestY + 10:AbestY + 25, AbestX - 60:AbestX + 60]
            #red_pixels = np.argwhere(cv2.inRange(bar, (0, 0, 150), (70, 70, 255)))
            #cv2.putText(frame, "ARROW X:" + str(AbestX), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.putText(frame, "ARROW Y:" + str(AbestY), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            #pixelArray = np.array(bar)
            #print(bar)
            if AbestX >= 50:
                search = [AbestX-40,AbestY+self.gfxArrow.shape[0]+1, AbestX+40, AbestY+self.gfxArrow.shape[0]+10] # search relative to arrow
            else:
                print(AbestX)
                search = [20,AbestY+self.gfxArrow.shape[0]+1, AbestX+40, AbestY+self.gfxArrow.shape[0]+10] # search relative to arrow
            Bmatch, BbestX, BbestY = mySearchCVtmp(self.gfxBar, frame, src , search, 0.5, 50, 120, 'Bar: ', grey=False, mark=False, silent=False)
            
        self.gcvdata[0] = 0 # default value, for doing nothing
        if Amatch:
            self.gcvdata[0] = 1 #shot happening
            if Bmatch:  # when both templates are found
                #cv2.putText(frame, "B X:" + str(BbestX), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                #cv2.putText(frame, "B Y:" + str(BbestY), (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                #cv2.putText(frame, "PIXEL RIGHT:" + str(frame[BbestY, BbestX + 4][2]), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                #cv2.putText(frame, "PIXEL LEFT:" + str(frame[BbestY, BbestX - 4][2]), (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                AposX = AbestX + self.gfxArrow.shape[1] / 2
                BposX = BbestX + self.gfxBar.shape[1] / 2
                #print("object centers at  arrow:%d, bar:%d" % (AposX,BposX))
                if self.Xdist == 0.0 and frame[BbestY, BbestX - 4][2] > 150 and frame[BbestY, BbestX + 4][2] > 150:
                    self.Xdist = BposX - AposX
                #print("Xdist: %d" % (self.Xdist))
        elif get_val(BUTTON_6) < 1:
            self.Xdist = 0.0
        
        self.gcvdata.extend(int(self.Xdist*0x10000).to_bytes(4, byteorder='big', signed=True))
        return (frame, self.gcvdata) # send data
            
############### other functions
    
def mySearchCVtmp(temp, frame, temprect, framerect, limit, dx=-1, dy=-1, title='Val:',**kwargs):
    ''' search an area of a template file in an area of the frame for the best match value
    
        frame: current frame captured from GTuner
        temprect: [x1,y1,x2,y2] where x1,y1 is upper left, x2,y2 is lower right corner of the rect
        framerect: [x1,y1,x2,y2] where x1,y1 is upper left, x2,y2 is lower right corner of the area to search in
        limit: if the best match value is in that limit the function returns "True" else "False"
        dx, dy, title: Debug / graphical feedback 
        optional keywords:
        stepX=n : where n is a number to search each n pixel in X direction (default=1)
        stepY=n : where n is a number to search each n pixel in X direction (default=1)
        maxHits=n : exit loop on n matches limit
        '''
        
    stepX = kwargs.pop('stepX', 1)
    stepY = kwargs.pop('stepY', 1)
    maxHits = kwargs.pop('maxHits', 999999999)
    txtOnMatch = kwargs.pop('txtOnMatch', False)
    silent = kwargs.pop('silent', False)
    mark = kwargs.pop('mark', False)
    markColor = kwargs.pop('markColor', (0,0,255))
    grey = kwargs.pop('grey', False)
    
    temp_gfx = temp[temprect[1]:temprect[3], temprect[0]:temprect[2]]  # y, x
    if grey:
        temp_gfx = cv2.cvtColor(temp_gfx, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('_1_temp_gfx.png', temp_gfx)
    inLimit = False
            
    bestVal = 0
    bestValX = -1
    bestValY = -1

    # Store width in variable w and height in variable h of template  
    #w, h = temp_gfx.shape[::-1]
    w = temprect[2]-temprect[0]
    h = temprect[3]-temprect[1]
    
    # frames area
    frame_gfx = frame[framerect[1]:framerect[3], framerect[0]:framerect[2]]  # y, x
    if grey:
        frame_gfx = cv2.cvtColor(frame_gfx, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('_1_frame_gfx.png', frame_gfx)
    #FIX ERROR ON NEXT LINE WITH MATCHING SIZE (shooting from corner gives error sometimes)
    # Now we perform match operations. 
    #print("FRAME: " + str(frame_gfx))
    #print("TEMP: " + str(temp_gfx))
    res = cv2.matchTemplate(frame_gfx, temp_gfx, cv2.TM_CCOEFF_NORMED)
    # Store the coordinates of matched location in a numpy array   
    loc = np.where(res >= limit)
    
    found = False
    hits = 0
    for pt in zip(*loc[::-1]):
        #print("res:", res[pt[1]][pt[0]])
        #if res[pt[1]][pt[0]] == 1:
        #    continue
        found = True
        hits += 1
        ptmod = (pt[0]+framerect[0],pt[1]+framerect[1])
        # Draw the rectangle around the matched region.
        #if mark:
        #    cv2.rectangle(frame, ptmod, (ptmod[0] + w, ptmod[1] + h), markColor, 2)

        if res[pt[1]][pt[0]] > bestVal:
            bestVal = res[pt[1]][pt[0]]
            bestValX, bestValY = ptmod[0], ptmod[1]
            
        if hits == maxHits:
            break
    
    if dx > -1 and dy > -1:
        color = (0, 0, 255)
        if found:
            ptmod = (pt[0]+framerect[0],pt[1]+framerect[1])
            # Draw the rectangle around the matched region.
            if mark:
                cv2.rectangle(frame, ptmod, (ptmod[0] + w, ptmod[1] + h), (255,0,0), 2)
            color = (0, 255, 0)
    
    if found:
        '''if not silent:
            print("tools.bestval:", bestVal, " , X:", bestValX, " , Y:", bestValY)'''
        return bestVal, bestValX, bestValY
    
    return 0, bestValX, bestValY
