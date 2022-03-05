
import cv2
import argparse
import numpy as np
#import serial
from time import sleep
#ser = serial.Serial('/dev/ttyACM0')  # open serial port
ctr=300
center_y=238
topleft=0
topright=0
bottomleft=0
bottomright=0
cap = cv2.VideoCapture('testvideo.mp4')
c=0
k=0
i=0
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


        
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    #global c
    #c+=1
    #print(c) prints numbers
    label = str(classes[class_id])
    conf=round(confidence,1)
    conf=str(conf)
    color = (255,255,0)
    if(label=="person"):     
        
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        w=x_plus_w-x
        h=y_plus_h-y
        #pointx=x+w/2
        #pointy=y+h/2
        #xv=pointx-300
        #yv=pointy-238
        '''
        if(xv<0 and yv<0):
            print("top left")
        elif(xv>0 and yv<0):
            print("top right")
        elif(xv>0 and yv <0):
            print("bottom left")
        elif(xv>0 and yv>0):
            print("bottom right")
    '''
       # cv2.circle(img,(xv,yv),5,(255,0,0),-1)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(img, conf, (x+20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    else:
        pass      

while True:
    ret,image=cap.read()

    w = image.shape[1]
    h = image.shape[0]
    w=int(w/2)
    h=int(h/2)

    cv2.line(image,(0,h),(w*2,h),(0,0,255),5)
    cv2.line(image,(w,0),(w,h*2),(0,0,255),5)
    cv2.imshow("frame",image)
    if ret:        
        
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        classes=None
        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4  
    


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        j=0
        for i in indices:
            j+=1
            #print(j)
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            #print(i+1)
            
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            #W = image.shape[1]
            #w=W/2
            #w=int(w)
            #H = image.shape[0]
            #h=H/2
            #h=int(h)
            #pointx=int(x+w)
            #pointy=int(y+h)
            #cv2.circle(image,(pointx,pointy),3,(0,255,0),2)
            #xv=pointx-w
            #yv=pointy-h
            xv=round(x+w/2)
            yv=round(y+h/2)
            cv2.circle(image,(xv,yv),3,(0,255,0),-1)
            if(xv>0 and yv>0):
                topleft=1
            
            if(xv<0 and yv>0):
                topright=1
            
            if(xv>0 and yv <0):
                bottomleft=1
            
            if(xv>0 and yv<0):
                bottomright=1
            
            i=i+1

        i=str(i)
        cv2.putText(image, 'Count:', (100,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, i, (200,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        print('TL'+'/'+'TR'+'/'+'BL'+'/'+'BR')
        print(str(topleft)+'/'+str(topright)+'/'+str(bottomleft)+'/'+str(bottomright))
        #s = str(topleft)+'\t'+str(topright)+'\t'+str(bottomleft)+'\t'+str(bottomright)+'\n'
        #ser.write(s.encode())
        color=(0,0,255)
        cv2.imshow("object detection", image)
        topleft=0
        topright=0
        bottomleft=0
        bottomright=0
       
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break  
cap.release()

cv2.destroyAllWindows()