## USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2
import pygame
import requests

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=29,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            #frame = np.array(frame, dtype=object)
            #print(frame)
            try:
                frame = cv2.resize(frame,(300,300))
            except Exception as e:
                print(str(e))
            #frame = imutils.resize(frame, width=400)
            #frame = cv2.resize(frame,  dsize=(720, 480), interpolation=cv2.INTER_AREA) #300 300
            #frame = np.array(frame)
            blob = cv2.dnn.blobFromImage(frame, 0.007843,(300, 300), 127.5)

            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            # write the detections to the output queue
            outputQueue.put(detections)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
    outputQueue,))
p.daemon = True
p.start()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
vs = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

time.sleep(2.0)
fps = FPS().start()
cnt = 0
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it, and
    # grab its imensions
    ret, frame = vs.read()

    #frame = cv2.flip(frame,0)
    try:
        frame = cv2.resize(frame, (400,400))
        frame = imutils.resize(frame, width=400) #400
        (fH, fW) = frame.shape[:2]
    except Exception as e:
        print(str(e))
    # if the input queue *is* empty, give the current frame to
    # classify
    if inputQueue.empty():
        inputQueue.put(frame)

    # if the output queue *is not* empty, grab the detections
    if not outputQueue.empty():
        detections = outputQueue.get()

    # check to see if our detectios are not None (and if so, we'll
    # draw the detections on the frame)
    if detections is not None:
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if confidence < args["confidence"]:
                continue

            # otherwise, extract the index of the class label from
            # the `detections`, then compute the (x, y)-coordinates
            # of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            dims = np.array([fW, fH, fW, fH])
            box = detections[0, 0, i, 3:7] * dims
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            
            pygame.mixer.init()
            pygame.mixer.music.load("exa.mp3")
            pygame.mixer.music.set_volume(1)
            longitude = 1.123213
            latitude = 29.545345
            label_tmp =  "\""+label+"\""
            print(label)

            if "dog" in label or "cat" in label or "horse" in label or "cow" in label:
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)  
                data = {"value1":latitude,"value2":longitude,"value3":label_tmp}
                pygame.mixer.music.play()
                
                if(cnt == 0):
                    resp = requests.post('http://cloud.park-cloud.co19.kr/jetson_nano/post-data.php', params=data)
                    cnt = cnt + 1
    # show the output frame
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #up-side reverse
    #frame = cv2.flip(frame,0)
    #frame = np.array(frame ,dtype = object)
    try:
        frame = cv2.resize(frame, (720,480))
    except Exception as e:
        print(str(e))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
