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
import threading

left_camera = None
vs=None

class CSI_Camera:

    def __init__ (self) :
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False


    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running=True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running=False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed=grabbed
                    self.frame=frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened
        

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed=self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=3,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
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
            frame = cv2.resize(frame,(300,300))
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




def start_app():
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
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=0,
            display_height=400,
            display_width=960,
        )
    )
    left_camera.start()

    vs = CSI_Camera()
    vs.open(
        gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=0,
            display_height=400,
            display_width=960,
        )
    )
    vs.start()
    cv2.namedWindow("camwindow", cv2.WINDOW_AUTOSIZE)
    print('start opencv')
    time.sleep(2.0)
    fps = FPS().start()
    cnt = 0
    found_tmp = "chair"


#    cv2.namedWindow("camwindow", cv2.NORMAL)
    if (
        not left_camera.video_capture.isOpened()
        or not vs.video_capture.isOpened()
    ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)
   

    while True:
#    while cv2.getWindowProperty("camwindow", 0) >= 0 :

        _ , left_image=left_camera.read()
        _ , frame=vs.read()


        frame = cv2.resize(frame, (960,400))
        frame = imutils.resize(frame, width=960) #400
        (fH, fW) = frame.shape[:2]
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
                    animal = label.split(":")
                    pet = str(animal[0])
                    #print(pet)
                    if cnt == 0 and found_tmp != pet:
                        resp = requests.post('http://cloud.park-cloud.co19.kr/jetson_nano/post-data.php', params=data)
                        cnt = cnt + 1
                        found_tmp = pet 
                    else:
                        cnt=0

            camera_images = np.hstack((left_image, frame))
            cv2.imshow("camwindow", camera_images)


        # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
            if keyCode == 27:
                break

    left_camera.stop()
    left_camera.release()
    vs.stop()
    vs.release()
    cv2.destroyAllWindows()
    print('stop?;')
        # update the FPS counter
    fps.update()

# stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup


if __name__ == "__main__":
    start_app()
