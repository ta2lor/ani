# ani
sudo python3 pi.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel


#dual screen with php


sudo python3 bye.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel



#opencv
https://www.jetsonhacks.com/2019/11/22/opencv-4-cuda-on-jetson-nano/

after finish install
but have a error about ;no module named cv2'
then check https://www.learnopencv.com/install-opencv3-on-ubuntu/


The site says
############ For Python 2 ############
cd ~/.virtualenvs/facecourse-py2/lib/python2.7/site-packages
ln -s /usr/local/lib/python2.7/dist-packages/cv2.so cv2.so

############ For Python 3 ############
cd ~/.virtualenvs/facecourse-py3/lib/python3.6/site-packages
ln -s /usr/local/lib/python3.6/dist-packages/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so

then it works


#imutils
