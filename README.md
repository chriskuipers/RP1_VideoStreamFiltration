# Python Video Stream Filtration
## Prerequisites
These are based on Ubuntu 16.0.4.1 LTS

### Apt packages
```
build-essential
cmake 
pkg-config
libjpeg8-dev 
libtiff5-dev 
libjasper-dev 
libpng12-dev
libavcodec-dev 
libavformat-dev 
libswscale-dev 
libv4l-dev
libxvidcore-dev 
libx264-dev
libgtk-3-dev
libatlas-base-dev 
gfortran
ffmpeg
```

```
python2.7-dev 
```

Or

```
python3.5-dev
```

### PIP packages
```
numpy
imutils
```

### OpenCV
```
OpenCV 3.3.0
```

## Options
```
usage: rp85.py [-h] [-z BLUR] [-Z BLURMEDIAN] [-x BLURGAUSSIAN] [-y WARPING]
               [-Y FILL] [-D DRAW] [-E ENCRYPT] [-b BLURLEVEL]
               [-B BLURPADDING] [-c CODEC] [-C CONFIDENCE] [-d DETECTION]
               [-i FEEDIP] [-e FEEDUSB] [-f FRAMES] [-I INSTALL] [-l LABELS]
               [-L LOGGING] [-m MODEL] [-o ORIGINAL] [-O OUTPUT] [-R RESTREAM]
               [-p PROTOTXT] [-s SHOWLOCAL] [-t TIMER]

optional arguments:
  -h, --help            show this help message and exit
  -z BLUR, --blur BLUR  Set to yes to apply blurring to the detections
  -Z BLURMEDIAN, --blurmedian BLURMEDIAN
                        Set to yes to apply median blurring to the detections
  -x BLURGAUSSIAN, --blurgaussian BLURGAUSSIAN
                        Set to yes to apply gaussian blurring to the
                        detections
  -y WARPING, --warping WARPING
                        Set to yes to apply warping to the detections
  -Y FILL, --fill FILL  Set to yes to apply filling to the detections
  -D DRAW, --draw DRAW  Set to yes to draw a rectangle on the detections
  -E ENCRYPT, --encrypt ENCRYPT
                        Set to yes to apply AES encryption to the original
                        stream
  -b BLURLEVEL, --blurlevel BLURLEVEL
                        Set the blur intensity (size of pixel square
  -B BLURPADDING, --blurpadding BLURPADDING
                        Set the blur padding applied to the frames
  -c CODEC, --codec CODEC
                        Set the type of codec of output video
  -C CONFIDENCE, --confidence CONFIDENCE
                        Confidence level, to filter out weak/incorrect
                        detections
  -d DETECTION, --detection DETECTION
                        Set to yes to turn on detections using DNN
  -i FEEDIP, --feedip FEEDIP
                        Set the URL to the camera
  -e FEEDUSB, --feedusb FEEDUSB
                        Set the USB feed to be used (USB socket number as X,
                        aka /dev/videoX)
  -f FRAMES, --frames FRAMES
                        Set the FPS rate for the video output file
  -I INSTALL, --install INSTALL
                        Install opencv on the specified platform: ubuntu-16.04
                        (Note: this does not work)
  -l LABELS, --labels LABELS
                        Set to 'yes' to label the detections
  -L LOGGING, --logging LOGGING
                        Specify the optional logfile
  -m MODEL, --model MODEL
                        path to Caffe pre-trained DNN model
  -o ORIGINAL, --original ORIGINAL
                        path to output unaltered video file
  -O OUTPUT, --output OUTPUT
                        path to output video file
  -R RESTREAM, --restream RESTREAM
                        restream the edited feed
  -p PROTOTXT, --prototxt PROTOTXT
                        path to Caffe 'deploy' prototxt file
  -s SHOWLOCAL, --showlocal SHOWLOCAL
                        Show the video stream on the local machine
  -t TIMER, --timer TIMER
                        Set the time the script runs
```

## Example
```
python rp85.py --model Caffenet.Model --prototxt Caffenet.Proto --feedusb 0 --detection yes --confidence 0.2 --draw yes --labels yes --blur yes --blurlevel 50 --blurpadding 25
```

