# K9 License Plate Detector - Board

This repository is a part of an project prepared for Team Programming classes in UKW university in Bydgoszcz. Here we're providing
scripts to run object detection model in tensorflow lite format to detect license_plates.

The main goal of this project is to create mechanism for recognizing vehicles license plates.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development.

### Prerequisites

What things you need to install the software and how to install them

* Python 3.4+ (tensorflow does not support python 3.8 yet)
* Git - instruction

### Installing

A step by step series of examples that tell you how to get a development env running

#### Install python

Python can be downloaded [here](https://www.python.org/downloads/release/python-377/)

**Make sure to install pip manager! Add python to your path!**

Check `Python` installation

``` 
python3 --version
pip3 --version
virtualenv --version
```

Preferably create `Virtualenv` 

``` 
virtualenv --system-site-packages -p python3 ./venv
```

Activate `Virtualenv` 

``` 
source ./venv/bin/activate  # sh, bash, ksh, or zsh
```

To exit `Virtualenv` 

``` 
deactivate # don't exit until you're done using TensorFlow
```

Install reqiurements provided in [requirements.txt](requirements.txt) file.

```
pip install -r requirements.txt # This installs needed python dependencies
pip list # shows installed packages
```

To verify `tensorflow` installation use this command

``` 
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## Running the detection

To run detection script use `detect.py` file.

Make sure to edit `detect.py` to swith to your detection source _(lines: 125-126) - this will be parameterized in future release_:

```
# Get input source
# cap = cv2.VideoCapture(0) #Use this for camera input as a source
cap = cv2.VideoCapture('data/test_video.mp4') # Use this for video file as an input
```

Example usage:
```
python detect.py --model=model/detect.tflite --labels=labels/label.txt
```

## Features

* Loading tensorflow light model and labels file
* Detecting license plates on a video source
* Draws bounding boxes around detected objects with detection score ant detection time

### Todo

* Parameterize input source
* OCR for detetected licene plate object
* Matching detected license plate number against known numbers

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/KonradKarimi/K9UVR_Board/tags). 

## Authors

* **Konrad Karimi** - *Initial work* - [KonradKarimi](https://github.com/KonradKarimi)
* **Przemysław Tarnecki** - *On device integration* - [Isaac-mtg](https://github.com/Isaac-mtg)
* **Mariusz Frelke** - *Data for model training* - [mfrelke](https://github.com/mfrelke)

See also the list of [contributors](https://github.com/KonradKarimi/K9UVR_Board/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* On MacOS Mojave and higher choosing detection source to built in camera will cause program to crash _(due to security/permissions issues)_
* Using [tensorflow/examples](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi) code as reference.
* Works slow when with preview window.
