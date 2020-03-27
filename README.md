# camera_tracker
A Camera tracking system

## Features

1. auto start on boot
2. tracks moving objects
3. web access for manual control and select target to track
4. gimbal reset to initial position after inactivity
5. change the behavior of the gimbal easily with setting files
6. support different setting profiles for various conditions

## Implementation

1. The tracking system consists of two parts: a detector that detects any movement, 
and a tracker that tracks the moving object.
2. The tracking system computes the location, accepts commands and sending data in parallel.
3. The camera tracker outputs the current frame and moving object location by named pipes.
3. The web interface is highly decoupled from the camera tracker. They are only connected 
by named pipes.
4. The motors are solely controled by the camera tracker.
5. The motors are abstracted by the motor_control module.

## Installation

1. activate virtual environment
    1. cd to directory that contains `venv`
    2. `source activate venv/bin/activate`

2. install `camera_tracker` package: 
    1. cd to `camera_tracker/`
    2. `python setup.py develop`
3. install `motor_control` package:
    1. cd to `motor_control/`
    2. `python setup.py develop`



## Run tracker
```
python app/app.py
```
