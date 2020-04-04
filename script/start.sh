#! /bin/bash

tmux \
    new-session  'sudo pigpiod && source ~/tracking_system/venv/bin/activate && python ~/tracking_system/camera_tracker/app/app.py' \; \
    split-window 'source ~/tracking_system/venv/bin/activate && python ~/tracking_system/CamServer/webapp/app.py' \; \
    detach-client
