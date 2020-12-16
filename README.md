# dblive
DBlive is used for running live code, typically connecting to danwand.
DBlive is deployed with python manage.py runserver 0.0.0.0:8000 to connect to the wand
DanWand runs startloop.py from rc.local on power up, startloop uses send.py, diascan.py for taking triple shots and uploading them to:...
Init runs a buffer queing routine that adds the incoming buffers to inque
every 20 sec, inque is polled and buffers are moved to individual render(i) buffers in scan_im...
Init__.py in pylibtrain i cnnpredict has a loop for moving incooming images from inque to individual folders
Destination folder is: folder = '/home/samir/Desktop/blender/pycode/inputscans/render'

Training is a major task for dblive, Folder format for valid input is:
UNTrain.py is the main training input py, this is a Unet trainer with an identical neural net for processing:
    HFringe to HWrapped
    HWrapped to Kvalue file
