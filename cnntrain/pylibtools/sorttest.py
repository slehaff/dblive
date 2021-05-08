import os
import shutil


folder = 'mydir'
for i in range(len(os.listdir(folder))):
    for file  in os.walk(folder):
        if file[-4:]== '.ply':

            shutil.copyfile()
