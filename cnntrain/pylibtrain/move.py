import os
import shutil

mainpath = '/home/samir/Desktop/blender/pycode/'
infolder = 'multi/300newplanes/'
outfolder = 'multi/multiscans/'
offset = 800


def movfiles(infolder, outfolder, offset):
    for i in range(0, len(os.listdir(mainpath+infolder))):
        shutil.move(mainpath+infolder+'render'+str(i) , mainpath+outfolder+'render'+str(i+offset))

movfiles(infolder, outfolder, offset)