import os
import shutil

def getplys(infolder ):       
    count=len(os.listdir(infolder))
    for i in range(count):
        print(count)
        shutil.copyfile(infolder+'/render'+str(i)+'/pointcl-depth.ply', infolder +'files/plyfolder/points'+str(i)+'.ply')
        shutil.copyfile(infolder+'/render'+str(i)+'/image8.png', infolder +'files/imgfolder/image'+str(i)+'.png')
        
