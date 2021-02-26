import os 
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
#from scan_project.models import Scanner

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import PicForm2
import myglobals
import numpy as np
import cv2
import shutil

mainpath = '/home/samir/dblive/scan/'
infolder = 'peterscan/'
outfolder = 'inputscans/'
offset = 0


def copyfiles(infolder, outfolder, offset):
    # for i in range(0, len(os.listdir(mainpath+outfolder))):
    shutil.copytree(mainpath+infolder+'render0' , mainpath+outfolder+'render'+str(offset))

# movfiles(infolder, outfolder, len(os.listdir(mainpath+outfolder)))

# The module receives http buffer transfers from the scanner and adds them to a global Q, defined in myglobals

@csrf_exempt
def save_uploaded_file(handle, filepath):
    # print ('Handle: ', handle)
    print ('Filename', filepath)
    with open(filepath, 'wb+') as destination:
        for chunk in handle.chunks():
            destination.write(chunk)
    return

@csrf_exempt   
def sendpic(request): 
    picform = PicForm2()
    mycontext = {
        'pic': picform,
        'name': "Peter",
    }
   
    if request.method == 'POST':
        picform = PicForm2(request.POST, request.FILES)
        if picform.is_valid():
            FileFolder = '/home/samir/dblive/scan/peterscan/render0/'
            filelist =[]
            if request.FILES.get('Picture'):
                pictures = request.FILES.getlist('Picture')
                for p in pictures:
                    filepath = FileFolder + p.name
                    filelist.append(filepath)
                    print(filepath)
                    save_uploaded_file(p, filepath)
                # move triplet to contailer !! to be implemented by sal
                copyfiles(infolder, outfolder, len(os.listdir(mainpath+outfolder)))

            # for debug
            for p in ['Pic1','Pic2','Pic3']:
                if request.FILES.get(p): 
                    filelist.append(FileFolder + request.FILES[p].name)
                    save_uploaded_file(request.FILES[p], FileFolder + request.FILES[p].name)
                    
            if request.POST.get('cmd')=="stitch":
                print("Stitching.....")
            if request.FILES.get('Pic1'):
                # debug
                param = "?"
                for f in request.FILES.items():
                    param += "picture=" +f[1].name + "&"
                param += 'stitch=ud.jpg'
                url = "/test/pic_info/" + param
                #print(url)
                return redirect(url)
            else:
                return JsonResponse({'result':"OK"})
        print ("not valid", picform.errors)
        return render(request, 'pic.html', mycontext)
    return render(request, 'pic.html', mycontext)


@csrf_exempt
def pic(request):
    f1 = '/home/samir/dblive/scan/static/scan_image_folder/black.jpeg'
    f2 = '/home/samir/dblive/scan/static/scan_image_folder/color.jpeg'
    f3 =  '/home/samir/dblive/scan/static/scan_image_folder/high.jpeg'
    print('picd')
    
    picform = PicForm2()
    mycontext = {
        'title': "Test Upload",
        'pic': picform,
        'name': "Samir",
    }

    if request.method == 'POST':
        #print('FILES: ', request.FILES)
        picform = PicForm2(request.POST, request.FILES)
        if picform.is_valid():
            print('valid')
            save_uploaded_file(request.FILES['Pic1'], f1)
            save_uploaded_file(request.FILES['Pic2'], f2)
            save_uploaded_file(request.FILES['Pic3'], f3)
            b1 = cv2.imread(f1, 0).astype(np.float32)
            b2 = cv2.imread(f2, 1).astype(np.float32)
            b3 = cv2.imread(f3, 0).astype(np.float32)
            myglobals.inque.append(b1)
            myglobals.inque.append(b2)
            myglobals.inque.append(b3)
            print('Queue:', len(myglobals.inque))
            return JsonResponse({'result':"OK"})
        return render(request, 'api/pic.html', mycontext)

    return render(request, 'api/pic.html', mycontext)

def test(request):
    return render(request, "web/test.html")
