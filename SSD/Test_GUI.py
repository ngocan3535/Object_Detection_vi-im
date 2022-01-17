import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
#load the trained model to classify the images

top=tk.Tk()
top.geometry('800x600')
top.title('Object Detection')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
check = 0
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        up=Image.open(file_path)
        print(up)
        uploaded = run(up)
        uploaded.thumbnail(((top.winfo_width()/1.50),(top.winfo_height()/1.50)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        global check
        check = 1
        load()
        print(check)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,
  padx=10,pady=5)

upload.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))

def run(image):
    image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    img = image
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    print(img)
    return img

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top,pady=20, font=('arial',20,'bold'))
heading.configure(text="NO IMAGES TO PROCESS" ,background='#CDCDCD', foreground='#364156')
def load():
    if check == 1:
        heading.configure(text="IMAGE WAS DETECTED SUCCESSFULLY",background='#CDCDCD', foreground='#364156')
        label.after(400,load())
heading.pack()
top.mainloop()























