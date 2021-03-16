# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:12:55 2021

@author: Asus
"""
import os
import tarfile
import urllib
import shutil
import json
import random
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

xml_dir = 'D:/Luyi/data/Luyi Huang/2_People_annotations'
xml_files = [os.path.join(xml_dir, x) for x in os.listdir(xml_dir) if x[-3:] == 'xml']
xml_files[0]
all_images=os.listdir('D:/Luyi/data/Luyi Huang/2 People')
image_dir='D:/Luyi/data/Luyi Huang/2 People'
classes = ['human']
categories = [
    {
        'class_id': 0,
        'name': 'sitting'
    },
        {
        'class_id': 1,
        'name': 'standing'
    },
            {
        'class_id': 2,
        'name': 'lying down'
    },
                {
        'class_id': 3,
        'name': 'Undefinable'
    },
                {
        'class_id': 4,
        'name': 'Heat Source'
    },
                
]
relation={
    '0':'Sitting',
    '1':'Standing',
    '2':'Lying down',
    '3':'Undefinable',
    '4':'Heat Source'}
picture=[]
posture=[]    
def extract_annotation(xml_file_path):
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    annotation = {}
    
    annotation['file'] = root.find('filename').text
    annotation['categories'] = categories
    
    size = root.find('size')
    
    annotation['image_size'] = [{
        'width': int(size.find('width').text),
        'height': int(size.find('height').text),
        'depth': int(size.find('depth').text)
    }]
    
    annotation['annotations'] = []
    
    for item in root.iter('object'):
        class_id = classes.index(item.find('name').text)
        ymin, xmin, ymax, xmax = None, None, None, None
        
        for box in item.findall('bndbox'):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
        
            if all([xmin, ymin, xmax, ymax]) is not None:
                 annotation['annotations'].append({
                     'class_id': class_id,
                     'boundingbox': [xmin, ymin, xmax, ymax]
                    
                 })
    return annotation
for i,image in enumerate(all_images):
    xml_file_path=xml_dir+'/'+image[:-4]+'.xml'
    annot=extract_annotation(xml_file_path)
    for l in range(len(annot['annotations'])):
        classid=annot['annotations'][l].get('class_id')
        gesture=relation.get(str(classid))
        bbox=annot['annotations'][l].get('boundingbox')
        im=Image.open(os.path.join(image_dir,image))
        im=im.crop(bbox)
        im.save('D:/Luyi/gesture/gesture_resize/crop/'+image[:-4]+'{}'.format(l)+".jpg")
        picture.append(image[:-4]+'{}'.format(l)+".jpg")
        posture.append(gesture)
df1=pd.DataFrame (posture, columns=['gesture'])
df2=pd.DataFrame (picture,columns=['id'])
df=pd.concat([df1,df2],axis=1)
df.to_csv(r'gesture.csv', index = False)