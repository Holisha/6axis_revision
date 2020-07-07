import os,csv,math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from cycler import cycler
from PIL import Image
import random
from csv_classify import classify
from utils import argument_setting
cnt=0
csv_cnt=0
# myfile="char00312_stroke.txt" #########
newfile="new6axis.txt"
def addnoise():
    with open(newfile, "w") as fw:
        with open(args.filename, "r", newline='') as f:
            for line in f:
                fw.write('movl 0 ')
                L=line.split()# L[2:8]為六軸
                for item in L[2:8]: 
                    fw.write(str(float(item)+random.uniform(args.noise[0],args.noise[1]))+' ')# noise 範圍
                fw.write('100.0000 ')
                fw.write(L[-1]+'\n')
def angle2deg(angle):
    return angle * math.pi / 180 

def get_6d_3d(path):
    """
    input: 6 axis txt file
    output: (x, y, z) visualized data 
    """
    data = []
    with open(path) as txtFile:
        for row in txtFile:

            row = row.lstrip().split(' ')

            x = float(row[2])
            y = float(row[3])
            z = float(row[4])
            a = angle2deg(float(row[5]))
            b = angle2deg(float(row[6]))
            c = angle2deg(float(row[7]))
            n_stroke = int(row[9][6:])
            
            Ra = [1, 0, 0,
                0, math.cos(a), -1 * math.sin(a),  
                0, math.sin(a), math.cos(a)      ]

            Rb = [math.cos(b), 0, math.sin(b),
                0, 1, 0,  
                -1 * math.sin(b), 0, math.cos(b)      ]

            Rc = [math.cos(c), -1 * math.sin(c), 0,
                math.sin(c), math.cos(c), 0,  
                0, 0, 1      ]

            Ra = np.array(Ra).reshape(3, 3)
            Rb = np.array(Rb).reshape(3, 3)
            Rc = np.array(Rc).reshape(3, 3)
            
            R = np.dot(np.dot(Rc, Rb), Ra)
            
            A = [R[0, 0], R[0, 1], R[0, 2], x,
                R[1, 0], R[1, 1], R[1, 2], y,
                R[2, 0], R[2, 1], R[2, 2], z,
                0, 0, 0, 1]
            A = np.array(A).reshape((4, 4))

            B = np.identity(4)
            B[2, 3] = 185 # 毛筆長度 185 mm

            T = np.dot(A, B)

            data.append([T[0, 3], T[1, 3], T[2, 3], n_stroke])
            
    return data  

def vis_2d(data, character, fn):
    global cnt
    data = np.array(data).reshape(-1, 4)
    x = []
    y = []
    last_stroke = 1
    end = data.shape[0] - 1
    plts = []
    a = np.arange(0, 1, 0.001)


    for n in range(data.shape[0]):
        
        if n == end: # 最後一個筆畫的結束點
            plt.figure(figsize=(10, 10))
            plt.axis( 'off' )
            cnt+=1
            plts.append(plt.plot(x, y,color="blue", label='%d' % last_stroke)[0]) # 畫出前一筆劃
            plt.text(x[0], y[0], str(int(last_stroke)), fontsize=12) # 在前一筆劃起始點標記筆劃數
            x = []
            y = []
            plt.savefig('%s/' % (fn[:-4])  +str(cnt) +'.png')
            plt.close()


        if data[n, 3] == last_stroke: # 同一個筆畫
            if data[n, 2] > 5:
                continue
            x.append(data[n, 0])
            y.append(data[n, 1])


        else: # 到達下一個筆畫的起點
            
            plt.figure(figsize=(10, 10))
            plt.axis( 'off' )
            if data[n, 2] > 5:
                continue
            if x != [] and y != []:
                plt.text(x[0], y[0], str(int(last_stroke)), fontsize=12) # 在前一筆劃起始點標記筆劃數
                plts.append(plt.plot(x, y, color="blue", label='%d' % last_stroke)[0]) # 畫出前一筆劃
                cnt+=1
                plt.savefig('%s/' % (fn[:-4]) +str(cnt) +'.png')
                plt.close('all')

            last_stroke = data[n, 3]
            x = [data[n, 0]]
            y = [data[n, 1]]
        
        
            
    n_stroke = int(data[-1, 3])  

def read_common(fn):
    common = {}
    with open(fn, newline='') as csvFile:
        rows = csv.reader(csvFile)
        for row in rows:
            f_name = '%s' % row[0][0:4] + '%05d' % int(row[0][4:])  + '_stroke.txt'
            charactor = row[1]
            common[f_name] = charactor
    return common

def get_6d(fn):
    # get 6d data from robot used txt.
    # Input: movl 0 -19.8346 416.6156 169.5282 150.5452 17.2783 -85.6258 100.0000 stroke1
    # Output: -19.8346 416.6156 169.5282 150.5452 17.2783 -85.6258
    global csv_cnt
    csv_cnt +=1
    new_6d = []
    
    with open(fn) as txtFile:
        for row in txtFile:
            row = row.strip().split()
            new_6d.append([row[2], row[3], row[4], row[5], row[6], row[7], row[9]])
    
    if not os.path.isdir('%s' % (fn[:-4])):
        os.mkdir('%s' % (fn[:-4]))
    
    fn = '%s/6axis_data' % (fn[:-4]) + str(csv_cnt) + '.csv'

    with open(fn, "w", newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        writer.writerows(new_6d)

def copy(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
        if not os.path.isdir(dst):
            os.mkdir(dst)
            print('create ', dst)
    for f in os.listdir(src):
        shutil.copyfile(src + '/' + f, dst + '/' + f)

def makepic(number,character):
    
    try:
        shutil.rmtree('./new6axis')
    except:
        pass
    # vis_2d(data, character, fn)
    fn=newfile
    if not os.path.isdir('%s' % (fn[:-4])):
        os.mkdir('%s' % (fn[:-4])) # 建立資料夾
        print('create ' + '%s' % (fn[:-4]))
    for i in range(number): 
        addnoise()   
        get_6d(fn) # 從原始txt取出6軸資料並存於資料夾
        data = get_6d_3d('%s' % fn) # 將6d資料轉成(x, y, z)
        # vis_2d(data, character, fn) # 視覺化(x, y, z)#########################

if __name__ == '__main__':
    args = argument_setting()
    number = args.output_num
    with open(args.filename,'r') as f:
        data=f.readlines()
        stroke=data[-1]
        tol_strokes=stroke.split()[-1][6:]  
    with open("char_list.csv",'r',encoding='utf8',  newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[0] == "char"+args.filename[6:9]:
                character = row[1]
    makepic(number,character)
    os.chdir("./new6axis")
    try:
        os.mkdir("6axis_data")
    except:
        pass
    for i in range(1,number+1):
        shutil.move("6axis_data"+str(i)+".csv","6axis_data")
    classify(number,int(tol_strokes))
    
