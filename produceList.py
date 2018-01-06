import os
import re
path = './pos'
bbxPath = './annotations/'
savePath = './save/'
imgName = []
allName = os.listdir(path)
for fileName in allName:
    if not 'bike' in fileName:
        imgName.append(fileName)
if not os.path.exists('./save'):
    os.mkdir('./save')
for fileName in imgName:
    name = fileName.split('.')[0]
    txtName = bbxPath + name + '.txt'
    saveName = savePath + name + '.txt'
#    print(txtName)
#    break
    with open(txtName) as f:
        txt = ''
        for line in f.readlines():
            if 'Image size (X x Y x C) : ' in line:
                num = re.findall('\d+',line)
                txt = num[0] + ' ' + num[1] + '\n'
                x = int(num[0])
                y = int(num[1])
            if 'Bounding box for' in line:
                tmp = line.split(':')[-1]
                tmp = tmp.replace(' ','')
                num = re.findall('\d+',tmp)
                num = [int(i) for i in num]
                txt += str(num[0]/x) + ' '
                txt += str(num[1]/y) + ' '
                txt += str(num[2]/x) + ' '
                txt += str(num[3]/y) + '\n' 
        with open(saveName,'w') as f:
            f.write(txt)
allName = os.listdir(savePath)
maxLen = 0
for file in allName:
    tmpLen = 0
    with open(savePath+file) as f:
        for line in f.readlines():
            tmpLen += 1
        if tmpLen > maxLen:
            maxLen = tmpLen
            maxName = file
if os.path.exists('./test.lst'):
    os.remove('./test.lst')
imgName = os.listdir(savePath)
index = 0
for img in imgName:
    imgPath = savePath + img
    with open(imgPath,'r') as f:
        if index == 0:
            txt = str(index) + '\t4\t5\t'
        else:
            txt = '\n' + str(index) + '\t4\t5\t'          #to ensure there is no redundant empty line
        index += 1
        cnt = 1
        line = f.readline()
        h,w = line.strip().split(' ')
        txt += str(h) + '\t'
        txt += str(w)
        for line in f.readlines():
            cnt += 1
            Dot = line.strip().split(' ')
            txt += '\t1'
            for tmp in Dot:
                txt += '\t' + tmp
        while cnt < maxLen:
            cnt += 1
            txt += '\t-1\t0\t0\t0\t0'
        prefix_img = img.split('.')[0]
        txt += '\t' + prefix_img + '.png'
    with open('./test.lst','a') as f:
        f.write(txt)