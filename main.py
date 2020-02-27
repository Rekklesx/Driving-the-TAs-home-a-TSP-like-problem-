import os
import sys

path="/Users/rekkles/Desktop/cs170/project/newfolder/100w/"

files=os.listdir(path)

num=0
count=0
for i in files:
    print(i)
    front="python3 project.py "+path+i
    #print(front)
    count+=1
    print("Running:",round(count*100/1088,2),"%")
    os.system(front)

