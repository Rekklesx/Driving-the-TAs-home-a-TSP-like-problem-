import string
import numpy as np
import sys
file=sys.argv[1][:-3]
f=open(file+".in")
#f=open('50.in','r')
#读入location和home节点的个数
lo_num=(int)(f.readline())
ho_num=(int)(f.readline())
#读入location节点的名字并储存到列表lo_list中
temp=f.readline()
temp=temp.split('\n',1)[0]
if temp.find(' ')!=-1:
    lo_list=temp.split(' ',lo_num)
elif temp.find('\t')!=-1:
    lo_list=temp.split('\t',lo_num)
#lo_list=temp.split(' ',lo_num)
#读入home节点的名字并储存到列表ho_list中
temp=f.readline()
temp=temp.split('\n',1)[0]
if temp.find(' ')!=-1:
    ho_list=temp.split(' ',ho_num)
elif temp.find('\t')!=-1:
    ho_list=temp.split('\t',ho_num)
#ho_list=temp.split(' ',ho_num)
#读入起点名字并在lo_list中搜索，并将起点的索引存放到s_index中
temp=f.readline()
start=temp.split('\n',1)[0]
for index in range(0,lo_num):
    if lo_list[index] == start:
        s_index=index
#在邻接矩阵中存储weight
con_matrix=np.zeros((lo_num,lo_num))
for i in range(0,lo_num):
    temp=f.readline()
    temp=temp.split('\n',1)[0]
    if temp.find(' ')!=-1:
        temp_list=temp.split(' ',lo_num)
    elif temp.find('\t')!=-1:
        temp_list=temp.split('\t',lo_num)
    for j in range(0,lo_num):
        #如果两边不相连就把weight设为比较大的一个值 
        #表示"infinity" : float("inf") / math.inf (需要导入math包) - MXX
        if temp_list[j]=='x':
            con_matrix[i][j]=float("inf")
        else:
            con_matrix[i][j]=temp_list[j]
connect_m=np.zeros((lo_num,lo_num))
for i in range(0,lo_num):
    for j in range(0,lo_num):
        connect_m[i][j]=con_matrix[i][j]

connect_f=np.zeros((lo_num,lo_num))
for i in range(0,lo_num):
    for j in range(0,lo_num):
        connect_f[i][j]=con_matrix[i][j]
    
#Floyd - Marshal
V = range(len(con_matrix))
for k in V:
    for u in V:
        for v in V:
            con_matrix[u][v] = min(con_matrix[u][v],
                                con_matrix[u][k] + con_matrix[k][v])
        con_matrix[u][u]=0
s_dis=[[0 for i in range(2)] for i in range(lo_num)]
for i in range(lo_num):
    s_dis[i][0]=lo_list[i]
    s_dis[i][1]=con_matrix[0][i]
#print(s_dis)
path=[]
path.append(s_index)



#def make_table():
path_leng=0
visit = [0 for i in range(ho_num)]
#维护home节点和任意location节点之间的最短路径表
#[h][i][0]-home name;  [h][i][1]-location name;   [h][i][2]-distance;
dropoff=[[[0 for i in range(3)] for i in range(lo_num)]for i in range(ho_num)]
for h in range(ho_num):
    for i in range(lo_num):
        dropoff[h][i][0]=ho_list[h]
        dropoff[h][i][1]=lo_list[i]
        for a in range(lo_num):
            if lo_list[a]==ho_list[h]:
                lo_ind=a
        dropoff[h][i][2]=(con_matrix[lo_ind][i])
    def takeSecond(elem):
        return elem[2]
    dropoff[h].sort(key=takeSecond)
#print(dropoff)

#make_table()
#每个home找到path上离它最近的点，i是path上点的数
'''
def compute_cost(x):
    to_cost=0#总代价
    path_leng=0
    #tpath=path#临时path
    #tpath.append(x)#把一个点x加到临时path里
    tpath=[0 for i in range(len(path))]
    for i in range(len(path)):
        tpath[i]=path[i]
    tpath.append(x)
    for h in range(ho_num):
        bigger = [0 for i in range(len(tpath))]#path上的每个位置
        for i in range(len(tpath)):
            for j in range(lo_num):
                if dropoff[h][j][1]==lo_list[tpath[i]]:#path[i]存的是location的序号，lo_list是该序号对应的名字
                    bigger[i]=dropoff[h][j][2]#就把当前home到当前locaton的距离存在bigger[i]
                    break
        small=float("inf")
        for i in range(len(tpath)):
            if bigger[i]<=small:
                small=bigger[i]#每个home到path上最近的点的距离
        to_cost+=small
    for i in range(len(tpath)-1):
        path_leng+=con_matrix[tpath[i]][tpath[i+1]]
    to_cost+=(2/3)*path_leng
    to_cost+=con_matrix[x][s_index]
    #tpath.remove(x)
    tpath.pop(-1)
    return to_cost
'''
def compute_cost(x,y):
    to_cost=0#总代价
    path_leng=0
    #tpath=path#临时path
    #tpath.append(x)#把一个点x加到临时path里
    tpath=[0 for i in range(len(path))]
    for i in range(len(path)):
        tpath[i]=path[i]
    tpath.append(x)
    tpath.append(y)
    for h in range(ho_num):
        bigger = [0 for i in range(len(tpath))]#path上的每个位置
        for i in range(len(tpath)):
            for j in range(lo_num):
                if dropoff[h][j][1]==lo_list[tpath[i]]:#path[i]存的是location的序号，lo_list是该序号对应的名字
                    bigger[i]=dropoff[h][j][2]#就把当前home到当前locaton的距离存在bigger[i]
                    break
        small=float("inf")
        for i in range(len(tpath)):
            if bigger[i]<=small:
                small=bigger[i]#每个home到path上最近的点的距离
        to_cost+=small
    for i in range(len(tpath)-1):
        path_leng+=con_matrix[tpath[i]][tpath[i+1]]
    to_cost+=(2/3)*path_leng
    to_cost+=con_matrix[y][s_index]
    #tpath.remove(x)
    tpath.pop(-1)
    tpath.pop(-1)
    return to_cost


#################################

#初始值改变

current_min_cost=compute_cost(s_index,s_index)
#print("cost of not driving:",current_min_cost)
last_min_cost=float("inf")
l_i=0
l_w=0

while current_min_cost<last_min_cost:
    current_lo=l_i
    last_min_cost=current_min_cost
    min_cost=float("inf")
    #for i in range(0,lo_num):
    
    for i in range(0,lo_num):
        #只看邻居  #转念一想满足三角关系就没必要了
        if connect_m[path[-1]][i] != float("inf") and i != path[-1] :
            for w in range(0,lo_num):
                if connect_m[i][w] != float("inf") and w != i:
                    temp=compute_cost(i,w)
                    if temp<=min_cost:
                        min_cost=temp
                        l_i=i
                        l_w=w
    #if(path[-1]!=l_i):
    current_min_cost=min_cost
    if current_min_cost<last_min_cost:
        path.append(l_i)
        path.append(l_w)
        #print(path)
    #last_min_cost=current_min_cost
    #total_min_cost=min_cost


if(int(path[-1])!=s_index):
    distFloyd = np.zeros((lo_num,lo_num))
    pathFloyd = [[[] for _ in range(lo_num)] for __ in range(lo_num)]
    for i in range(lo_num):
        for j in range(lo_num):
            distFloyd[i][j] = connect_f[i][j]
            pathFloyd[i][j] = [i,j]
    for i in range(lo_num):
        distFloyd[i][i] = 0
        pathFloyd[i][i] = [i]
    for k in range(lo_num):
        for i in range(lo_num):
            for j in range(lo_num):
                if distFloyd[i][j] > distFloyd[i][k] + distFloyd[k][j]:
                    distFloyd[i][j] = distFloyd[i][k] + distFloyd[k][j]
                    pathFloyd[i][j] = pathFloyd[i][k] + pathFloyd[k][j][1:]
    tempStart = int(path[-1])
    # print(type(pathFloyd[1][1]))
    tempMxx = 0
    for i in pathFloyd[tempStart][s_index]:
        if tempMxx != 0:
            path.append(i)
        tempMxx = tempMxx + 1
    # print(tempMxx)

f=open(file+".out","w")
#f=open("50.out","a")
for i in range(len(path)):
    #print(lo_list[path[i]],end=' ')
    f.write(lo_list[path[i]]+" ")
#print()
f.write("\n")


drop_lo_num=0
drop_lo = [0 for i in range(lo_num)]
dropoff_matrix=[[0 for i in range(2)] for i in range(lo_num*lo_num)]
for h in range(ho_num):
    bigger = [0 for i in range(len(path))]#path上的每个位置
    drop_name = [0 for i in range(len(path))]
    for i in range(len(path)):
        for j in range(lo_num):
            if dropoff[h][j][1]==lo_list[path[i]]:#path[i]存的是location的序号，lo_list是该序号对应的名字
                bigger[i]=dropoff[h][j][2]#就把当前home到当前locaton的距离存在bigger[i]
                drop_name[i]=dropoff[h][j][1]
                break
    small=float("inf")
    small_i=float("inf")
    for i in range(len(path)):
        if bigger[i]<=small:
            small=bigger[i]#每个home到path上最近的点的距离
            small_i=i
    for i in range(lo_num):
        if lo_list[i]==drop_name[small_i] and drop_lo[i]==0:
            drop_lo[i]=1
            drop_lo_num+=1
    dropoff_matrix[h][0]=drop_name[small_i]
    dropoff_matrix[h][1]=ho_list[h]
#print(drop_lo[0])
f.write(str(drop_lo_num)+"\n")
used_drop_lo=[0 for i in range(lo_num)]
if len(path)>1:
    for i in range(len(path)-1):
        if drop_lo[path[i]]==1 and used_drop_lo[path[i]]==0:
            f.write(lo_list[path[i]]+" ")
            used_drop_lo[path[i]]=1
            #print(lo_list[path[i]],end=' ')
            for j in range(lo_num*lo_num):
                if dropoff_matrix[j][0]==lo_list[path[i]]:
                    f.write(dropoff_matrix[j][1]+" ")
                    #print(dropoff_matrix[j][1],end=' ')
            #print()
            f.write("\n")
else:
    for i in range(len(path)):
        if drop_lo[path[i]]==1 and used_drop_lo[path[i]]==0:
            f.write(lo_list[path[i]]+" ")
            used_drop_lo[path[i]]=1
            #print(lo_list[path[i]],end=' ')
            for j in range(lo_num*lo_num):
                if dropoff_matrix[j][0]==lo_list[path[i]]:
                    f.write(dropoff_matrix[j][1]+" ")
                    #print(dropoff_matrix[j][1],end=' ')
            #print()
            f.write("\n")
#print(current_min_cost)
#print(path)
f.close()

