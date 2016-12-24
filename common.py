'''
some common functions(e.g. read txt files, normalize landmarks, etc)
Author: YadiraF 
Mail: fengyao@sjtu.edu.cn
Date: 2016/12/24
'''
import numpy as np
import os

# read points index (e.g. 3D-model/9_3D_points.txt)
def read_txt(name):
	fp=open(name,'r')
	line=fp.readlines()
	fp.close()
	p=np.array(line,dtype=int)
	return p

# read real landmarks (e.g. test-data/k.txt)
def read_key(name):
	fp=open(name,'r')
	lines=fp.readlines()
	fp.close()
	print len(lines)
	landmarks=np.zeros((len(lines),18))
	for ind,line in enumerate(lines):
		tmp=line.strip().split(' ')
		x=tmp[1:10]
		y=tmp[10:19]
		for i in range(9):
			landmarks[ind,i*2]=x[i]
			landmarks[ind,i*2+1]=y[i]
	return landmarks

# read real landmarks format 2 (e.g. test-data/data) 
def read_key_single(name):
	fp=open(name,'r')
	lines=fp.readlines()
	fp.close()
	landmark=np.zeros((2,9))
	for ind,line in enumerate(lines):
		tmp=line.strip().split('  ')
		landmark[0,ind]=float(tmp[0])
		landmark[1,ind]=-float(tmp[1])
	#tmp=np.zeros((2,))
	#tmp[:]=landmarks[:,7]
	#landmarks[:,7]=landmarks[:,6]
	#landmarks[:,6]=tmp
	landmark=normalize_landmarks(landmark,9)
	landmark=np.reshape(landmark.T,(1,2*9))
	return landmark
def read_key_all(folder):
	examples_num=17
	landmarks=np.zeros((examples_num,18))
	
	for i in range(examples_num):
		name=folder+str(i+1)+'.txt'
		landmarks[i,:]=read_key_single(name)
	return landmarks

# normalize landmarks	
def normalize_landmarks(landmarks,flag):
	if flag==9:
		center_ind=0
		p1_ind=2
		p2_ind=3
	elif flag==50:
		center_ind=17
		p1_ind=23
		p2_ind=26
	# transform to center [0,0]
	tmp=landmarks.T-landmarks[:,center_ind].T
	landmarks=tmp.T
	# let the distance of left eye corner and right eye corner=1
	p1=landmarks[:,p1_ind] # left
	p2=landmarks[:,p2_ind] # right
	lamb= np.linalg.norm(p1-p2) 
	landmarks/=lamb
	return landmarks

if __name__=='__main__':
	read_key_single('test-data/data/1.txt')
	#read_key_all('test-data/data/')
