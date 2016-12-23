import numpy as np
import os

def read_txt(name):
	fp=open(name,'r')
	line=fp.readlines()
	fp.close()
	p=np.array(line,dtype=int)
	return p
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
def read_key_single(name):
	fp=open(name,'r')
	lines=fp.readlines()
	fp.close()
	landmarks=np.zeros((2,9))
	for ind,line in enumerate(lines):
		tmp=line.strip().split('  ')
		landmarks[0,ind]=tmp[0]
		landmarks[1,ind]=tmp[1]
	tmp=np.zeros((2,))
	tmp[:]=landmarks[:,7]
	print tmp
	landmarks[:,7]=landmarks[:,6]
	landmarks[:,6]=tmp
	print landmarks
	landmarks=normalize_landmarks(landmarks,9)
	landmarks=np.reshape(landmarks.T,(1,2*9))
	return landmarks

def read_key_all(folder):
	for i in range(17):
		name=folder+str(i+1)+'.txt'
	
	
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
