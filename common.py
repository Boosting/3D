import numpy as np

def read_txt(name):
	fp=open(name,'r')
	line=fp.readlines()
	fp.close()
	p=np.array(line,dtype=int)
	return p


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
