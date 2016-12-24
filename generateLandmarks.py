'''
Generate 2D face landmarks using SFM
Author: YadiraF 
Mail: fengyao@sjtu.edu.cn
Date: 2016/12/24
'''
import numpy as np
import argparse
import sys
sys.path.append('eos-maked/bin/')
import eos
from common import *
#import matplotlib.pyplot as plt

#https://en.wikipedia.org/wiki/Rotation_matrix
def get_rotation_matrix(y,p,r):
    # convert angle
    y=y*np.pi/180.0
    p=p*np.pi/180.0
    r=r*np.pi/180.0
    #yaw  Ry
    Y=np.array([[np.cos(y),  0, np.sin(y)],
                [0,          1,         0],
                [-np.sin(y), 0, np.cos(y)]])
    #pitch Rx
    P=np.array([[1,         0,          0],
                [0, np.cos(p), -np.sin(p)],
                [0, np.sin(p),  np.cos(p)]])
    #roll Rz
    R=np.array([[np.cos(r), -np.sin(r), 0],
                [np.sin(r),  np.cos(r), 0],
                [        0,          0, 1]])
    Rotation=R.dot(P.dot(Y))
    return Rotation

# pinhole camera projection
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
def get_projection_matrix(p):
    [yaw,pitch,roll]= p
    [tx,ty,tz,alpha_u,alpha_v,u_0,v_0]=[0,0,0,1,1,0,0]
 
    intr=np.array([[alpha_u,     0.0,   u_0],
                   [    0.0, alpha_v,   v_0],
                   [    0.0,     0.0,   1.0]])
    
    R=get_rotation_matrix(yaw,pitch,roll) # yaw pitch roll 
    
    T=np.array([[tx],[ty],[tz]]) # tx ty tz   
    
    extr=np.hstack((R,T))

    P=intr.dot(extr)
    return P

# show face and landmarks for verification
def show(shape,landmarks_2D,P):
	s2=P.dot(shape)
	p2=landmarks_2D
	plt.figure()
	plt.plot(s2[0,:],s2[1,:],'b+') # face
	plt.plot(p2[0],p2[1],'r.') # landmarks
	plt.plot(p2[0,0],p2[0,0],'wo') # center
	#plt.axis([-150, 150, -150, 150] 

# SFM
# https://github.com/patrikhuber/eos
def get_2D_landmarks(shape_param,rotation_param,p_len):

	# get shape and 3D landmarks from SFM
	model = eos.morphablemodel.load_model("3D-model/sfm_shape_3448.bin")
	shape = model.get_shape_model().draw_sample(shape_param)
	shape = np.array(shape)
	txt_name="3D-model/%d_3D_point.txt" % p_len
	points_index=read_txt(txt_name)

	# to homo cord
	shape=np.reshape(shape,(len(shape)/3,3)).T
	shape=np.vstack((shape,np.ones((1,shape.shape[1]))))
	landmarks_3D= shape[:,points_index]

	# projection 3D --> 2D
	 # yaw,pitch,roll
	P=get_projection_matrix(rotation_param)
	landmarks_2D=P.dot(landmarks_3D) # 13 center

	# show (for test)
	#show(shape,landmarks_2D,P)
	landmarks_2D=landmarks_2D[:2,:]

	return landmarks_2D

def test(yaw,pitch,roll):
	shape_param=[1.0, -0.5, 0.7]
	rotation_param=[yaw,pitch,yaw]

	landmarks_2D=get_2D_landmarks(shape_param,rotation_param)
	print landmarks_2D

def main_random(sample_args):
	# sample params
	p_len=sample_args.point_len
	face_num=sample_args.face_num
	examples_num=sample_args.face_num
	print 'landmarks points length:',p_len
	print 'face number:', face_num
	print 'examples number:', examples_num
	# save path
	save_folder=sample_args.save_folder
	save_landmarks_path=save_folder+'landmarks_'+str(p_len)+'_'+str(face_num)+'_test.txt'
	save_rotation_param_path=save_folder+'rotation_param_'+str(p_len)+'_'+str(face_num)+'_test.txt'
	# sample params
	landmarks_2D_list=np.zeros((examples_num,2*p_len))
	rotation_param_list=np.zeros((examples_num,3))
	for i in range(examples_num):
		# shape
		mu=2*np.random.random_sample()-1 #[-1,1]
		sigma=np.random.random_sample()/2 #[0,0.5]
		shape_param = np.random.normal(mu, sigma, 63)
		# rotation
		y=np.random.randint(-30,30)
		p=np.random.randint(-25,25)
		r=np.random.randint(-10,10)
		#rotation_param=[y,p,r]
		rotation_param=[y+np.random.sample()*2,p+np.random.sample()*2,r+np.random.sample()*2]
		
		landmarks_2D=get_2D_landmarks(shape_param,rotation_param,p_len)
		landmarks_2D=normalize_landmarks(landmarks_2D,p_len)
		landmarks_2D=np.reshape(landmarks_2D.T,(2*p_len))
		landmarks_2D_list[i,:]=landmarks_2D
		rotation_param_list[i,:]=np.array(rotation_param)
		
		print '%d/%d faces have been processed' % (i+1,face_num)

	np.savetxt(save_landmarks_path,landmarks_2D_list)			
	np.savetxt(save_rotation_param_path,rotation_param_list)
	print '----files have been saved to ',save_landmarks_path,' and ',save_rotation_param_path

def main(sample_args):
	# sample params
	p_len=sample_args.point_len
	face_num=sample_args.face_num
	print 'landmarks points length:',p_len
	print 'face number:', face_num
	# save path
	save_folder=sample_args.save_folder
	save_landmarks_path=save_folder+'landmarks_'+str(p_len)+'_'+str(face_num)+'.txt'
	save_rotation_param_path=save_folder+'rotation_param_'+str(p_len)+'_'+str(face_num)+'.txt'

	# rotation param range
	y_list=np.arange(-30,30,5)
	p_list=np.arange(-25,25,5)
	r_list=np.arange(-10,10,5)

	examples_num=face_num*len(y_list)*len(p_list)*len(r_list)
	print 'examples number:',examples_num

	# initial  
	landmarks_2D_list=np.zeros((examples_num,2*p_len))
	rotation_param_list=np.zeros((examples_num,3))
	count=0

	for k in range(face_num):
		# shape param
		mu=2*np.random.random_sample()-1 #[-1,1]
		sigma=np.random.random_sample()/2 #[0,0.5]
		shape_param = np.random.normal(mu, sigma, 63)
		# rotation param : yaw pitch roll
		for y in y_list:
			for p in p_list:
				for r in r_list:
					rotation_param=[y+np.random.sample()*2,p+np.random.sample()*2,r+np.random.sample()*2] # add noise
					#rotation_param=[y,p,r]
					landmarks_2D=get_2D_landmarks(shape_param,rotation_param,p_len)
					landmarks_2D=normalize_landmarks(landmarks_2D,p_len)
					landmarks_2D=np.reshape(landmarks_2D.T,(2*p_len))
					landmarks_2D_list[count,:]=landmarks_2D
					rotation_param_list[count,:]=np.array(rotation_param)
					count+=1
					#print '-------------',rotation_param
		print '{}/{} faces have been processed'.format(k+1,face_num)
		#sys.stdout.flush()  
	np.savetxt(save_landmarks_path,landmarks_2D_list)			
	np.savetxt(save_rotation_param_path,rotation_param_list)
	print 'Files have been saved to ',save_landmarks_path,'and ',save_rotation_param_path

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--random_flag', type=int, default=0,
						help='if 1, each face num generate 1 landmarks and save file path will be added with _test. default is 0')
	parser.add_argument('--point_len', type=int, default=9,
						help='the length of landmarks, 9 50 are supported now.')
	parser.add_argument('--face_num', type=int, default=100,
						help='the number of face to sample, each face generate 480 landmarks, the number of all examples = face_len*480.')
	parser.add_argument('--save_folder', type=str, default='generated-landmarks/',
						help='the folder to save files. landmarks(rotation parameter) will be saved in result/landmarks(rotation_param)_|point_len|_|face_len|.')
	sample_args = parser.parse_args()
	

	if sample_args.random_flag==0:
		main(sample_args)
	else:
		main_random(sample_args)
