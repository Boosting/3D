'''
Generate 2D face landmarks using SFM
Date: 2016/12/22
'''
import numpy as np
import sys
sys.path.append('./eos-maked/bin/')
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

def show(shape,landmarks_2D,P):
	s2=P.dot(shape)
	p2=landmarks_2D
	#plt.plot(s2[0],s2[1],'b.')
	plt.figure()
	plt.plot(s2[0,:],s2[1,:],'b+')
	plt.plot(p2[0],p2[1],'r.')
	plt.plot(p2[0,0],p2[0,0],'wo')
	#plt.axis([-150, 150, -150, 150]

def get_2D_landmarks(shape_param,rotation_param):

	# get shape and 3D landmarks from SFM
	model = eos.morphablemodel.load_model("3D-model/sfm_shape_3448.bin")
	shape = model.get_shape_model().draw_sample(shape_param)
	shape = np.array(shape)
	#txt_name="3D-model/50_3D_point.txt" #50
	txt_name="3D-model/9_3D_point.txt" #9
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

	get_2D_landmarks(shape_param,rotation_param)

def main_random():
	examples_num=100
	p_len=9
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
		
		landmarks_2D=get_2D_landmarks(shape_param,rotation_param)
		landmarks_2D=normalize_landmarks(landmarks_2D,p_len)
		landmarks_2D=np.reshape(landmarks_2D.T,(2*p_len))
		landmarks_2D_list[i,:]=landmarks_2D
		rotation_param_list[i,:]=np.array(rotation_param)
		print 'example ',i
	print '----save lists----'
	np.savetxt('generated_landmarks/landmarks_9_test.txt',landmarks_2D_list)			
	np.savetxt('generated_landmarks/rotation_param_9_test.txt',rotation_param_list)

def main():
	y_list=np.arange(-30,30,5)
	p_list=np.arange(-25,25,5)
	r_list=np.arange(-10,10,5)
	p_len=9
	face_num=10

	examples_num=face_num*len(y_list)*len(p_list)*len(r_list)
	print 'examples number:',examples_num
	landmarks_2D_list=np.zeros((examples_num,2*p_len))
	rotation_param_list=np.zeros((examples_num,3))
	count=0
	for k in range(face_num):
		mu=2*np.random.random_sample()-1 #[-1,1]
		sigma=np.random.random_sample()/2 #[0,0.5]
		shape_param = np.random.normal(mu, sigma, 63)
		# roll pitch yaw
		for y in y_list:
			for p in p_list:
				for r in r_list:
					print count
					rotation_param=[y+np.random.sample()*2,p+np.random.sample()*2,r+np.random.sample()*2]
					#rotation_param=[y,p,r]
					landmarks_2D=get_2D_landmarks(shape_param,rotation_param)
					landmarks_2D=np.reshape(landmarks_2D.T,(2*p_len))
					landmarks_2D_list[count,:]=landmarks_2D
					rotation_param_list[count,:]=np.array(rotation_param)
					count+=1
					#print '-------------',rotation_param

	print '----save lists----'
	np.savetxt('generated_landmarks/landmarks_9_11250_3.txt',landmarks_2D_list)			
	np.savetxt('generated_landmarks/rotation_param_9_11250_3.txt',rotation_param_list)

if __name__ == '__main__':
	main()
	#main_random()
