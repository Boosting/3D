import numpy as np
import sys
sys.path.append('./eos_maked/install/bin')
import eos

def read_3D_points_index():
    fp=open("3D-model/50_3D_point.txt",'r')
    line=fp.readlines()
    fp.close()
    points_index=np.array(line,dtype=int)
    return points_index

#https://en.wikipedia.org/wiki/Rotation_matrix
def get_rotation_matrix(r,p,y):
    roll=np.array([[1,         0,       0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    pitch=np.array([[np.cos(p),  0, np.sin(p)],
                    [0,          1,         0],
                    [-np.sin(p), 0, np.cos(p)]])
    yaw=np.array([[np.cos(y), -np.sin(y), 0],
                  [np.sin(y),  np.cos(y), 0],
                  [        0,          0, 1]]) 
    R=yaw.dot(pitch.dot(roll))
    return R

# pinhole camera projection
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
def get_projection_matrix(p):
    [pitch,yaw,roll]= p
    [tx,ty,tz,alpha_u,alpha_v,u_0,v_0]=[0,0,0,1,1,0,0]
 
    intr=np.array([[alpha_u,     0.0,   u_0],
                   [    0.0, alpha_v,   v_0],
                   [    0.0,     0.0,   1.0]])
    
    R=get_rotation_matrix(roll,pitch,yaw) # roll pitch yaw 
    
    T=np.array([[tx],[ty],[tz]]) # tx ty tz   
    
    extr=np.hstack((R,T))

    P=intr.dot(extr)
    return P

def show(shape,landmarks_2D,P):
	s2=P.dot(shape)
	p2=P.landmarks_2D
	#plt.plot(s2[0],s2[1],'b.')
	plt.plot(s2[0,:],s2[1,:],'b+')
	plt.plot(p2[0],p2[1],'r.')
	plt.plot(p2[0,13],p2[0,13],'wo')
	#plt.axis([-150, 150, -150, 150]

def get_2D_landmarks(shape_param,rotation_param):

	# get shape and 3D landmarks from SFM
	model = eos.morphablemodel.load_model("3D-model/sfm_shape_3448.bin")
	shape = model.get_shape_model().draw_sample(shape_param)
	shape = np.array(shape)
	points_index=read_3D_points_index()

	# to homo cord
	shape=np.reshape(shape,(len(shape)/3,3)).T
	shape=np.vstack((shape,np.ones((1,shape.shape[1]))))
	landmarks_3D= shape[:,points_index]

	# projection 3D --> 2D
	 # pitch,yaw,roll
	P=get_projection_matrix(rotation_param)
	landmarks_2D=P.dot(landmarks_3D) # 13 center

	# show (for test)
	#show(shape,landmarks_2D,P)
	landmarks_2D=landmarks_2D[:2,:]
	return landmarks_2D

	  

def test():
	shape_param=[1.0, -0.5, 0.7]
	rotation_param=[0,0,0]

	get_2D_landmarks(shape_param,rotation_param)

def main():
	#fp=open('generated_landmarks/50_landmarks.txt','wb')
	examples_num=20*71*31*71
	landmarks_2D_list=np.zeros((examples_num,2*50))
	rotation_param_list=np.zeros((examples_num,3))
	count=0
	for k in range(20):
		mu=2*np.random.random_sample()-1 #[-1,1]
		sigma=np.random.random_sample()/2 #[0,0.5]
		shape_param = np.random.normal(mu, sigma, 63)
		# roll pitch yaw
		for r in (np.arange(71)-35):
			for p in (np.arange(31)-15):
				for y in (np.arange(71)-35):
					print count
					rotation_param=[r,p,y]
					landmarks_2D=get_2D_landmarks(shape_param,rotation_param)
					landmarks_2D=np.reshape(landmarks_2D.T,(2*50))
					#tmp=np.reshape(landmarks_2D.T,(2*50))
					#str1=' '.join(str(i) for i in tmp)
					#string=str1+' '+str(r)+' '+str(p)+' '+str(y)+'\n'
					#fp.writelines(string)
					landmarks_2D_list[count,:]=landmarks_2D
					rotation_param_list[count,:]=np.array(rotation_param)
					count+=1

	#fp.close()
	print '----save lists----'
	np.save('generated_landmarks/landmarks.npy',landmarks_2D_list)			
	np.save('generated_landmarks/rotation_param.npy',rotation_param_list)

if __name__ == '__main__':
	main()
