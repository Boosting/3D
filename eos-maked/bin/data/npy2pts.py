import numpy as np
import os

input='2.npy'
output='2.pts'

f_i=open(input,'r')
lines=f_i.readlines()
f_i.close()

f=open(output,'wb')
f.writelines('version: 1\n')
f.writelines('n_points:  68\n')
f.writelines('{\n')
for line in lines:
	[x,y]=line.strip().split('  ')
	s='%s %s\n' % (x, y)
	f.writelines(s)
f.writelines('}')
f.close()
