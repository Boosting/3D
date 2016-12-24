Files  
============

python file  
-----------
1. generateLandmarks.py  
generate 2d landmarks using Surrey Face Model.  
please enter  
''' 
python generateLandmarks.py --help
'''
to see help  

2. common.py  
some common functions like read keypoints txt, landmarks normalize, etc.  

3. regression.py  
use tensorflow to train a regression networks with 2d landmarks as input and corresponding rotation parameters as output.   
please enter  
''' 
python regression.py --help
'''
to see help  

folders  
-----------
1. 3D-model  
save files for generating 2d landmarks.  

2. generated-landmarks  
save generated 2d landmarks here.  

3. TF-model  
save trained tensorflow model and outputresult.  

4. test-data  
save detected 2d landmarks in real scenarios, for testing.  

5. eos-maked
maked eos file. in eos-maked/bin, the *eos.so* is used for this project.  


Environment
==============
Basic env: numpy tensorflow  
Special env: eos (only used in generate landmarks)  

The eos.so has been maked, but for using it, must have the Needed dependencies:  
> Boost system (>=1.50.0), OpenCV core (>=2.4.3)

Also, you can use my computer to generate data:
IP: 192.168.50.82  
the whole project is saved in: /home/fengyao/github/3D  

If there is any question, please contact with me **after 01/09** !

Thanks!

