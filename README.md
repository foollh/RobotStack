# RobotStack
stack objects through using pybullet

the demonstration result in the img folder

Simulation Environment:  
<img src="img/simulation_result.png" width = "50%" alt="注意" align=center />

File Instruction:
cameraCalibration.py
use opencv detects the center points of cubes, which take from pybullet. 
use cv2.solvePnP() API calibrates the camera extrinsics matrix, in order to prove the  correctness of transform between view matrix and extrinsics matrix  
 # remember comment out the tray urdf loader if you want to run this file  