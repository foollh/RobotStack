# RobotStack
stack objects through using pybullet

the demonstration result in the img folder

Simulation Environment:  
<img src="img/simulation_result.png" width = "50%" alt="注意" align=center />

_File Instruction:_  
-- cameraCalibration.py  
&emsp;&emsp;use opencv detects the center points of cubes, which take from pybullet.  
&emsp;&emsp;use cv2.solvePnP() API calibrates the camera extrinsics matrix, in order to prove the  correctness of transform between view matrix and extrinsics matrix   
&emsp;&emsp;remember comment out the tray urdf loader if you want to run this file.  