import numpy as np

def getCameraPoseFromViewMatrix(viewMatrix):
    '''
    Args:
    inputs:
        viewMatrix -> numpy.array(4, 4)
    output:
        transform -> numpy.array(4, 4), represent: camera_axis = transform @ world_axis
    '''
    # get camera pose from the view matrix
    # view_mat4x4 = np.array(viewMatrix).reshape(4, 4)
    flip_axis = np.array([[1., 0, 0, 0], 
                        [0, -1., 0, 0], 
                        [0, 0, -1., 0],
                        [0, 0, 0, 1.]])

    transform = viewMatrix @ flip_axis
    return transform
    # cameraPose = np.eye(4)
    # N = view_array[:3, 2].reshape(3, 1)
    # U = view_array[:3, 0].reshape(3, 1)
    # V = view_array[:3, 1].reshape(3, 1)
    # R = np.concatenate((U, V, N), axis=1)
    # T =  - R @ view_array[-1, :3]
    # cameraPose[:3, :3] = R.T
    # cameraPose[:3, -1] = T
    # N = np.array(args.cameraPos - args.cameraFocus)
    # U = np.cross(np.array(args.cameraVector), N)
    # V = np.cross(N, U)
    # return cameraPose

def transDepthBufferToRealZ(img_width, img_height, depthImg, projectionMatrix):
    '''
    Args:
    inputs:
        img_width -> the width of image from pybullet.getcameraImg() API
        img_height -> the height of image from pybullet.getCameraImg() API
        depthImg -> the depth image from pybullet.getCameraImg() API
        projectionMatrix -> numpy.array(4, 4)
    outputs:
        realDepthImg -> the real depth of rgba image 
    '''
    realDepthImg = depthImg.copy()

    A = projectionMatrix[2, 2]
    B = projectionMatrix[2, 3]
    for h in range(0, img_height):
        for w in range(0, img_width):
            x = (2*w - img_width)/img_width
            y = -(2*h - img_height)/img_height  # be careful！ deepth and its corresponding position
            z = 2*depthImg[h,w] - 1
            realDepthImg[h, w] = B / (z + A)
    return realDepthImg

def transPixelToWorldCoordinate(img_width, img_height, keypoints, depthImg, projectionMatrix, viewMatrix):
    '''
    Args:
    inputs:
        img_width -> the width of image from pybullet.getcameraImg() API
        img_height -> the height of image from pybullet.getCameraImg() API
        keypoints -> the keypoints in images, usually detected by opencv, so be careful about the xy coordinate.
        depthImg -> the depth image from pybullet.getCameraImg() API
        projectionMatrix -> numpy.array(4, 4)
        viewMatrix -> numpy.array(4, 4)
    outputs:
        keypointsWorldPos -> the keypoints in world coordinate
    '''
    keypointsWorldPos = []
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
    for point in keypoints:
        x = (2 * int(point[0]) - img_width)/img_width
        y = -(2 * int(point[1]) - img_height)/img_height  # be careful！ deepth and its corresponding position
        z = 2 * depthImg[int(point[1]), int(point[0])] - 1
        pixPos = np.asarray([x, y, z, 1])
        position = np.matmul(tran_pix_world, pixPos)
        pointWorldPos = (position / position[3])[:3]
        keypointsWorldPos.append(pointWorldPos)
    return np.array(keypointsWorldPos)


def transWorldToPixelCoordinate(worldPoints, intrinsicMatrix, extrinsicMatrix):
    homoIntrinsicMatrix = np.zeros([3, 4])
    homoIntrinsicMatrix[:3, :3] = intrinsicMatrix

    tran_world_pix = np.matmul(homoIntrinsicMatrix, extrinsicMatrix)
    keypointsImagePos = []
    for wp in worldPoints:
        keypointPos = np.zeros(2)
        homoPos = np.asarray([[wp[0]], [wp[1]], [wp[2]], [1]])
        pixPos = np.matmul(tran_world_pix, homoPos)
        pixPos = pixPos / pixPos[-1]
        keypointPos[0], keypointPos[1] = pixPos[0], pixPos[1]

        # keypointPos[0] = (pixPos[0]*img_width + img_width)/2
        # keypointPos[1] = (pixPos[0]*img_height + img_height)/2
        keypointsImagePos.append(keypointPos)
    
    return keypointsImagePos
