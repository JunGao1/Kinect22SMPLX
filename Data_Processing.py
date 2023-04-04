from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)

def show_postion(joints):
    nums = 0
    joints_data = []
    while nums < PyKinectV2.JointType_Count: # 25
        x = joints[nums].Position.x
        y = joints[nums].Position.y
        z = joints[nums].Position.z
        print("x: ", x, " y: ", y, " z: ", z)
        nums += 1

if __name__ == '__main__':
    while True:
        if kinect.has_new_body_frame(): 
            _bodies = kinect.get_last_body_frame() 
        if _bodies is not None:
            for i in range(0, kinect.max_body_count): # 6
                body = _bodies.bodies[i] 
                if not body.is_tracked: 
                    continue
                joints = body.joints 
                show_postion(joints) 