import numpy as np
import pandas as pd
np.set_printoptions(precision=3, suppress=True)

KINECT2_JOINTS = {
    'SpineBase': 0,
    'SpineMid': 1,
    'Neck': 2,
    'Head': 3,
    'ShoulderLeft': 4,
    'ElbowLeft': 5,
    'WristLeft': 6,
    'HandLeft': 7,
    'ShoulderRight': 8,
    'ElbowRight': 9,
    'WristRight': 10,
    'HandRight': 11,
    'HipLeft': 12,
    'KneeLeft': 13,
    'AnkleLeft': 14,
    'FootLeft': 15,
    'HipRight': 16,
    'KneeRight': 17,
    'AnkleRight': 18,
    'FootRight': 19,
    'SpineShoulder': 20,
    'HandTipLeft': 21,
    'ThumbLeft': 22,
    'HandTipRight': 23,
    'ThumbRight': 24
}

class Load_Data(object):
    
    def __init__(self, path = r"F:\2022_Fall\A_大创Prog\Previous\Kinect_Data_Raw\2022_04_15_15_29.npy") -> None:
        self.path = path
    
    def load_npy(self):
        kinect_data = np.load(self.path)
        return kinect_data
    
    def load_txt(self):
        kinect_data = np.loadtxt(self.path)
        kinect_data = kinect_data.reshape(1, 21, 3)
        return kinect_data

    def print_data(self, arr):
        print('type :', type(arr)) #原为self.load(), 但添加了load_txt()函数所以需要修改
        print('shape :', arr.shape)
        #print('data :')
        print('shape of one frame:', arr[0].shape)
    
    def create_duplicate(self):
        #创建形状为(2000, 21, 3)的numpy数组
        dup = self.load().copy()
        return dup


class Store(object):
    
    # 将数组存储为txt文件
    def to_txt(self, arr):
        with open('./datas/data.txt', 'w') as f:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    
                    if j == 0:
                        f.write('The ' + f'{i+1}' + 'th group: ---------------\n')
                    
                    # 在每行的行首添加行序号
                    row_num = j
                    f.write(f'{row_num}:\t')

                    # 将每3个数据存储为一行
                    row_data = [f'{x:.6f}' for x in arr[i, j]]
                    f.write(f'{row_data[0]}' + '   ' + f'{row_data[1]}' + '   ' + f'{row_data[2]}' + '\n')
    
    # 将数组存储为csv文件
    def to_csv(self, arr):
        arr_reshaped = arr.reshape((2000*21, 3))
        np.savetxt('./datas/data.csv', arr_reshaped, delimiter=',', fmt='%.6f')


class Kinect_2_SMPLX(object):
    
    #KINECT2_JOINTS is a dictionaty ///// KINECT2_JOINTS[name] -> integer
    def __init__(self, test1 = None, test2 = None) -> None:
        self.test_f = test1
        self.test_s = test2

    def father_and_son_trans(self, j_position_f, j_position_s, default = [0,1,0]):
        vector_before = np.array(default, dtype = np.float32)
        vector_after = j_position_s - j_position_f
        theta = np.arccos(np.dot(vector_before, vector_after) / (np.linalg.norm(vector_before) * np.linalg.norm(vector_after)))
        angle_deg = np.rad2deg(theta)
        rotate = np.cross(vector_before, vector_after) / np.linalg.norm(np.cross(vector_before, vector_after))
        #return [rotate[0], rotate[1], rotate[2], angle_deg]
        return np.array([rotate[0] * angle_deg, rotate[1] * angle_deg, rotate[2] * angle_deg])
    
    def test(self):
        #hipLeft = np.array([0.13384, -0.12943, 1.14159]); kneeLeft = np.array([-0.20588, -0.02905, 1.06783])
        elbowLeft = np.array([0.38278, 0.36449, 1.02764]); wristLeft = np.array([0.44955, 0.13521, 0.96022])
        Transformer = Kinect_2_SMPLX(elbowLeft, wristLeft)
        res = Transformer.father_and_son_trans(Transformer.test_f, Transformer.test_s,default=[1,0,0])
        print(res)
        #print(np.linalg.norm(x=res[:-1], ord=2))
    
    def recons_body(self, kinect_joint_position):
        torso = np.zeros((3,3), dtype = np.float32) 
        upper_limbs = np.zeros((6,3), dtype = np.float32) 
        lower = np.zeros((6,3), dtype = np.float32) 

        # lower[0,:] = self.father_and_son_trans(kinect_joint_position[0][12], kinect_joint_position[0][13])
        # lower[1,:] = self.father_and_son_trans(kinect_joint_position[0][16], kinect_joint_position[0][17])
        index_torso = [[KINECT2_JOINTS['Neck'], KINECT2_JOINTS['Head']], [KINECT2_JOINTS['SpineShoulder'], KINECT2_JOINTS['Neck']], 
                 [KINECT2_JOINTS['SpineMid'], KINECT2_JOINTS['SpineShoulder']]]
        index_upper = [[KINECT2_JOINTS['ShoulderLeft'], KINECT2_JOINTS['ElbowLeft']], [KINECT2_JOINTS['ShoulderRight'], KINECT2_JOINTS['ElbowRight']], 
                 [KINECT2_JOINTS['ElbowLeft'], KINECT2_JOINTS['WristLeft']], [KINECT2_JOINTS['ElbowRight'], KINECT2_JOINTS['WristRight']],
                 [KINECT2_JOINTS['WristLeft'], KINECT2_JOINTS['HandLeft']], [KINECT2_JOINTS['WristRight'], KINECT2_JOINTS['HandRight']]]
        index_lower = [[KINECT2_JOINTS['HipLeft'], KINECT2_JOINTS['KneeLeft']], [KINECT2_JOINTS['HipRight'], KINECT2_JOINTS['KneeRight']], 
                 [KINECT2_JOINTS['KneeLeft'], KINECT2_JOINTS['AnkleLeft']], [KINECT2_JOINTS['KneeRight'], KINECT2_JOINTS['AnkleRight']],
                 [KINECT2_JOINTS['AnkleLeft'], KINECT2_JOINTS['FootLeft']], [KINECT2_JOINTS['AnkleRight'], KINECT2_JOINTS['FootRight']]]
        
        for i in range(3):
            torso[i,:] = self.father_and_son_trans(kinect_joint_position[0][index_torso[i][0]], kinect_joint_position[0][index_torso[i][1]])
            upper_limbs[2*i,:] = self.father_and_son_trans(kinect_joint_position[0][index_upper[2*i][0]], kinect_joint_position[0][index_upper[2*i][1]], default=[-1,0,0])
            upper_limbs[2*i + 1,:] = self.father_and_son_trans(kinect_joint_position[0][index_upper[2*i + 1][0]], kinect_joint_position[0][index_upper[2*i + 1][1]], default=[1,0,0])
        for i in range(6):
            lower[i,:] = self.father_and_son_trans(kinect_joint_position[0][index_lower[i][0]], kinect_joint_position[0][index_lower[i][1]], default=[0,-1,0])
        # lower[6,:] = lower[4,:]
        # lower[7,:] = lower[5,:]
        body = np.vstack((torso,upper_limbs, lower))
        return body

if __name__ == '__main__':
    a = Kinect_2_SMPLX()
    a.test()
    
    # ld = Load_Data(path = r"F:\2022_Fall\A_大创Prog\coding\datas\pose.txt")
    # pose_data = ld.load_txt()
    # #ld.print_data(pose_data)
    # print(Kinect_2_SMPLX().recons_body(pose_data).shape)
    # print(Kinect_2_SMPLX().recons_body(pose_data))
    










