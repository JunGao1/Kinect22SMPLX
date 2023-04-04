import numpy as np

# 定义SMPL模型关节名称和序号
SMPL_JOINTS = {
    'root': 0,
    'left_hip': 12,
    'right_hip': 16,
    'spine1': 1,
    'left_knee': 13,
    'right_knee': 17,
    'spine2': 2,
    'left_ankle': 14,
    'right_ankle': 18,
    'spine3': 3,
    'left_foot': 15,
    'right_foot': 19,
    'neck': 4,
    'left_collar': 5,
    'right_collar': 9,
    'head': 6,
    'left_shoulder': 7,
    'right_shoulder': 11,
    'left_elbow': 8,
    'right_elbow': 10,
    'left_wrist': 20,
    'right_wrist': 24,
    'left_hand': 21,
    'right_hand': 25,
}

# 定义SMPL模型关节之间的父子关系
SMPL_PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
]

# 定义Kinect坐标系和SMPL坐标系的转换矩阵
KINECT_TO_SMPL_MATRIX = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
])

# 读取Kinect骨骼数据
kinect_data = np.load('kinect_data.npy')

# 转换坐标系
kinect_data[:, :, :3] = np.matmul(kinect_data[:, :, :3], KINECT_TO_SMPL_MATRIX)

# 匹配关节名称
smpl_data = np.zeros((kinect_data.shape[0], len(SMPL_JOINTS), 3))
for j, name in enumerate(SMPL_JOINTS.keys()):
    if name in ['root', 'left_foot', 'right_foot']:
        # 如果是根节点或末端节点，直接拷贝坐标
        smpl_data[:, j, :] = kinect_data[:, SMPL_JOINTS[name], :3]
    else:
        # 如果是中间节点，取其与父节点的中点作为坐标
        parent_index = SMPL_PARENTS[j]
        smpl_data[:, j, :] = (kinect_data[:, j, :3] + kinect_data[:, parent_index, :3]) / 2

# 保存SMPL模型格式的数据
np.save('smpl_data.npy', smpl_data)
