import torch
from igibson.envs.igibson_env import iGibsonEnv
import yaml
import cv2 as cv
import numpy as np
from copy import deepcopy
import math
from pre_conv_net import PreConvNet
import lpips
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from torch_geometric.data import Data
import torch_geometric

config_data = yaml.load(open('turtlebot_nav.yaml', "r"), Loader=yaml.FullLoader)
scene = ['Beechwood_0_int', 'Benevolence_0_int', 'Benevolence_1_int', 'Benevolence_2_int', 'Ihlen_0_int', 'Merom_0_int',
         'Merom_1_int', 'Pomaria_1_int', 'Pomaria_2_int ', 'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int']
scene_id = 'Beechwood_0_int'

env = iGibsonEnv(config_file=deepcopy(config_data), scene_id=scene_id, mode="gui_non_interactive")

q = None
# odd number
mask_size = 15
num_episodes = 1

def action(type):
    global q
    if type == "random":
        return env.action_space.sample()
    elif type == "control":
        if q == ord("w"):
            return np.array([1.2, 0.0])
        elif q == ord("a"):
            return np.array([0, -0.3])
        elif q == ord("d"):
            return np.array([0, 0.3])
        else:
            return np.array([0.0, 0.0])


def ori_matrix(obv, orientation):
    # 设置一个0矩阵，用于存放机器人朝向角
    matrix = np.zeros((obv.shape[0], obv.shape[1]))
    # 计算朝向角在矩阵中对应的位置，并将对应列赋为1
    ori_index = round((2 * math.pi - (orientation + math.pi)) / (2 * math.pi) * (obv.shape[1] - 1))
    # print(ori_index)
    if ori_index < (mask_size - 1) / 2:
        # ori_mat[:, 0:15] = 1
        matrix[:, 0:ori_index + 1] = 1
        matrix[:, ori_index + obv.shape[1] - mask_size + 1:] = 1
    elif ori_index > obv.shape[1] - (mask_size + 1) / 2:
        # ori_mat[:, 945:] = 1
        matrix[:, ori_index:] = 1
        matrix[:, 0:ori_index - obv.shape[1] + mask_size] = 1
    else:
        matrix[:, int(ori_index - (mask_size - 1) / 2):int(ori_index + (mask_size + 1) / 2)] = 1
    matrix = matrix[:, :, np.newaxis]

    return matrix


def obv_pre_process(pos_x, pos_y, orientation):
    camera_pose = np.array([pos_x, pos_y, 1.2])
    # 设置相机朝向与机器人朝向一致
    view_direction = np.array([1, orientation, 0])
    env.simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

    panoramic = env.simulator.renderer.get_equi(mode="rgb", use_robot_camera=False)
    panoramic = np.array(panoramic)

    matrix = ori_matrix(panoramic, orientation)
    result = np.concatenate((panoramic, matrix), axis=2)
    result = result[np.newaxis, :, :, :].transpose((0, 3, 1, 2))
    result = torch.Tensor(result)

    return panoramic, matrix, result


pre_conv = PreConvNet()
metrics = lpips.LPIPS(net='alex')

gt_map = np.array([[i for i in j] for j in env.scene.floor_map[env.task.floor_num]])

# initialization
env.reset()
init_pos = env.robots[0].get_position()[:2]
init_ori = env.robots[0].get_rpy()[-1:-2:-1][0]
print(init_ori)
init_x = init_pos[0]
init_y = init_pos[1]
init_obv = obv_pre_process(init_x, init_y, init_ori)[2]
init_emb = pre_conv(init_obv)

G = nx.Graph()
G.add_node(0)
memory = [init_emb]
node_pos = {0: [init_x, init_y]}
last_loc_node = 0

plt.ion()
plt.figure()

for episode in range(num_episodes):
    for t in itertools.count():
        q = cv.waitKey(1)
        env.step(action(type='control'))

        yaw = env.robots[0].get_rpy()[-1:-2:-1]
        pos = env.robots[0].get_position()[:2]

        x = pos[0]
        y = pos[1]
        ori = yaw[0]
        # print(ori)
        pano, ori_mat, pano_ori = obv_pre_process(x, y, ori)
        pano = pano[:, :, 2::-1]
        current_emb = pre_conv(pano_ori)
        cv.imshow('Panoramic', pano)
        cv.imshow('Orientation', ori_mat)
        index_x = int(gt_map.shape[0] / 2 + 10 * x)
        index_y = int(gt_map.shape[1] / 2 + 10 * y)
        gt_map[index_y, index_x] = 0

        # localization
        dist_set = []
        is_Localized = True
        threshold = 0.4

        for index, emb in enumerate(memory):
            dist = metrics(current_emb, emb)
            dist = torch.squeeze(dist).detach().numpy()
            dist_set.append(dist)
        dist_set = np.array(dist_set)
        # print(dist_set)
        min_dist = np.sort(dist_set)[0]
        # print(np.sort(dist_set))
        nearest_node = np.argsort(dist_set)[0]
        # print(np.argsort(dist_set))

        if min_dist > threshold:
            is_Localized = False

        # graph update
        if is_Localized:
            if nearest_node != last_loc_node:
                G.add_edge(nearest_node, last_loc_node)
                print("Current localized at Node {}. A new edge is connected between it and Node {}!".
                      format(nearest_node, last_loc_node))
                last_loc_node = nearest_node
            else:
                print("Current localized at Node {}. No new edge is added!".format(nearest_node))
                # last_loc_node = nearest_node
        else:
            print("Not localized in current graph. A new node is created, from which a new edge is connected to Node "
                  "{}!".format(last_loc_node))
            G.add_node(G.number_of_nodes())
            new_node = G.number_of_nodes() - 1
            G.add_edge(last_loc_node, new_node)
            memory.append(current_emb)
            node_pos[new_node] = [x, y]
            # print(node_pos)
            last_loc_node = new_node

        # plot the graph
        plt.subplot(121)
        plt.imshow(gt_map)

        plt.subplot(122)

        node_color_map = ["black"]
        node_color_map.extend(["black"] * (G.number_of_nodes() - 1))

        edge_color_map = ["#ffc20e"]
        edge_color_map.extend(["#ffc20e"] * (G.number_of_nodes() - 1))

        nx.draw_networkx_nodes(G, node_pos, node_size=100, node_color=node_color_map)
        nx.draw_networkx_edges(G, node_pos, width=3, edge_color=edge_color_map)
        nx.draw_networkx_labels(G, node_pos, font_size=10, font_color='w')

        plt.show()
        plt.pause(0.1)

        if q == ord("t"):
            cv.destroyAllWindows()
            break

env.close()
