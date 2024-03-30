import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
import matplotlib.ticker as mtick
from PIL import Image
# df = pd.read_csv("run-.-tag-train_loss env1.csv")
# # print(df)

# ts_factor = 0.99

# smooth = df.ewm(alpha=(1 - ts_factor)).mean()

# plt.plot(df["Step"], df["Value"], alpha=0.4, label='Original')
# plt.plot(df["Step"], smooth["Value"], label='Smoothing:{}'.format(ts_factor))
# plt.title("Loss Curve")
# plt.xlabel("Step")
# plt.ylabel("Loss Value")
# plt.grid(alpha=0.3)
# plt.legend()
# plt.show()

# x = np.arange(-10, 10, 0.1)
# y = np.maximum(x,0)
# plt.plot(x, y)
# plt.title("ReLU Activation Function")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# plt.figure()
# fig, ax = plt.subplots(2, 2, figsize=(5,5))
# # plt.subplot(2,4,1)
# i7 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/rgb_1694.png")
# # plt.subplot(2,4,2)
# i8 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/Panoramic_1694.png")
# # plt.imshow(i2)
# # plt.subplot(2,4,3)
# i9 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/Mat_1694.png")
# i4 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/rgb_1856.png")
# # plt.subplot(2,4,2)
# i5 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/Panoramic_1856.png")
# # plt.imshow(i2)
# # plt.subplot(2,4,3)
# i6 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/Mat_1856.png")
# i1 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/rgb_2088.png")
# # plt.subplot(2,4,2)
# i2 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/Panoramic_2088.png")
# # plt.imshow(i2)
# # plt.subplot(2,4,3)
# i3 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/pics/Mat_2088.png")
# # # plt.imshow(i3)
# # # plt.subplot(2,4,4)
# # i4 = matplotlib.image.imread("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/test_env/4_floor_trav_0.png")
# # plt.suptitle("Evaluation Environments")
# # ax[0, 0].imshow(i1)
# # ax[0, 1].imshow(i2)
# # ax[1, 0].imshow(i3)
# # ax[1, 1].imshow(i4)
# # ax[0, 0].set_title("Ihlen_0_int")
# # ax[0, 1].set_title("Rs_int")
# # ax[1, 0].set_title("Wainscott_0_int")
# # ax[1, 1].set_title("Wainscott_1_int")

# # fig.tight_layout()
# # plt.subplots_adjust(hspace=0.2,wspace=0.2)
# # for i in range(2):
# #     for j in range(2):
# #         ax[i, j].set_xticks([])
# #         ax[i, j].set_yticks([])
# # plt.show()

# plt.figure(figsize=(10,6))
# grid = plt.GridSpec(3,5)
# plt.subplot(grid[0,0])
# plt.imshow(i1)

# plt.subplot(grid[0,1:3])
# plt.imshow(i2)

# plt.subplot(grid[0,3:5])
# plt.imshow(i3)

# plt.subplot(grid[1,0])
# plt.imshow(i4)

# plt.subplot(grid[1,1:3])
# plt.imshow(i5)

# plt.subplot(grid[1,3:5])
# plt.imshow(i6)

# plt.subplot(grid[2,0])
# plt.imshow(i7)

# plt.subplot(grid[2,1:3])
# plt.imshow(i8)

# plt.subplot(grid[2,3:5])
# plt.imshow(i9)
# plt.suptitle('Observation Sequence Sampling')
# plt.show()

# steps = list(range(1, 25001))
# o1 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-Ours_envs-['Ihlen_0_int']_a-10_t-100.npy").tolist()
# o2 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-Ours_envs-['Rs_int']_a-10_t-100.npy").tolist()
# o3 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-Ours_envs-['Wainscott_0_int']_a-10_t-100.npy").tolist()
# o4 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-Ours_envs-['Wainscott_1_int']_a-10_t-100.npy").tolist()
# s1 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-SPTM_envs-['Ihlen_0_int']_a-10_t-100.npy").tolist()
# s2 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-SPTM_envs-['Rs_int']_a-10_t-100.npy").tolist()
# s3 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-SPTM_envs-['Wainscott_0_int']_a-10_t-100.npy").tolist()
# s4 = np.load("C:/Users/Muriate_C/Desktop/exp/acc_model-SPTM_envs-['Wainscott_1_int']_a-10_t-100.npy").tolist()

# list = [[o1, o2, o3, o4], [s1, s2, s3, s4]]


# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
# plt.suptitle("Similarity Estimator Accuracy")
# for i in range(2):
#     for j in range(2):
#         ax[i, j].plot(steps, list[0][i * 2 + j], label="Ours")
#         ax[i, j].text(steps[24999], list[0][i * 2 + j][24999],'{:.2%}'.format(list[0][i * 2 + j][24999]), color='r')
#         ax[i, j].plot(steps, list[1][i * 2 + j], label="SPTM")
#         ax[i, j].text(steps[24999], list[1][i * 2 + j][24999],'{:.2%}'.format(list[1][i * 2 + j][24999]), color='r')
#         if ax[i, j].is_last_row():
#             ax[i, j].set_xlabel('number of samples')
#         if ax[i, j].is_first_col():
#             ax[i, j].set_ylabel('accuracy')
#         ax[i, j].set_title("Environment: {}".format(env_list[i * 2 + j]))

# plt.xlim((1,25001))
# plt.ylim((0,1))
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# # fig.tight_layout()
# plt.subplots_adjust(top=0.85,wspace=0.4)
# lines, labels = fig.axes[-1].get_legend_handles_labels()
# fig.legend(lines, labels, loc = 'upper right')
# plt.show()
env_list = ['Ihlen_0_int', 'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int']
env = 3
g1 = Image.open("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/topomap_th0.04/e{}_gt.png".format(env+1))
# g1 = g1.resize((480,640))
t1 = Image.open("C:/Users/Muriate_C/Desktop/graduate_design/thesis/images/topomap_th0.04/e{}_tm.png".format(env+1))
w, h = t1.size
print(w, h)
t1 = t1.resize((480, 960))

fig, ax = plt.subplots(1, 2, figsize=(8, 7.5))
plt.suptitle("Online Topological Map Construction \nin Environment '{}'".format(env_list[env]))
ax[0].imshow(g1)
ax[1].imshow(t1)
ax[0].set_title("Groundtruth Trajectory")
ax[1].set_title("Topological Map")
for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
# plt.axis('off')
fig.tight_layout()
plt.subplots_adjust(left=0, right=1,hspace=0)
plt.show()