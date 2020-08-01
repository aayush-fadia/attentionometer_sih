import face_alignment
import numpy as np

# import matplotlib.pyplot as plt

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)


def get_face_keypoints(frame):
    return fa.get_landmarks(frame)


def calculate_attention(keypoints):
    if keypoints is None:
        return -1
    keypoints = keypoints[0]
    mean_pos = np.mean(keypoints, 0)
    nose_pos = (keypoints[31] + keypoints[34]) / 2
    orientation_diff = nose_pos - mean_pos
    # print("Attention:", orientation_diff[2] / np.linalg.norm(orientation_diff))
    return orientation_diff[2] / np.linalg.norm(orientation_diff)

# plt.ion()
# fig = plt.figure(figsize=plt.figaspect(.5))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122, projection='3d')


# def calculate_attention_and_plot(frame, keypoints):
#     if keypoints is None:
#         return
#     keypoints = keypoints[0]
#     ax1.cla()
#     ax2.cla()
#     ax1.imshow(frame)
#     ax1.plot(keypoints[0:17, 0], keypoints[0:17, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[17:22, 0], keypoints[17:22, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[22:27, 0], keypoints[22:27, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[27:31, 0], keypoints[27:31, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[31:36, 0], keypoints[31:36, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[36:42, 0], keypoints[36:42, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[42:48, 0], keypoints[42:48, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[48:60, 0], keypoints[48:60, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.plot(keypoints[60:68, 0], keypoints[60:68, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
#     ax1.axis('off')
#     ax2.plot3D(keypoints[:17, 0] * 1.2, keypoints[:17, 1], keypoints[:17, 2], color='blue')
#     ax2.plot3D(keypoints[17:22, 0] * 1.2, keypoints[17:22, 1], keypoints[17:22, 2], color='blue')
#     ax2.plot3D(keypoints[22:27, 0] * 1.2, keypoints[22:27, 1], keypoints[22:27, 2], color='blue')
#     ax2.plot3D(keypoints[27:31, 0] * 1.2, keypoints[27:31, 1], keypoints[27:31, 2], color='blue')
#     ax2.plot3D(keypoints[31:36, 0] * 1.2, keypoints[31:36, 1], keypoints[31:36, 2], color='blue')
#     ax2.plot3D(keypoints[36:42, 0] * 1.2, keypoints[36:42, 1], keypoints[36:42, 2], color='blue')
#     ax2.plot3D(keypoints[42:48, 0] * 1.2, keypoints[42:48, 1], keypoints[42:48, 2], color='blue')
#     ax2.plot3D(keypoints[48:, 0] * 1.2, keypoints[48:, 1], keypoints[48:, 2], color='blue')
#     mean_pos = np.mean(keypoints, 0)
#     nose_pos = (keypoints[31] + keypoints[34]) / 2
#     orientation_vect = np.stack([mean_pos, nose_pos])
#     ax2.plot3D(orientation_vect[:, 0] * 1.2, orientation_vect[:, 1], orientation_vect[:, 2], color='red')
#     ax2.view_init(elev=90., azim=90.)
#     ax2.set_xlim(ax2.get_xlim()[::-1])
#     plt.pause(0.00001)
#     orientation_diff = nose_pos - mean_pos
#     return orientation_diff[2] / np.linalg.norm(orientation_diff)
