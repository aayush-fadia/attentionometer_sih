import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

def get_face_keypoints(frame):
    return fa.get_landmarks(frame)
