import face_alignment
from skimage import io


path = "C:/DPLMKData/FRAMES/"
user = "michael_anomal/"
test = "test/"


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

preds = fa.get_landmarks_from_directory(path + test)
print(preds)