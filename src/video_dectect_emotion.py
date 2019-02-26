from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_text_pos
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import detect_face
#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.7, 0.8, 0.9 ]  # three steps's threshold
factor = 0.709 # scale factor


# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 1
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './models')


c = 0
time_F = 5
# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    if(c % time_F == 0):
    #    faces = detect_faces(face_detection, gray_image)
        faces, _ = detect_face.detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor) 
    
        for face_position in faces:
            
            face_position=face_position.astype(int)
    #        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            x1, y1, x2, y2 = face_position[0], face_position[1], face_position[2], face_position[3]
    #        print (y1, x1, y2, x2)
    
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
    
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)
    
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue
    
            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))
    
            color = color.astype(int)
            color = color.tolist()
            
            cv2.rectangle(rgb_image, (face_position[0], 
                face_position[1]), (face_position[2], 
                face_position[3]), color, 2)
     
    #        draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text_pos(face_position, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)
    
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    c += 1


video_capture.release()
cv2.destroyAllWindows()

