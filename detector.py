import os
import cv2
from tensorflow.keras.models import load_model
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# load the trained face and mask models
def load_models(face_path, mask_path):
    face_model = cv2.dnn.readNet(os.path.join(face_path, 'res10_300x300_ssd_iter_140000.caffemodel'),
                                 os.path.join(face_path, 'deploy.prototxt'))
    mask_model = load_model(mask_path)
    return face_model, mask_model

# Using each frames from WEBCAM, predict the face with face model and pass the
# face model output as input to the mask model to predict Mask or NoMask
def predict(image, face_mod, mask_mod):
    h, w = image.shape[:2]
    out = cv2.dnn.blobFromImage(image, 1, (224, 224))
    out = face_mod.setInput(out)
    det = face_mod.forward(out)
    face = []
    loc = []
    conf = []
    for con in range(0, det.shape[2]):
        confidence = det[:, :, con, 2]
        if confidence > 0.5:
            box = det[0, 0, con, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype('int')
            x1, x2, y1, y2 = max(0, x1), max(0, x2), max(0, y1), max(0, y2)
            frame_face = image[y1:y2, x1:x2]
            frame_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            frame_face = cv2.resize(frame_face, (224, 224))
            frame_face = img_to_array(frame_face)
            mask_input = preprocess_input(frame_face)
            face.append(mask_input)
            loc.append((x1, y1, x2, y2))
            conf.append(confidence)
    if len(face) > 0:
        for i in range(0, len(face)):
            inp = np.array(face, 'float32')
            prediction = mask_mod.predict(inp)
            cat = ['Mask', 'No Mask']
            classes = cat[int(np.argmax(prediction[i]))]
            colour = [0, 0, 255] if classes == 'No Mask' else [0, 255, 0]
            cv2.rectangle(image, (loc[i][0], loc[i][1]), (loc[i][2], loc[i][3]), colour, 2)
            cv2.putText(image, classes + ' ' + str(round(float(conf[i][0]) * 100, 2)), (loc[i][0], loc[i][1] - 3),
                        cv2.FONT_HERSHEY_PLAIN, 1, colour, 2)
            cv2.imshow('frame', frame)


face_path1 = 'face_model'
mask_path1 = 'Mask_detector_model'
face_model1, mask_model1 = load_models(face_path1, mask_path1)


cap = cv2.VideoCapture(0)
# to save each frame as a video in avi format, uncomment line 59, 60, 67 and 74
# four = cv2.VideoWriter_fourcc(*'MJPG')
# writer = cv2.VideoWriter('sione_mask_detection.avi', four, 10, (600, 450))


# Creating While loop to predict each frame from WEBCAM
while True:
    success, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)
    predict(frame, face_model1, mask_model1)
    # writer.write(frame)
    cv2.waitKey(1) & 0xFF
    if cv2.waitKey(1) == 27:
        break
    # quit prediction by pressing 'q'
    if ord('q') == cv2.waitKey(1):
        break

# writer.release()
cap.release()
