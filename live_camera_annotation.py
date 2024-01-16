import os
import random
import cv2
import tensorflow as tf
import numpy as np
import time

TFLITE_MODEL_PATH_PERSON_DETECTOR = "./Model/efficientnet_person_detection.tflite"

TFLITE_MODEL_PATH_SIAMESE = './Model/siamese_model.tflite'

interpreter_person_detector = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH_PERSON_DETECTOR)
interpreter_person_detector.allocate_tensors()

interpreter_siamese = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH_SIAMESE)
interpreter_siamese.allocate_tensors()

input_details_siamese = interpreter_siamese.get_input_details()
output_details_siamese = interpreter_siamese.get_output_details()



def display_image_with_boxes(image, boxes, scores, ids):

    img_height, img_width, _ = image.shape

    annotated_image = np.copy(image)

    for i, (box, score, obj_id) in enumerate(zip(boxes, scores, ids)):
        ymin, xmin, ymax, xmax = box
        ymin = int(ymin * img_height)
        xmin = int(xmin * img_width)
        ymax = int(ymax * img_height)
        xmax = int(xmax * img_width)

        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        text = f'Score: {score.item():.2f}, ID: {obj_id}'
        cv2.putText(annotated_image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    return annotated_image
    

def predict_and_filter(input_tensor):
    interpreter_person_detector.set_tensor(interpreter_person_detector.get_input_details()[0]['index'], input_tensor)
    interpreter_person_detector.invoke()

    output_details = interpreter_person_detector.get_output_details()

    boxes = interpreter_person_detector.get_tensor(output_details[0]['index'])
    scores = interpreter_person_detector.get_tensor(output_details[2]['index'])
    classes = interpreter_person_detector.get_tensor(output_details[1]['index'])

    class_1_indices = np.where(classes[0] == 0)
    class_1_boxes = boxes[0][class_1_indices]
    class_1_scores = scores[0][class_1_indices]

    confidence_threshold = 0.50
    selected_indices = np.where(class_1_scores > confidence_threshold)
    selected_boxes = class_1_boxes[selected_indices]
    selected_scores = class_1_scores[selected_indices]

    return selected_boxes, selected_scores


def resize_and_pad_image(img, target_size=(64, 64)):
    # this function resizes the image to the target size and pads it with zeros to keep the aspect ratio
    
    if img.size == 0:
        return np.zeros(target_size, dtype=np.uint8)

    height, width, _ = img.shape

    target_height, target_width = target_size
    aspect_ratio = width / height

    if width == height:
        padded_img = cv2.resize(img, target_size)
    else:
        if aspect_ratio > 1:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_width = int(target_height * aspect_ratio)
            new_height = target_height

        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        top_pad = (target_height - new_height) // 2
        bottom_pad = target_height - new_height - top_pad
        left_pad = (target_width - new_width) // 2
        right_pad = target_width - new_width - left_pad

        padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_img

from picamera2 import Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320,240)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

person_history = []
target_size = (320, 320)

# Iterate through the frames
while True:
    # Capture a frame from the camera
    start_time = time.time()
    im = picam2.capture_array()
    frame = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = resize_and_pad_image(frame_rgb, target_size)
    input_image = np.expand_dims(input_image.astype(np.uint8), axis=0)
    
    tf_boxes, tf_scores = predict_and_filter(input_image)
    end_detection_time = time.time()
    pred_per_id = []


    for box in tf_boxes:
        ymin, xmin, ymax, xmax = box
        ymin = int(ymin * 320)
        xmin = int(xmin * 320)
        ymax = int(ymax * 320)
        xmax = int(xmax * 320)
        cropped_box = input_image[0][ymin:ymax, xmin:xmax]
        scaled_person_img = resize_and_pad_image(cropped_box)

        if len(person_history) == 0:
            person_history.append([np.expand_dims(scaled_person_img.astype(np.float32), axis=0)])
            pred_per_id.append(0)
        else:
            # Calculate the similarity between the current person and the persons in the history
            # and check if the current person is already in the history
            similarity = []
            input_a_data = scaled_person_img.astype(np.float32) / 255.0
            input_a_data = np.expand_dims(input_a_data, axis=0)
            for person_id in person_history:
                current_person_preds = []
                for person in person_id:
                    # Calculate the similarity between the current person and the persons in the history
                    interpreter_siamese.set_tensor(input_details_siamese[0]['index'], input_a_data)
                    interpreter_siamese.set_tensor(input_details_siamese[1]['index'], person)
                    interpreter_siamese.invoke()

                    output = interpreter_siamese.get_tensor(output_details_siamese[0]['index'])
                    pred = np.where(output[0][0] > 0.5, 1, 0)
                    current_person_preds.append(pred)
                similarity.append(current_person_preds)

            max_images_per_person_id = 12
            similarity_threshold = 0.668
            added = False

            for i, person_id in enumerate(similarity):
                avg_similarity = sum(person_id) / len(person_id)
                if avg_similarity > similarity_threshold:
                    added = True
                    pred_per_id.append(i)
                    if len(person_history[i]) < max_images_per_person_id:
                        person_history[i].append(input_a_data)
                    else:
                        random_image_index = random.randint(0, len(person_history[i]) - 1)
                        person_history[i][random_image_index] = input_a_data
                    break

            if not added:
                person_history.append([input_a_data])
                pred_per_id.append(len(person_history) - 1)

        end_similarity_time = time.time()
        
        detection_time = end_detection_time - start_time 
        similarity_time = end_similarity_time - end_detection_time 
        total_time = end_similarity_time - start_time
        
        print(f"Detection Time: {detection_time:.4f}s, Similarity Time: {similarity_time:.4f}s, Total Time: {total_time:.4f}s")
        
        box_image = display_image_with_boxes(input_image[0], tf_boxes, tf_scores, pred_per_id)
        cv2.imshow("Camera", box_image)
        if cv2.waitKey(1)==ord('q'):
            break

# Stop the camera and close all windows
picam2.stop()
cv2.destroyAllWindows()
