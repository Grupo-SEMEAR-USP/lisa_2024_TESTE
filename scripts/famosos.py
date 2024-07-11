#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
import cv2
from cv_bridge import CvBridge
import numpy as np
import face_recognition
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from joblib import load

class CompareWithCelebritiesService:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/Imagens', Image, self.image_callback)
        self.current_image = None
        self.images = []
        self.capture_count = 5  # Número de frames a capturar

        # Carregar modelos
        self.face_net, self.gender_net = self.load_face_and_gender_models()
        self.fair_face_model, self.device = self.load_fair_face_model()
        self.clf = load('/home/murilo/lisa/lisa_desktop/modelo_svm_face_recognition_combined.joblib')

    def load_fair_face_model(self):
        fair_face_model_path = '/home/murilo/lisa/lisa_desktop/res34_fair_align_multi_4_20190809.pt'
        device = torch.device("cpu")
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(fair_face_model_path, map_location=device))
        model.eval()
        return model, device

    def load_face_and_gender_models(self):
        face_pbtxt = "/home/murilo/lisa/lisa_desktop/Gender-and-Age-Detection-/opencv_face_detector.pbtxt"
        face_pb = "/home/murilo/lisa/lisa_desktop/Gender-and-Age-Detection-/opencv_face_detector_uint8.pb"
        gender_prototxt = "/home/murilo/lisa/lisa_desktop/Gender-and-Age-Detection-/gender_deploy.prototxt"
        gender_model = "/home/murilo/lisa/lisa_desktop/Gender-and-Age-Detection-/gender_net.caffemodel"
        
        face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
        gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)
        
        return face_net, gender_net

    def image_callback(self, msg):
        if len(self.images) < self.capture_count:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.images.append(self.current_image)
            rospy.loginfo("Image captured")

    def handle_compare(self, req):
        if len(self.images) < self.capture_count:
            return TriggerResponse(success=False, message="Not enough images captured")

        results = []
        for image in self.images:
            result, probability = self.compare_faces(image)
            results.append((result, probability))

        # Calcular a média dos resultados
        most_likely = self.calculate_average_result(results)

        self.images = []  # Limpar as imagens após a comparação

        return TriggerResponse(success=True, message=f"Parecido com {most_likely[0]}")

    def calculate_average_result(self, results):
        results_dict = {}
        for result, prob in results:
            if result not in results_dict:
                results_dict[result] = []
            results_dict[result].append(prob)
        
        average_results = {k: np.mean(v) for k, v in results_dict.items()}
        most_likely = max(average_results.items(), key=lambda x: x[1])
        
        return most_likely

    def compare_faces(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_encodings:
            return "No faces found", 0.0

        encoding = face_encodings[0]
        face_bounds = self.detect_faces(image)

        if not face_bounds:
            return "No face detected by OpenCV model", 0.0

        x1, y1, x2, y2 = face_bounds[0]
        cropped_face = image[max(0, y1-15):min(y2+15, image.shape[0]-1), max(0, x1-15):min(x2+15, image.shape[1]-1)]

        gender = self.predict_gender(cropped_face)
        race_result = self.classify_race([cropped_face])
        race_probs = race_result[0]
        top_race = max(race_probs, key=race_probs.get)

        feature_vector = np.concatenate([encoding, [gender == 'Mulher'], [race_probs['White']], [race_probs['Black']]])

        filtered_predictions = []
        predictions = self.clf.predict_proba([feature_vector])[0]
        for prob, label in zip(predictions, self.clf.classes_):
            label_parts = label.rsplit('_', 2)
            label_name = label_parts[0]
            label_race = label_parts[1]
            label_gender = label_parts[2]
            if gender == label_gender and top_race == label_race:
                filtered_predictions.append((prob, label_name))

        if not filtered_predictions:
            return "No matching famous person found", 0.0

        best_match = min(filtered_predictions, key=lambda x: x[0])
        return best_match[1], float(best_match[0])

    def detect_faces(self, image):
        img_h, img_w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 117, 123), True, False)
        self.face_net.setInput(blob)
        detected_faces = self.face_net.forward()
        
        face_bounds = []
        for i in range(detected_faces.shape[2]):
            confidence = detected_faces[0,0,i,2]
            if confidence > 0.7:
                x1 = int(detected_faces[0,0,i,3] * img_w)
                y1 = int(detected_faces[0,0,i,4] * img_h)
                x2 = int(detected_faces[0,0,i,5] * img_w)
                y2 = int(detected_faces[0,0,i,6] * img_h)
                face_bounds.append([x1, y1, x2, y2])
        return face_bounds

    def predict_gender(self, face_img):
        MODEL_MEAN_VALUES = (104, 117, 123)
        gender_classes = ['Homem', 'Mulher']
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, True)
        
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = gender_classes[gender_preds[0].argmax()]
        
        return gender

    def classify_race(self, face_images):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        labels = ['White', 'Black']
        results = []

        for face in face_images:
            img = preprocess(face).unsqueeze(0).to(self.device)
            outputs = self.fair_face_model(img).cpu().detach().numpy().squeeze()
            race_probs = np.exp(outputs[:2]) / np.sum(np.exp(outputs[:2]))  # Consider only White and Black
            race_probs_dict = {labels[i]: float(race_probs[i]) for i in range(len(race_probs))}
            results.append(race_probs_dict)

        return results

    def start_service(self):
        rospy.init_node('compare_with_celebrities_service')
        service = rospy.Service('/compare_with_celebrities', Trigger, self.handle_compare)
        rospy.spin()

if __name__ == '__main__':
    service = CompareWithCelebritiesService()
    service.start_service()
