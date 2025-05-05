import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class BodyLanguageAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_data = []
        self.timestamps = []
        
    def analyze_pose(self, frame):
        # BGR'den RGB'ye dönüştür
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose tespiti yap
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Vücut noktalarını çiz
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Vücut duruşunu analiz et
            landmarks = results.pose_landmarks.landmark
            
            # Omuz açısını hesapla
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Duruş analizi
            posture = self._analyze_posture(landmarks)
            
            # Hareket analizi
            movement = self._analyze_movement(landmarks)
            
            # Verileri kaydet
            self.pose_data.append({
                'posture': posture,
                'movement': movement,
                'shoulder_angle': self._calculate_shoulder_angle(left_shoulder, right_shoulder)
            })
            self.timestamps.append(datetime.now())
            
            # Analiz sonuçlarını ekrana yaz
            cv2.putText(frame, f"Posture: {posture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Movement: {movement}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def _analyze_posture(self, landmarks):
        # Omuz pozisyonlarını al
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Omuz yüksekliği farkını hesapla
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        
        if shoulder_height_diff > 0.1:
            return "Asimetrik Duruş"
        else:
            return "Düzgün Duruş"
    
    def _analyze_movement(self, landmarks):
        # Hareket analizi için gerekli noktaları al
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # El hareketlerini analiz et
        if left_wrist.y < landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y:
            return "Eller Yukarıda"
        elif left_wrist.y > landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y:
            return "Eller Aşağıda"
        else:
            return "Normal Pozisyon"
    
    def _calculate_shoulder_angle(self, left_shoulder, right_shoulder):
        # Omuz açısını hesapla
        return np.arctan2(right_shoulder.y - left_shoulder.y,
                         right_shoulder.x - left_shoulder.x) * 180 / np.pi
    
    def visualize_results(self):
        if not self.pose_data:
            return
        
        # Zaman serisi grafiği oluştur
        plt.figure(figsize=(12, 6))
        
        # Omuz açısı grafiği
        angles = [data['shoulder_angle'] for data in self.pose_data]
        plt.plot(range(len(angles)), angles, label='Omuz Açısı')
        
        plt.title('Vücut Dili Analizi')
        plt.xlabel('Zaman')
        plt.ylabel('Açı (Derece)')
        plt.legend()
        plt.grid(True)
        
        # Grafiği kaydet
        plt.savefig('body_language_analysis.png')
        plt.close()
    
    def get_statistics(self):
        if not self.pose_data:
            return "Veri yok"
        
        # İstatistikleri hesapla
        postures = [data['posture'] for data in self.pose_data]
        movements = [data['movement'] for data in self.pose_data]
        
        stats = {
            'posture_counts': {posture: postures.count(posture) for posture in set(postures)},
            'movement_counts': {movement: movements.count(movement) for movement in set(movements)},
            'total_frames': len(self.pose_data)
        }
        
        return stats 