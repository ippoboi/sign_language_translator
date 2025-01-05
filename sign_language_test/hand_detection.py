import cv2
import mediapipe as mp
import numpy as np
import time

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.fps_time = time.time()
        self.fps = 0

    def find_hands(self, img, draw=True):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.fps_time)
        self.fps_time = current_time

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        
        return img

    def find_positions(self, img):
        hand_positions = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                positions = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    positions.append([id, cx, cy])
                hand_positions.append(positions)
        return hand_positions

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Find hands
        img = detector.find_hands(img)
        
        # Get hand positions
        positions = detector.find_positions(img)
        
        # Display FPS
        cv2.putText(img, f'FPS: {int(detector.fps)}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2)
        
        # Show the image
        cv2.imshow("Hand Detection", img)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()