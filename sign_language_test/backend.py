from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import asyncio
from typing import List
import time

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe with optimized settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lower this if you want faster but less accurate detection
    min_tracking_confidence=0.5,   # Lower this if you want faster but less accurate tracking
    model_complexity=0  # Use simplest model for faster processing
)

class FrameRateController:
    def __init__(self, target_fps=15):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_frame_time = time.time()
        self.current_fps = 0
        self.smoothing = 0.9  # FPS smoothing factor

    async def wait_for_next_frame(self):
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Update FPS
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.current_fps = (self.smoothing * self.current_fps + 
                              (1.0 - self.smoothing) * instant_fps)

        # If we're running too fast, wait
        if elapsed < self.frame_time:
            await asyncio.sleep(self.frame_time - elapsed)
        
        self.last_frame_time = time.time()
        return self.current_fps

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_controller = FrameRateController(target_fps=15)
    
    try:
        while True:
            # Receive base64 image from client
            data = await websocket.receive_text()
            
            try:
                # Process frame timing
                fps = await frame_controller.wait_for_next_frame()
                
                # Decode base64 image more efficiently
                encoded_data = data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process image with MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                # Prepare response
                response = {
                    "detected": False,
                    "landmarks": [],
                    "fps": round(fps, 1),
                    "timestamp": time.time()
                }
                
                if results.multi_hand_landmarks:
                    response["detected"] = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for point in hand_landmarks.landmark:
                            landmarks.append({
                                "x": round(point.x, 3),
                                "y": round(point.y, 3),
                                "z": round(point.z, 3)
                            })
                        response["landmarks"].append(landmarks)
                
                # Send response
                await websocket.send_json(response)
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                await websocket.send_json({
                    "error": str(e),
                    "timestamp": time.time()
                })
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)