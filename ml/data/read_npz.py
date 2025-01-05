import numpy as np
from pathlib import Path

def analyze_npz(file_path):
    """Analyze the contents of a .npz file"""
    data = np.load(file_path, allow_pickle=True)
    
    print("Arrays in the file:", data.files)
    
    # Analyze frames
    if 'frames' in data:
        frames = data['frames']
        print("\nFrames array:")
        print(f"Shape: {frames.shape}")
        print(f"Data type: {frames.dtype}")
        
    # Analyze landmarks
    if 'landmarks' in data:
        landmarks = data['landmarks']
        print("\nLandmarks array:")
        print(f"Shape: {landmarks.shape}")
        print(f"Data type: {landmarks.dtype}")
        
        # Analyze the first frame's landmarks if available
        if len(landmarks) > 0:
            print("\nFirst frame landmarks:")
            first_frame = landmarks[0]
            print(f"Number of hands detected: {len(first_frame)}")
            
            if len(first_frame) > 0:
                first_hand = first_frame[0]
                print(f"Number of landmarks per hand: {len(first_hand)}")
                print(f"Coordinates per landmark: {len(first_hand[0])}")
                print("\nSample landmark coordinates:")
                print(first_hand[0])  # Print first landmark of first hand

def main():
    # Replace with path to one of your .npz files
    file_path = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'landmarks' / 'accident' / '00630.npz'
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    analyze_npz(file_path)

if __name__ == "__main__":
    main()