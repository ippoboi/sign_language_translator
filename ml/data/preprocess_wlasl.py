import json
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
import multiprocessing
from functools import partial
import pandas as pd
from typing import List, Dict, Optional, Tuple
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

class WLASLPreprocessor:
    def __init__(self, base_path: str = '../../data', batch_size: int = 100, cpu_limit: float = 0.5, cooling_delay: float = 0.1):
        self.base_path = Path(base_path)
        self.batch_size = batch_size
        self.cpu_limit = cpu_limit  # Percentage of CPU cores to use (0.5 = 50%)
        self.cooling_delay = cooling_delay  # Delay in seconds between processing videos
        
        # Initialize directories
        self.dirs = {
            'raw_videos': self.base_path / 'raw' / 'videos',
            'interim': self.base_path / 'interim',
            'landmarks': self.base_path / 'interim' / 'landmarks',
            'metadata': self.base_path / 'raw' / 'metadata'
        }
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def analyze_dataset_availability(self, metadata: List[Dict]) -> Dict:
        """Analyze available vs missing videos"""
        total_in_json = 0
        available_videos = 0
        missing_by_class = {}
        available_by_class = {}
        
        video_files = list(self.dirs['raw_videos'].glob('*.mp4'))
        actual_videos = len(video_files)
        
        for entry in metadata:
            gloss = entry['gloss']
            instances = entry['instances']
            total_in_json += len(instances)
            
            missing_count = 0
            available_count = 0
            for instance in instances:
                video_id = instance['video_id']
                video_path = self.dirs['raw_videos'] / f'{video_id}.mp4'
                if not video_path.exists():
                    missing_count += 1
                else:
                    available_count += 1
            
            if missing_count > 0:
                missing_by_class[gloss] = missing_count
            if available_count > 0:
                available_by_class[gloss] = available_count
            available_videos += available_count
        
        logger.info("\nDataset Availability Analysis:")
        logger.info(f"Total videos in JSON: {total_in_json}")
        logger.info(f"Videos available: {actual_videos}")
        logger.info(f"Videos missing: {total_in_json - actual_videos}")
        logger.info(f"Availability percentage: {(actual_videos/total_in_json)*100:.2f}%")
        
        analysis_path = self.dirs['metadata'] / 'dataset_analysis.json'
        analysis_results = {
            "total_in_json": total_in_json,
            "available_videos": actual_videos,
            "missing_videos": total_in_json - actual_videos,
            "missing_by_class": missing_by_class,
            "available_by_class": available_by_class,
            "availability_percentage": (actual_videos/total_in_json)*100
        }
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
            
        logger.info(f"Detailed analysis saved to: {analysis_path}")
        return analysis_results
    
    def save_checkpoint(self, processed_videos: List[str], failed_videos: List[str]):
        """Save processing checkpoint"""
        checkpoint = {
            'processed': processed_videos,
            'failed': failed_videos,
            'timestamp': str(pd.Timestamp.now())
        }
        checkpoint_path = self.dirs['interim'] / 'checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {len(processed_videos)} videos processed")

    def load_checkpoint(self) -> Tuple[List[str], List[str]]:
        """Load processing checkpoint"""
        checkpoint_path = self.dirs['interim'] / 'checkpoint.json'
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint['timestamp']}")
            return checkpoint['processed'], checkpoint['failed']
        return [], []

    @staticmethod
    def extract_frames_and_landmarks(video_info: Dict) -> bool:
        import time  # Add at top of file if not already present
        # Get cooling delay from video_info
        cooling_delay = video_info.get('cooling_delay', 0)
        """Extract frames and hand landmarks from video"""
        try:
            # Initialize MediaPipe hands in the worker process
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            video_path = video_info['video_path']
            output_path = video_info['output_path']
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
                
            frames = []
            landmarks = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame for consistency
                frame = cv2.resize(frame, (224, 224))
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                frame_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract normalized landmarks
                        hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        frame_landmarks.append(hand_points)
                
                frames.append(frame)
                landmarks.append(frame_landmarks)
            
            cap.release()
            hands.close()
            
            # Add cooling delay
            if cooling_delay > 0:
                time.sleep(cooling_delay)
            
            # Save processed data
            if len(frames) > 0:
                frames = np.array(frames)
                landmarks = np.array(landmarks, dtype=object)
                np.savez_compressed(
                    output_path,
                    frames=frames,
                    landmarks=landmarks
                )
                return True
            else:
                logger.warning(f"No frames extracted from video: {video_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video_info['video_id']}: {str(e)}")
            return False

    def verify_video_integrity(self, video_path: Path) -> bool:
        """Verify if video file is readable and not corrupted"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return False
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.warning(f"Invalid frame count in video: {video_path}")
                return False
                
            # Check if we can read some frames
            frames_to_check = min(10, total_frames)  # Check first 10 frames or all if less
            valid_frames = 0
            
            for _ in range(frames_to_check):
                ret, frame = cap.read()
                if ret and frame is not None:
                    valid_frames += 1
                    
            cap.release()
            
            # Ensure we could read at least 80% of the frames we tried to check
            success_rate = valid_frames / frames_to_check
            if success_rate < 0.8:
                logger.warning(f"Low frame read success rate ({success_rate*100:.1f}%) for video: {video_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying video {video_path}: {str(e)}")
            return False

    def verify_processed_data(self, output_path: Path) -> bool:
        """Verify if processed data file is valid"""
        try:
            data = np.load(output_path, allow_pickle=True)
            return all(k in data.files for k in ['frames', 'landmarks'])
        except Exception as e:
            logger.error(f"Error verifying processed data {output_path}: {str(e)}")
            return False

    def process_batch(self, batch_videos: List[Dict], batch_id: int):
        """Process a batch of videos"""
        batch_results = {
            'processed': [],
            'failed': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Calculate number of cores based on cpu_limit
        available_cores = multiprocessing.cpu_count()
        num_cores = max(1, int(available_cores * self.cpu_limit))
        logger.info(f"Using {num_cores}/{available_cores} CPU cores")
        chunk_size = max(1, len(batch_videos) // (num_cores * 2))
        
        with multiprocessing.Pool(num_cores) as pool:
            for i, result in enumerate(tqdm(
                pool.imap(self.extract_frames_and_landmarks, batch_videos, chunksize=chunk_size),
                total=len(batch_videos),
                desc=f"Processing batch {batch_id}"
            )):
                video_id = batch_videos[i]['video_id']
                if result:
                    batch_results['processed'].append(video_id)
                else:
                    batch_results['failed'].append(video_id)
        
        end_time = datetime.now()
        batch_results['end_time'] = end_time.isoformat()
        batch_results['duration'] = (end_time - datetime.fromisoformat(batch_results['start_time'])).total_seconds()
        
        # Save batch results
        batch_path = self.dirs['interim'] / 'batch_results' / f'batch_{batch_id}.json'
        batch_path.parent.mkdir(exist_ok=True)
        with open(batch_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        return batch_results

    def process_videos(self, metadata: List[Dict]):
        """Process videos in batches with verification"""
        processed_videos, failed_videos = self.load_checkpoint()
        processed_set = set(processed_videos)
        failed_set = set(failed_videos)
        
        # Prepare video information
        videos_to_process = []
        for entry in metadata:
            gloss = entry['gloss']
            class_dir = self.dirs['landmarks'] / gloss
            class_dir.mkdir(exist_ok=True)
            
            for instance in entry['instances']:
                video_id = instance['video_id']
                if video_id not in processed_set and video_id not in failed_set:
                    video_path = self.dirs['raw_videos'] / f'{video_id}.mp4'
                    if video_path.exists() and self.verify_video_integrity(video_path):
                        videos_to_process.append({
                            'video_id': video_id,
                            'gloss': gloss,
                            'video_path': video_path,
                            'output_path': class_dir / f'{video_id}.npz',
                            'cooling_delay': self.cooling_delay
                        })
        
        if not videos_to_process:
            logger.info("No new videos to process")
            return processed_videos, failed_videos
        
        # Process in batches
        total_batches = (len(videos_to_process) + self.batch_size - 1) // self.batch_size
        batch_stats = []
        
        for batch_id in range(total_batches):
            start_idx = batch_id * self.batch_size
            end_idx = min((batch_id + 1) * self.batch_size, len(videos_to_process))
            batch_videos = videos_to_process[start_idx:end_idx]
            
            logger.info(f"\nProcessing batch {batch_id + 1}/{total_batches}")
            batch_result = self.process_batch(batch_videos, batch_id)
            batch_stats.append(batch_result)
            
            # Update overall progress
            processed_videos.extend(batch_result['processed'])
            failed_videos.extend(batch_result['failed'])
            self.save_checkpoint(processed_videos, failed_videos)
            
            # Log batch statistics
            logger.info(f"Batch {batch_id + 1} completed:")
            logger.info(f"Processed: {len(batch_result['processed'])} videos")
            logger.info(f"Failed: {len(batch_result['failed'])} videos")
            logger.info(f"Duration: {batch_result['duration']:.2f} seconds")
        
        # Save final statistics
        stats_path = self.dirs['interim'] / 'preprocessing_stats.json'
        final_stats = {
            'total_videos': len(videos_to_process),
            'total_processed': len(processed_videos),
            'total_failed': len(failed_videos),
            'batch_stats': batch_stats
        }
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        return processed_videos, failed_videos

def main():
    # Initialize preprocessor with resource limits
    preprocessor = WLASLPreprocessor(
        batch_size=100,
        cpu_limit=0.3,  # Use only 30% of CPU cores
        cooling_delay=0.1  # Add 0.1 second delay between videos
    )
    
    # Load metadata
    metadata_path = preprocessor.dirs['metadata'] / 'WLASL_v0.3.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Analyze dataset availability
    availability = preprocessor.analyze_dataset_availability(metadata)
    
    # Process videos in batches
    processed_videos, failed_videos = preprocessor.process_videos(metadata)
    
    # Final verification
    logger.info("\nPerforming final verification...")
    total_verified = 0
    total_corrupt = 0
    
    for processed_video in tqdm(processed_videos, desc="Verifying processed data"):
        gloss = next(entry['gloss'] for entry in metadata 
                    if any(instance['video_id'] == processed_video 
                          for instance in entry['instances']))
        output_path = preprocessor.dirs['landmarks'] / gloss / f'{processed_video}.npz'
        
        if preprocessor.verify_processed_data(output_path):
            total_verified += 1
        else:
            total_corrupt += 1
            failed_videos.append(processed_video)
    
    logger.info("\nProcessing Complete!")
    logger.info(f"Successfully processed and verified: {total_verified} videos")
    logger.info(f"Corrupted or failed: {total_corrupt} videos")
    logger.info(f"Total failed: {len(failed_videos)} videos")

if __name__ == "__main__":
    main()