

import cv2
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import argparse

class FrameReconstructor:
    def __init__(self, video_path, output_path="reconstructed_video.mp4", use_gpu=False):
        self.video_path = video_path
        self.output_path = output_path
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.frames = []
        self.frame_count = 0
        self.fps = 60
        self.frame_size = None
        
    def extract_frames(self):
        """Extract all frames from the jumbled video"""
        print("Extracting frames from video...")
        cap = cv2.VideoCapture(self.video_path)
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        self.frames = frames
        self.frame_size = frames[0].shape[:2][::-1]  # (width, height)
        print(f"Extracted {len(frames)} frames at {self.fps} FPS")
        return frames
    
    def compute_frame_features(self, frame, resize_dim=(160, 90)):
        """
        Compute multiple features for a frame:
        - Downsampled frame for fast comparison
        - Histogram
        - Edge map
        """
        # Resize for faster computation
        small = cv2.resize(frame, resize_dim)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Compute edges (motion/change indicator)
        edges = cv2.Canny(gray, 50, 150)
        
        return {
            'small': small,
            'gray': gray,
            'hist': hist,
            'edges': edges
        }
    
    def compute_similarity(self, feat1, feat2):
        """
        Compute similarity between two frames using multiple metrics
        Higher score = more similar
        """
        # Structural similarity (pixel-level)
        mse = np.mean((feat1['small'].astype(float) - feat2['small'].astype(float)) ** 2)
        pixel_sim = 1.0 / (1.0 + mse / 1000.0)
        
        # Histogram similarity
        hist_sim = cv2.compareHist(feat1['hist'], feat2['hist'], cv2.HISTCMP_CORREL)
        hist_sim = (hist_sim + 1) / 2  # Normalize to [0, 1]
        
        # Edge similarity
        edge_diff = np.sum(np.abs(feat1['edges'].astype(float) - feat2['edges'].astype(float)))
        edge_sim = 1.0 / (1.0 + edge_diff / 10000.0)
        
        # Weighted combination
        total_sim = 0.5 * pixel_sim + 0.3 * hist_sim + 0.2 * edge_sim
        
        return total_sim
    
    def build_similarity_matrix(self):
        """Build pairwise similarity matrix between all frames"""
        print("Computing frame features...")
        
        # Compute features for all frames in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            features = list(tqdm(
                executor.map(self.compute_frame_features, self.frames),
                total=len(self.frames),
                desc="Feature extraction"
            ))
        
        print("Building similarity matrix...")
        n = len(self.frames)
        similarity_matrix = np.zeros((n, n), dtype=np.float32)
        
        # Compute similarities in parallel
        def compute_row(i):
            row = np.zeros(n, dtype=np.float32)
            for j in range(n):
                if i != j:
                    row[j] = self.compute_similarity(features[i], features[j])
            return i, row
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(
                executor.map(compute_row, range(n)),
                total=n,
                desc="Similarity computation"
            ))
        
        for i, row in results:
            similarity_matrix[i] = row
        
        return similarity_matrix
    
    def find_starting_frame(self, similarity_matrix):
        """
        Find the actual starting frame of the video sequence.
        The starting frame should have:
        1. Exactly ONE strong forward connection (to frame 2)
        2. NO strong backward connections (nothing comes before it)
        """
        print("Detecting starting frame...")
        n = len(similarity_matrix)
        
        # For each frame, compute:
        # - Best forward similarity (max similarity to any frame)
        # - Number of frames that strongly point to it (backward connections)
        
        start_scores = []
        
        for i in range(n):
            # Get top similarities for this frame
            similarities = similarity_matrix[i].copy()
            similarities[i] = 0  # Exclude self
            
            # Best forward connection
            best_forward = np.max(similarities)
            
            # Count how many frames strongly point TO this frame
            # (i.e., this frame is their best match)
            backward_count = 0
            threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 85)
            
            for j in range(n):
                if i != j:
                    # Is frame i the best or near-best match for frame j?
                    if similarity_matrix[j, i] > threshold:
                        # Check if i is among j's top candidates
                        j_similarities = similarity_matrix[j].copy()
                        j_similarities[j] = 0
                        if similarity_matrix[j, i] >= np.sort(j_similarities)[-3]:  # Top 3
                            backward_count += 1
            
            # Starting frame should have:
            # - Strong forward connection (high best_forward)
            # - Few backward connections (low backward_count)
            # Score: prioritize frames with strong forward but weak backward
            start_score = best_forward * (1.0 / (1.0 + backward_count))
            start_scores.append(start_score)
        
        # Get top candidates
        start_scores = np.array(start_scores)
        top_candidates = np.argsort(start_scores)[-10:][::-1]  # Top 10 candidates
        
        print(f"Top starting frame candidates: {top_candidates[:5]}")
        print(f"  Scores: {[start_scores[i] for i in top_candidates[:5]]}")
        
        # Return the best candidate
        return top_candidates[0]
    
    def greedy_path_reconstruction(self, similarity_matrix, start_idx):
        """
        Reconstruct frame order using greedy nearest-neighbor approach
        Starting from the detected start frame
        """
        print(f"Reconstructing frame order from frame {start_idx}...")
        n = len(self.frames)
        
        # Track used frames
        used = set([start_idx])
        path = [start_idx]
        
        current = start_idx
        
        # Build path by always selecting most similar unused frame
        pbar = tqdm(total=n-1, desc="Building sequence")
        while len(used) < n:
            # Get similarities to all unused frames
            candidates = [(i, similarity_matrix[current, i]) 
                         for i in range(n) if i not in used]
            
            if not candidates:
                break
            
            # Choose most similar frame
            next_idx = max(candidates, key=lambda x: x[1])[0]
            
            path.append(next_idx)
            used.add(next_idx)
            current = next_idx
            pbar.update(1)
        
        pbar.close()
        
        return path
    
    def try_multiple_starts_and_pick_best(self, similarity_matrix):
        """
        Try building sequences from multiple candidate starting points
        and pick the one with the highest average similarity
        """
        print("Testing multiple starting points...")
        n = len(similarity_matrix)
        
        # Get multiple starting candidates
        start_scores = []
        for i in range(n):
            similarities = similarity_matrix[i].copy()
            similarities[i] = 0
            best_forward = np.max(similarities)
            
            backward_count = 0
            threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 85)
            for j in range(n):
                if i != j and similarity_matrix[j, i] > threshold:
                    j_similarities = similarity_matrix[j].copy()
                    j_similarities[j] = 0
                    if similarity_matrix[j, i] >= np.sort(j_similarities)[-3]:
                        backward_count += 1
            
            start_score = best_forward * (1.0 / (1.0 + backward_count))
            start_scores.append(start_score)
        
        # Get top 5 candidates
        candidates = np.argsort(start_scores)[-5:][::-1]
        
        best_path = None
        best_avg_similarity = 0
        
        for candidate in candidates:
            # Build path from this candidate
            path = self.greedy_path_reconstruction(similarity_matrix, candidate)
            
            # Compute average similarity along this path
            total_sim = 0
            for i in range(len(path) - 1):
                total_sim += similarity_matrix[path[i], path[i+1]]
            avg_sim = total_sim / (len(path) - 1)
            
            print(f"  Candidate {candidate}: Average similarity = {avg_sim:.4f}")
            
            if avg_sim > best_avg_similarity:
                best_avg_similarity = avg_sim
                best_path = path
        
        print(f"Selected best path with average similarity: {best_avg_similarity:.4f}")
        return best_path
    
    def optimize_sequence(self, path, similarity_matrix, iterations=3):
        """
        Optimize the sequence using local search
        Try swapping nearby frames to improve overall similarity
        """
        print(f"Optimizing sequence ({iterations} iterations)...")
        
        for iteration in range(iterations):
            improved = False
            
            for i in range(1, len(path) - 1):
                # Try swapping with next frame
                if i < len(path) - 1:
                    original_score = (
                        similarity_matrix[path[i-1], path[i]] +
                        similarity_matrix[path[i], path[i+1]]
                    )
                    
                    new_score = (
                        similarity_matrix[path[i-1], path[i+1]] +
                        similarity_matrix[path[i+1], path[i]]
                    )
                    
                    if new_score > original_score:
                        path[i], path[i+1] = path[i+1], path[i]
                        improved = True
            
            if not improved:
                break
            
            print(f"  Iteration {iteration + 1}: Improvements made")
        
        return path
    
    def reconstruct_video(self, frame_order):
        """Create output video from ordered frames"""
        print("Writing reconstructed video...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        
        for idx in tqdm(frame_order, desc="Writing frames"):
            out.write(self.frames[idx])
        
        out.release()
        print(f"Reconstructed video saved to: {self.output_path}")
    
    def run(self):
        """Main execution pipeline"""
        start_time = time.time()
        
        # Step 1: Extract frames
        self.extract_frames()
        
        # Step 2: Build similarity matrix
        similarity_matrix = self.build_similarity_matrix()
        
        # Step 3: Find best starting point and reconstruct
        frame_order = self.try_multiple_starts_and_pick_best(similarity_matrix)
        
        # Step 4: Optimize sequence
        frame_order = self.optimize_sequence(frame_order, similarity_matrix)
        
        # Step 5: Create output video
        self.reconstruct_video(frame_order)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*50)
        print(f"RECONSTRUCTION COMPLETE")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Output saved to: {self.output_path}")
        print("="*50)
        
        return total_time


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct jumbled video frames'
    )
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to jumbled video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reconstructed_video.mp4',
        help='Output video path (default: reconstructed_video.mp4)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )
    
    args = parser.parse_args()
    
    # Run reconstruction
    reconstructor = FrameReconstructor(
        args.input_video,
        args.output,
        use_gpu=args.gpu
    )
    
    execution_time = reconstructor.run()
    
    # Save execution log
    with open('execution_time.log', 'w') as f:
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Input: {args.input_video}\n")
        f.write(f"Output: {args.output}\n")


if __name__ == "__main__":
    main()
