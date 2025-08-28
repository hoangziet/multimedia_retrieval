import numpy as np 
import faiss 
import logging
import json 
import os 

from glob import glob
from pathlib import Path 

from configs import configs
logging.basicConfig(level=logging.INFO)


class FaissProcessor:
    def __init__(self):
        pass 

    def build_faiss_index(self):
        features_chunks = []
        mapping = [] # (video_id, frame_id)
        total_frames = 0
        
        npy_dir = Path(configs.CLIP_FEATURE_32_DIR)
        npy_files = npy_dir.glob("*.npy")
        for npy_file in npy_files:
            video_id = npy_file.stem
            features = np.load(npy_file).astype("float32")  # (num_frames, 512)
            
            # get number of frames
            num_frames = features.shape[0]
            total_frames += num_frames
    
            # append features and mapping
            features_chunks.append(features)
            mapping.extend([(video_id, i) for i in range(num_frames)])
            
        # concatenate all features
        all_features = np.concatenate(features_chunks) # (total_frames, 512)

        # create faiss index 
        index = faiss.IndexFlatL2(512)
        index.add(all_features)

        logging.info(f"Loaded {total_frames} frames to FAISS index.")

        return index, mapping

    def save_faiss_index(self,
                         index,
                         mapping,
                         index_path = configs.INDEX_PATH,
                         mapping_path = configs.MAPPING_PATH):
        database_dir = os.path.dirname(index_path)
        os.makedirs(database_dir, exist_ok=True)
            
        faiss.write_index(index, index_path)
        logging.info(f"FAISS index saved to {index_path}.")

        with open(mapping_path, "w") as f:
            json.dump(mapping, f)
        logging.info(f"Mapping saved to {mapping_path}.")
    

    def query_faiss_index(self, index, query_features, K = 5):
        D, I = index.search(query_features, K)
        return D, I


    def load_faiss_index(self, 
                         index_path = configs.INDEX_PATH,
                         mapping_path = configs.MAPPING_PATH):

        # if not exist, create again
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            logging.info(f"FAISS index or mapping not found. Building new index.")
            index, mapping = self.build_faiss_index()
            self.save_faiss_index(index, mapping)
    
        index = faiss.read_index(index_path)
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        logging.info(f"FAISS index loaded from {index_path}.")
        logging.info(f"Mapping loaded from {mapping_path}.")

        return index, mapping
    

    def build_single_faiss_index(self, video_id):

        npy_path = os.path.join(configs.CLIP_FEATURE_32_DIR, f"{video_id}.npy")
        features = np.load(npy_path).astype("float32")  # (num_frames, 512)

        # create faiss index
        index = faiss.IndexFlatL2(512)
        index.add(features)

        return index
        
if __name__ == "__main__":
    processor = FaissProcessor()
    index, mapping = processor.load_faiss_index()