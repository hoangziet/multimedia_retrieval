import json
import os
from datetime import datetime

import torch
from transformers import CLIPModel, CLIPProcessor

from configs import configs

from .faiss_processing import FaissProcessor
from .query_cleaning import QueryCleaner
from .visualizing import Visualizer
from .vlm_answer import AnswerModel


class QueryModel:
    def __init__(self, 
                 faiss_processor = FaissProcessor(),
                 query_cleaner = QueryCleaner(),
                 model =  CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                 processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                 qa_model = AnswerModel(),
                 debug = configs.DEBUG_FLAG):
        
        self.faiss_processor = faiss_processor
        self.index, self.mapping = self.faiss_processor.load_faiss_index()
        self.query_cleaner = query_cleaner
        self.model = model.to(configs.DEVICE)
        self.processor = processor
        self.debug = debug
        self.qa_model = qa_model

    def process_query(self, query_features, K=5):
        D, I = self.faiss_processor.query_faiss_index(self.index, query_features, K)
        return D, I

    @torch.no_grad()
    def encode_text(self, texts, device = configs.DEVICE):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)

        text_features = self.model.get_text_features(**inputs) # (B, 512)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features.cpu().numpy()


    def search_text(self, query: str, device = configs.DEVICE, K=5):
        query_features = self.encode_text([query], device)
        distances, indices = self.faiss_processor.query_faiss_index(self.index, query_features, K)

        search_results = []
        for dist, idx in zip(distances[0], indices[0]):
            video_id, frame_idx = self.mapping[idx]
            result = {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "distance": float(dist)
            }

            search_results.append(result)

        return search_results


    def kis(self, query, K = 5):
        cleaned_query = self.query_cleaner.cleaning_kis(query)
        search_results = self.search_text(cleaned_query, K)

        kis_results = {
            "original_query": query,
            "cleaned_queries": [cleaned_query],
            "query_results": search_results
        }

        return kis_results

    
    def qa(self, query, K = 5):
        subqueries = self.query_cleaner.cleaning_qa(query)
        video_query = subqueries[0]
        keyframe_query = subqueries[1]

        search_results = self.search_text(video_query, K)

        qa_results = {
            "original_query": query,
            "cleaned_queries": subqueries,
            "qa": []
        }

        for search_result in search_results:
            video_id = search_result["video_id"]
            key_frame = search_result["frame_idx"]
            distance = search_result["distance"]
            keyframe_answer = self.qa_model.get_answer(keyframe_query, video_id, key_frame)
            qa_results["qa"].append({
                "video_id": video_id,
                "frame_idx": key_frame,
                "answer": keyframe_answer,
                "distance": distance
            })

        return qa_results

    def trake(self, query, K = 5):
        subqueries = self.query_cleaner.cleaning_trake(query)
        video_query = subqueries[0]
        keyframe_queries = subqueries[1:]

        search_results = self.search_text(video_query, K)
        
        trake_results = {
            "original_query": query,
            "cleaned_queries": subqueries,
        }
        
        trake = {}
        for search_result in search_results:
            video_id = search_result["video_id"]
            trake[video_id] = {
                "keyframes": [],
                "distances": [],
                "kis_distance": search_result["distance"]
            }

            video_faiss_index = self.faiss_processor.build_single_faiss_index(video_id)
            for keyframe_query in keyframe_queries:
                keyframe_features = self.encode_text([keyframe_query])
                D, I = video_faiss_index.search(keyframe_features, 1)  # K=1
                if I[0][0] != -1:
                    trake[video_id]["keyframes"].append(int(I[0][0]))
                    trake[video_id]["distances"].append(float(D[0][0]))

        trake_results["trake"] = trake
        return trake_results

    def run(self, query: str, task: str = "kis", K=5):
        if task == "kis":
            results =  self.kis(query, K)
        elif task == "qa":
            results =  self.qa(query, K)
        elif task == "trake":
            results =  self.trake(query, K)
            
        if self.debug:
            date = datetime.now()
            bd = date.strftime("%b%d")
            hm = date.strftime("%H%M")
            debug_dir = f"debug/{bd}/{hm}" 
            os.makedirs(debug_dir, exist_ok=True)


            # write results.json
            with open(f"{debug_dir}/results.json", "w") as f:
                json.dump(results , f)

            # # write results.png
            # visualizer = Visualizer()
            # fig = visualizer(results)
            # fig.savefig(f"{debug_dir}/results.png")

        return results