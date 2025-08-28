from app.utils.query_cleaning import QueryCleaner
from app.utils.faiss_processing import FaissProcessor
from app.utils.common import frame_idx_to_str
from transformers import CLIPModel, CLIPProcessor
from configs import configs

class TrakeService:
    def __init__(self):
        self.faiss_processor = FaissProcessor()
        self.index, self.mapping = self.faiss_processor.load_faiss_index()
        self.query_cleaner = QueryCleaner()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(configs.DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.detach().cpu().numpy()

    def search(self, query: str, K=5):
        subqueries = self.query_cleaner.cleaning_trake(query)
        video_query = subqueries[0]
        keyframe_queries = subqueries[1:]

        query_features = self.encode_text([video_query])
        distances, indices = self.faiss_processor.query_faiss_index(self.index, query_features, K)

        trake_results = {
            "original_query": query,
            "cleaned_queries": subqueries,
            "trake": {}
        }

        for dist, idx in zip(distances[0], indices[0]):
            video_id, _ = self.mapping[idx]
            trake_results["trake"][video_id] = {
                "keyframes": [],
                "distances": [],
                "kis_distance": float(dist)
            }
            video_faiss_index = self.faiss_processor.build_single_faiss_index(video_id)
            for keyframe_query in keyframe_queries:
                keyframe_features = self.encode_text([keyframe_query])
                D, I = video_faiss_index.search(keyframe_features, 1)
                if I[0][0] != -1:
                    frame_idx = int(I[0][0])
                    frame_idx_str = frame_idx_to_str(frame_idx)
                    image_url = f"/keyframes/{video_id}/{frame_idx_str}.jpg"
                    video_url = f"/video/{video_id}.mp4"
                    trake_results["trake"][video_id]["keyframes"].append({
                        "frame_idx": frame_idx,
                        "distance": float(D[0][0]),
                        "image_url": image_url,
                        "video_url": video_url
                    })
                    trake_results["trake"][video_id]["distances"].append(float(D[0][0]))

        return trake_results