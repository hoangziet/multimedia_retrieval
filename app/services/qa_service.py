from app.utils.query_cleaning import QueryCleaner
from app.utils.faiss_processing import FaissProcessor
from app.utils.vlm_answer import AnswerModel
from app.utils.common import frame_idx_to_str
from transformers import CLIPModel, CLIPProcessor
from configs import configs

class QAService:
    def __init__(self):
        self.faiss_processor = FaissProcessor()
        self.index, self.mapping = self.faiss_processor.load_faiss_index()
        self.query_cleaner = QueryCleaner()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(configs.DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.qa_model = AnswerModel()

    def encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.detach().cpu().numpy()

    def search(self, query: str, K=5):
        subqueries = self.query_cleaner.cleaning_qa(query)
        video_query = subqueries[0]
        keyframe_query = subqueries[1]

        query_features = self.encode_text([video_query])
        distances, indices = self.faiss_processor.query_faiss_index(self.index, query_features, K)

        qa_results = {
            "original_query": query,
            "cleaned_queries": subqueries,
            "qa": []
        }

        for dist, idx in zip(distances[0], indices[0]):
            video_id, frame_idx = self.mapping[idx]
            answer = self.qa_model.get_answer(keyframe_query, video_id, frame_idx)
            frame_idx_str = frame_idx_to_str(frame_idx)
            # image_url = f"{configs.KEYFRAMES_DIR}/{video_id}/{frame_idx_str}.jpg"
            # video_url = f"{configs.VIDEO_DIR}/{video_id}.mp4" 
            image_url = f"/keyframes/{video_id}/{frame_idx_str}.jpg"  
            video_url = f"/video/{video_id}.mp4" 
            qa_results["qa"].append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "answer": answer,
                "distance": float(dist),
                "image_url": image_url,
                "video_url": video_url
            })

        return qa_results