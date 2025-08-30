from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch 

from configs import configs


class AnswerModel:
    processor = None
    model = None
    
    @classmethod
    def initialize(cls):
        if cls.processor is None or cls.model is None:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            
            # Dùng processor và model đúng với BLIP-2
            cls.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            cls.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            
            if torch.cuda.is_available():
                cls.model = cls.model.to("cuda")
    
    @classmethod
    def get_answer(cls, query, video_id, frame_idx):
        cls.initialize()
        
        # Xử lý query và frame
        from PIL import Image
        import os
        from configs import configs
        
        frame_idx_str = str(frame_idx).zfill(3)
        image_path = os.path.join(configs.KEYFRAMES_DIR, video_id, f"{frame_idx_str}.jpg")
        
        try:
            image = Image.open(image_path)
            
            # Xử lý với BLIP-2
            import torch
            
            inputs = cls.processor(images=image, text=query, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            # Sinh câu trả lời
            with torch.no_grad():
                outputs = cls.model.generate(**inputs, max_new_tokens=50)
            
            # Giải mã câu trả lời
            answer = cls.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return answer
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {e}")
            return "Không thể trả lời câu hỏi này."
