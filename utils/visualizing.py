import cv2
import matplotlib.pyplot as plt


class Visualizer():
    def __init__(self):
        pass 
    
    def __call__(self, results):
        # result len in 1, 5, 20, 50, 100
        num_results = len(results)
        
        if num_results <= 5:
            nrows, ncols = 1, num_results
        else:
            nrows, ncols = num_results // 5, 5

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
        axes = axes.flatten()
        for i, res in enumerate(results):
            self.visualize(res, axes[i])

        return fig
    
    def visualize(self, res, ax):
        video_id = res["video_id"]
        frame_idx = res["frame_idx"]
        keyframes_path = res["keyframes_path"]
        distance = res["distance"]

        img = cv2.imread(str(keyframes_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.set_title(f"{video_id}/{frame_idx}/{distance:.2f}")
        ax.axis("off")

