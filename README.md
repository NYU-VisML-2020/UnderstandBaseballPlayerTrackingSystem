# Temporal Analysis on Baseball Players
Understand baseball players in real-time games

The pre-trained objdect detection model is provided by TensorFlow API:
[TensorFlowAPI](https://github.com/szhaofelicia/models).

And the depth estimation model is provided By [Ibraheem Alhashim and Peter Wonka](https://github.com/szhaofelicia/DenseDepth). The results of depth estimation are saved into JSON files.

The dataset comes from [MLB-Youtube daset](https://github.com/szhaofelicia/mlb-youtube). Add [classify_frames.py](https://github.com/NYU-VisML-2020/UnderstandBaseballPlayerTrackingSystem/blob/master/classify_frames.py) under mlb-youtube/ to extract clips of swing activity.

Please install TensorFlow Object Detection API before running [refine_bbox_mlb.py](https://github.com/NYU-VisML-2020/UnderstandBaseballPlayerTrackingSystem/blob/master/refine_bbox_mlb.py), and save refine_bbox_mlb.py under the folder models/research/object_detection.

Visualize bbox and color box in descending order of detection scores: replace file with [visualization_utils.py](https://github.com/NYU-VisML-2020/UnderstandBaseballPlayerTrackingSystem/blob/master/visualization_utils.py) under models/research/object_detection/utils.



