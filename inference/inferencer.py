import onnxruntime as ort
import numpy as np
from pathlib import Path
import cv2
import datetime

class InferenceONNX:
    def __init__(self, model_path: str, input_image_dir: str, output_image_dir: str):
        self.model_path = model_path
        self.input_image_dir = Path(input_image_dir)
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(exist_ok=True, parents=True)
        self.sess = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def infer(self):
        input_images = []
        output_paths = []

        for image_path in self.input_image_dir.iterdir():
            output_image_path = self.output_image_dir / image_path.relative_to(self.input_image_dir)
            input_image = cv2.imread(str(image_path))
            input_image = np.array([cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2,0,1))], dtype=np.float32)/255
            input_images.append(input_image)
            output_paths.append(output_image_path)

        start_time = datetime.datetime.now()
        output_images = [self.sess.run(["output"], {"input": input_image})[0] for input_image in input_images]
        end_time = datetime.datetime.now()