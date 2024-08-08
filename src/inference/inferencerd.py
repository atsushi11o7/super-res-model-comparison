import argparse

import numpy as np
import onnxruntime as ort
import torch


class Inferencer:
    def __init__(
        self,
        sess: ort.InferenceSession,
    ):
        """
        Args:
            embedder_sess (ort.InferenceSession): (N, 3, H, W) -> (N, 3, 4H, 4W)
        """
        super().__init__()

        self.sess = sess

        self.embedding_cache = np.zeros(shape=(5, 512), dtype=np.float32)

    def infer(self, x: np.ndarray) -> int:
        """

        Args:
            x (np.ndarray): (C, H, W)

        Returns:
            int: predicted future label
        """
        embedding = self.get_embedding(x[np.newaxis, :, :, :])  # (1, D)
        self.embedding_cache = np.concatenate([self.embedding_cache[1:, :], embedding], axis=0)
        predicted_future_embedding = self.predict_embedding(self.embedding_cache[np.newaxis, :, :])  # (1, D)
        predicted_future_label = self.get_label(predicted_future_embedding)  # (1,)
        return predicted_future_label.item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx_model_path", type=str, required=True)

    args = parser.parse_args()

    embedder_x = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    embedder_sess = ort.InferenceSession(args.embedder_onnx_path)
    output = embedder_sess.run(None, {"input": embedder_x.numpy()})
    print("embedder output shape", output[0].shape)

    inferencer = Inferencer(sess)

    for i in range(5):
        x = np.random.randn(3, 224, 224).astype(np.float32)
        label = inferencer.infer(x)
        print(f"--- {i} ---")
        print("label", label)
        print("embedding_cache", inferencer.embedding_cache)


if __name__ == "__main__":
    main()