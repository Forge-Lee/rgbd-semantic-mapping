import numpy as np
import torch
import torch.nn.functional as functional
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# inference model configuration
DEFAULT_MODEL_ID = (
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# schema
class SemanticPrediction:
    def __init__(self, labels, confidence):
        self.labels = labels
        self.confidence = confidence

# predictor
class SemanticSegmenter:
    def __init__(
            self, 
            model_id: str = DEFAULT_MODEL_ID, 
            device : str = "cpu",
        ):
        self.device = torch.device(device)

        self.processor = (
            SegformerImageProcessor.from_pretrained(
                model_id
            )
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(model_id)
            .to(self.device)
        )
        self.model.eval()

        self.id_to_label = {
            int(label_id): class_name
            for label_id, class_name
            in self.model.config.id2label.items()
        }

    def predict(
            self, 
            rgb: np.ndarray
        ) -> SemanticPrediction:
        rgb = np.asarray(rgb)

        # sanity check
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "Expected RGB image with shape "
                f"(H, W, 3), got {rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            raise ValueError(
                "Expected RGB image with dtype uint8, "
                f"got {rgb.dtype}"
            )

        # formulate input
        height, width = rgb.shape[:2] # store the original resolution for further restoration

        inputs = self.processor(
            images=rgb,
            return_tensors="pt",
        ) # becomes a dictionary after preprocessing

        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        } # put to the same device while keeping the input structure

        with torch.inference_mode():
            outputs = self.model(**inputs)

            logits = functional.interpolate(
                outputs.logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ) # restore to original resolution

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, labels = probabilities.max(
                dim=1
            )

        labels_numpy = (
            labels[0]
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        confidence_numpy = (
            confidence[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return SemanticPrediction(
            labels=labels_numpy,
            confidence=confidence_numpy,
        )

    def get_class_name(
        self,
        label_id: int,
    ) -> str:
        return self.id_to_label.get(
            int(label_id),
            f"unknown_{label_id}",
        )