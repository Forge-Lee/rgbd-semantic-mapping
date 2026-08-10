import numpy as np

# schema
class SemanticVoxelMapOutput:
    def __init__(
        self,
        points: np.ndarray,
        rgb_colors: np.ndarray,
        labels: np.ndarray,
        semantic_agreement: np.ndarray,
        mean_model_confidence: np.ndarray,
        observation_counts: np.ndarray
    ):
        self.points = points
        self.rgb_colors = rgb_colors
        self.labels = labels
        self.semantic_agreement = semantic_agreement
        self.mean_model_confidence = mean_model_confidence
        self.observation_counts = observation_counts

class SemanticVoxelMap:
    def __init__(
        self,
        voxel_size: float,
    ) -> None:
        '''
        self.voxels = {
            (ix, iy, iz): {
                "point_sum": ...,
                "color_sum": ...,
                "count": ...,
                "confidence_sum": ...,
                "label_scores": {
                    label_id: accumulated_confidence,
                },
            }
        }'''
        self.voxel_size = voxel_size

        if self.voxel_size <= 0:
            raise ValueError(
                f"voxel_size must be positive, got {self.voxel_size}"
            )

        if self.voxel_size <= 0:
            raise ValueError(
                f"voxel_size must be positive, got {self.voxel_size}"
            )

        self.voxels = {}

    def update(
        self,
        points: np.ndarray,
        rgb_colors: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        points = np.asarray(points)
        rgb_colors = np.asarray(rgb_colors)
        labels = np.asarray(labels)
        confidences = np.asarray(confidences)

        # sanity checks
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected points shape (N, 3), got {points.shape}"
            )

        number_of_points = len(points)

        

        if rgb_colors.shape != points.shape:
            raise ValueError(
                "RGB colors must have the same shape as points: "
                f"{rgb_colors.shape} vs {points.shape}"
            )
    
        if labels.shape != (len(points),):
            raise ValueError(
                f"Expected labels shape ({len(points)},), "
                f"got {labels.shape}"
            )

        if confidences.shape != (len(points),):
            raise ValueError(
                f"Expected confidences shape ({len(points)},), "
                f"got {confidences.shape}"
            )

        if number_of_points == 0:
            return

        if not np.all(np.isfinite(points)):
            raise ValueError("Points contain NaN or infinity.")
    
        if not np.all(np.isfinite(rgb_colors)):
            raise ValueError("RGB colors contain NaN or infinity.")
    
        if not np.all(np.isfinite(confidences)):
            raise ValueError("Confidences contain NaN or infinity.")
    
        if np.any(confidences < 0) or np.any(confidences > 1):
            raise ValueError(
                "Confidences must lie in the range [0, 1]."
            )

        # compute voxel indices for newly added pointcloud
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int64)

        for (point, color, label, confidence, voxel_index) in zip(points, rgb_colors, labels, confidences, voxel_indices):
            key = (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2]))
            if key not in self.voxels:
                self.voxels[key] = {
                    "point_sum": np.zeros(
                        3,
                        dtype=np.float64,
                    ),
                    "color_sum": np.zeros(
                        3,
                        dtype=np.float64,
                    ),
                    "confidence_sum": 0.0,
                    "count": 0,
                    "label_scores": {},
                }

            accumulator = self.voxels[key]

            accumulator["point_sum"] += point
            accumulator["color_sum"] += color
            accumulator["count"] += 1

            label_id = int(label)
            confidence_value = float(confidence)

            accumulator["confidence_sum"] += confidence_value

            label_scores: dict[int, float] = (accumulator["label_scores"])

            try:
                # if the id has been recorded in label_scores
                label_scores[label_id] += confidence_value
            except KeyError:
                # if this is the first time to add the id into label_scores
                label_scores[label_id] = confidence_value


    def export(self) -> SemanticVoxelMapOutput:
        number_of_voxels = len(self.voxels)
        fused_points = np.empty(
            (number_of_voxels, 3),
            dtype=np.float32,
        )
    
        fused_rgb_colors = np.empty(
            (number_of_voxels, 3),
            dtype=np.float32,
        )
    
        fused_labels = np.empty(
            number_of_voxels,
            dtype=np.int64,
        )
    
        semantic_agreement = np.empty(
            number_of_voxels,
            dtype=np.float32,
        )
    
        observation_counts = np.empty(
            number_of_voxels,
            dtype=np.int32,
        )

        mean_model_confidence_all = np.empty(
            number_of_voxels,
            dtype=np.float32,
        )
    
        # Sorting gives deterministic output order.
        sorted_voxel_keys = sorted(self.voxels)
    
        for output_index, key in enumerate(sorted_voxel_keys):
            accumulator = self.voxels[key]
    
            count = accumulator["count"]
            label_scores = accumulator["label_scores"]
    
            fused_points[output_index] = accumulator["point_sum"] / count
            fused_rgb_colors[output_index] = accumulator["color_sum"] / count
    
            winning_label, winning_score = max(label_scores.items(), key = lambda item: (item[1], -item[0]))
    
            total_score = accumulator["confidence_sum"]
    
            fused_labels[output_index] = winning_label
    
            if total_score > 0:
                semantic_agreement[output_index] = winning_score / total_score
            else:
                semantic_agreement[output_index] = 0.0
    
            observation_counts[output_index] = count

            mean_model_confidence = total_score / count
            mean_model_confidence_all[output_index] = mean_model_confidence
                
        return SemanticVoxelMapOutput(
            points = fused_points,
            rgb_colors = fused_rgb_colors,
            labels = fused_labels,
            semantic_agreement = semantic_agreement,
            mean_model_confidence=mean_model_confidence_all,
            observation_counts = observation_counts,
        )

    def __len__(
        self,
    ) -> int:
        return len(self.voxels)