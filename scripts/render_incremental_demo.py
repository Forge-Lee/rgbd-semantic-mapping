from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import os

os.environ["EGL_PLATFORM"] = "surfaceless"

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------
# Palette
# ---------------------------------------------------------

ADE20K_CLASS_COUNT = 150
UNKNOWN_LABEL_ID = 150


def create_semantic_palette(
    number_of_classes: int = 151,
) -> np.ndarray:
    """
    Create a deterministic semantic color palette.

    Returns:
        palette: (K, 3), float32, range [0, 1]
    """
    palette = np.zeros(
        (number_of_classes, 3),
        dtype=np.float32,
    )

    # Golden-ratio hue spacing gives deterministic,
    # reasonably separated colors.
    golden_ratio = 0.618033988749895

    for label_id in range(number_of_classes):
        hue = (label_id * golden_ratio) % 1.0

        rgb = colorsys.hsv_to_rgb(
            hue,
            0.70,
            0.95,
        )

        palette[label_id] = rgb

    # Unknown = neutral gray
    if UNKNOWN_LABEL_ID < number_of_classes:
        palette[UNKNOWN_LABEL_ID] = np.array(
            [0.45, 0.45, 0.45],
            dtype=np.float32,
        )

    return palette


# ---------------------------------------------------------
# Snapshot
# ---------------------------------------------------------

def load_snapshot(
    path: Path,
) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            key: data[key]
            for key in data.files
        }


def apply_unknown_threshold(
    labels: np.ndarray,
    semantic_agreement: np.ndarray,
    mean_model_confidence: np.ndarray,
    agreement_threshold: float,
    confidence_threshold: float,
) -> np.ndarray:
    """
    Convert uncertain semantic voxels to UNKNOWN_LABEL_ID.
    """
    output_labels = labels.copy()

    uncertain_mask = (
        (semantic_agreement < agreement_threshold)
        |
        (mean_model_confidence < confidence_threshold)
    )

    output_labels[uncertain_mask] = UNKNOWN_LABEL_ID

    return output_labels


# ---------------------------------------------------------
# Open3D utilities
# ---------------------------------------------------------

def create_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )

    cloud.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64)
    )

    return cloud


def compute_fixed_camera(
    final_points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    Compute one camera pose from the FINAL map.

    Every frame then uses exactly this camera.
    """
    minimum = final_points.min(axis=0)
    maximum = final_points.max(axis=0)

    center = (
        minimum + maximum
    ) / 2.0

    extent = maximum - minimum

    radius = max(
        float(np.linalg.norm(extent)) / 2.0,
        0.1,
    )

    # Change this direction later if the map is viewed
    # from an undesirable side.
    direction = np.array(
        [1.3, -1.3, 0.9],
        dtype=np.float64,
    )

    direction /= np.linalg.norm(direction)

    camera_distance = 3.0 * radius

    eye = (
        center
        + direction * camera_distance
    )

    # Assumes world Z is approximately upward.
    up = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    near_clip = max(
        0.01,
        camera_distance - 2.0 * radius,
    )

    far_clip = (
        camera_distance
        + 2.5 * radius
    )

    return (
        center,
        eye,
        up,
        near_clip,
        far_clip,
    )


class SemanticMapRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        point_size: float,
        final_points: np.ndarray,
    ) -> None:
        self.width = width
        self.height = height
        self.point_size = point_size

        self.renderer = (
            o3d.visualization.rendering.OffscreenRenderer(
                width,
                height,
            )
        )

        self.renderer.scene.set_background(
            np.array(
                [0.035, 0.035, 0.045, 1.0],
                dtype=np.float32,
            )
        )

        self.renderer.scene.show_axes(True)

        (
            self.center,
            self.eye,
            self.up,
            self.near_clip,
            self.far_clip,
        ) = compute_fixed_camera(
            final_points
        )

    def render(
        self,
        points: np.ndarray,
        semantic_colors: np.ndarray,
        current_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render accumulated semantic map.

        Current-frame points are optionally highlighted.
        """
        self.renderer.scene.clear_geometry()

        if len(points) > 0:
            map_cloud = create_point_cloud(
                points,
                semantic_colors,
            )

            material = (
                o3d.visualization.rendering.MaterialRecord()
            )

            material.shader = "defaultUnlit"
            material.point_size = self.point_size

            self.renderer.scene.add_geometry(
                "semantic_map",
                map_cloud,
                material,
            )

        # Highlight the current incoming frame.
        if (
            current_points is not None
            and len(current_points) > 0
        ):
            # Downsample highlight only for visualization.
            current_points = current_points[::4]

            highlight_colors = np.tile(
                np.array(
                    [[1.0, 0.85, 0.15]],
                    dtype=np.float32,
                ),
                (len(current_points), 1),
            )

            current_cloud = create_point_cloud(
                current_points,
                highlight_colors,
            )

            current_material = (
                o3d.visualization.rendering.MaterialRecord()
            )

            current_material.shader = "defaultUnlit"
            current_material.point_size = (
                self.point_size + 1.5
            )

            self.renderer.scene.add_geometry(
                "current_frame",
                current_cloud,
                current_material,
            )

        # Official OffscreenRenderer supports this
        # eye / center / up camera interface.
        self.renderer.setup_camera(
            50.0,  # vertical FOV
            self.center,
            self.eye,
            self.up,
            self.near_clip,
            self.far_clip,
        )

        rendered_image = (
            self.renderer.render_to_image()
        )

        image_array = np.asarray(
            rendered_image
        )

        # Some builds return RGBA.
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]

        return image_array.astype(
            np.uint8
        )


# ---------------------------------------------------------
# PIL video-frame composition
# ---------------------------------------------------------

def load_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size=size,
        )
    except OSError:
        return ImageFont.load_default()


def fit_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """
    Resize image without changing aspect ratio and place
    it onto a dark background.
    """
    image = image.convert("RGB")

    image.thumbnail(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (target_width, target_height),
        (18, 18, 22),
    )

    x = (
        target_width - image.width
    ) // 2

    y = (
        target_height - image.height
    ) // 2

    canvas.paste(
        image,
        (x, y),
    )

    return canvas


def create_map_only_frame(
    map_image: np.ndarray,
    frame_index: int,
    total_frames: int,
    voxel_count: int,
    current_point_count: int,
) -> np.ndarray:
    width = 1280
    height = 720

    canvas = Image.new(
        "RGB",
        (width, height),
        (16, 16, 20),
    )

    map_pil = Image.fromarray(
        map_image
    )

    panel = fit_image(
        map_pil,
        width,
        610,
    )

    canvas.paste(
        panel,
        (0, 55),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    title_font = load_font(26)
    info_font = load_font(20)

    draw.text(
        (25, 15),
        "Incremental RGB-D Semantic Mapping",
        font=title_font,
        fill=(245, 245, 245),
    )

    info = (
        f"Frame {frame_index + 1}/{total_frames}"
        f"    |    Map voxels: {voxel_count:,}"
        f"    |    Current observations: "
        f"{current_point_count:,}"
    )

    draw.text(
        (25, 680),
        info,
        font=info_font,
        fill=(220, 220, 220),
    )

    return np.asarray(
        canvas,
        dtype=np.uint8,
    )


def create_triple_frame(
    rgb_path: Path,
    semantic_path: Path,
    map_image: np.ndarray,
    frame_index: int,
    total_frames: int,
    voxel_count: int,
) -> np.ndarray:
    """
    Create:
        RGB | 2D semantic prediction | 3D semantic map
    """
    width = 1920
    height = 1080

    canvas = Image.new(
        "RGB",
        (width, height),
        (16, 16, 20),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    title_font = load_font(30)
    label_font = load_font(23)
    info_font = load_font(22)

    draw.text(
        (30, 18),
        "Replayable RGB-D Semantic Mapping",
        font=title_font,
        fill=(245, 245, 245),
    )

    panel_y = 100
    panel_height = 820

    left_width = 560
    middle_width = 560
    right_width = 760

    rgb = Image.open(
        rgb_path
    ).convert("RGB")

    semantic = Image.open(
        semantic_path
    ).convert("RGB")

    map_pil = Image.fromarray(
        map_image
    )

    rgb_panel = fit_image(
        rgb,
        left_width,
        panel_height,
    )

    semantic_panel = fit_image(
        semantic,
        middle_width,
        panel_height,
    )

    map_panel = fit_image(
        map_pil,
        right_width,
        panel_height,
    )

    canvas.paste(
        rgb_panel,
        (20, panel_y),
    )

    canvas.paste(
        semantic_panel,
        (590, panel_y),
    )

    canvas.paste(
        map_panel,
        (1160, panel_y),
    )

    draw.text(
        (20, 65),
        "RGB Input",
        font=label_font,
        fill=(225, 225, 225),
    )

    draw.text(
        (590, 65),
        "2D Semantic Prediction",
        font=label_font,
        fill=(225, 225, 225),
    )

    draw.text(
        (1160, 65),
        "Incremental 3D Map",
        font=label_font,
        fill=(225, 225, 225),
    )

    info = (
        f"Frame {frame_index + 1}/{total_frames}"
        f"    |    Persistent voxels: {voxel_count:,}"
    )

    draw.text(
        (30, 980),
        info,
        font=info_font,
        fill=(220, 220, 220),
    )

    draw.text(
        (30, 1020),
        "Confidence-weighted semantic voxel fusion",
        font=info_font,
        fill=(175, 175, 175),
    )

    return np.asarray(
        canvas,
        dtype=np.uint8,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--agreement-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--layout",
        choices=[
            "map",
            "triple",
        ],
        default="map",
    )

    parser.add_argument(
        "--final-hold-seconds",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_dir: Path = args.run_dir

    snapshot_dir = (
        run_dir / "snapshots"
    )

    snapshot_paths = sorted(
        snapshot_dir.glob(
            "snapshot_*.npz"
        )
    )

    if not snapshot_paths:
        raise FileNotFoundError(
            f"No snapshots found in "
            f"{snapshot_dir}"
        )

    output_path = (
        args.output
        if args.output is not None
        else run_dir
        / "incremental_mapping_demo.mp4"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Build fixed camera from the FINAL map.
    # -----------------------------------------------------

    final_snapshot = load_snapshot(
        snapshot_paths[-1]
    )

    final_points = final_snapshot[
        "points"
    ]

    if len(final_points) == 0:
        raise ValueError(
            "Final snapshot contains no map points."
        )

    renderer = SemanticMapRenderer(
        width=900,
        height=800,
        point_size=args.point_size,
        final_points=final_points,
    )

    palette = create_semantic_palette()

    # imageio-ffmpeg writes MP4 directly.
    writer = imageio.get_writer(
        str(output_path),
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
    )

    last_video_frame = None

    try:
        for frame_index, snapshot_path in enumerate(
            snapshot_paths
        ):
            
            snapshot = load_snapshot(
                snapshot_path
            )

            points = snapshot[
                "points"
            ]

            labels = snapshot[
                "labels"
            ]

            semantic_agreement = snapshot[
                "semantic_agreement"
            ]

            mean_model_confidence = snapshot[
                "mean_model_confidence"
            ]

            display_labels = (
                apply_unknown_threshold(
                    labels=labels,
                    semantic_agreement=semantic_agreement,
                    mean_model_confidence=mean_model_confidence,
                    agreement_threshold=(
                        args.agreement_threshold
                    ),
                    confidence_threshold=(
                        args.confidence_threshold
                    ),
                )
            )

            semantic_colors = palette[
                display_labels
            ]

            current_points = snapshot.get(
                "current_points_world",
                None,
            )

            map_image = renderer.render(
                points=points,
                semantic_colors=semantic_colors,
                current_points=current_points,
            )

            if args.layout == "triple":
                frame_id = snapshot_path.stem.split("_")[-1]
                rgb_path = (
                    run_dir
                    / "rgb"
                    / f"rgb_{frame_id}.png"
                )

                semantic_path = (
                    run_dir
                    / "semantic_2d"
                    / (
                        f"semantic_"
                        f"{frame_id}.png"
                    )
                )

                if not rgb_path.exists():
                    raise FileNotFoundError(
                        rgb_path
                    )

                if not semantic_path.exists():
                    raise FileNotFoundError(
                        semantic_path
                    )

                video_frame = (
                    create_triple_frame(
                        rgb_path=rgb_path,
                        semantic_path=semantic_path,
                        map_image=map_image,
                        frame_index=frame_index,
                        total_frames=len(
                            snapshot_paths
                        ),
                        voxel_count=len(points),
                    )
                )

            else:
                current_point_count = (
                    len(current_points)
                    if current_points is not None
                    else 0
                )

                video_frame = (
                    create_map_only_frame(
                        map_image=map_image,
                        frame_index=frame_index,
                        total_frames=len(
                            snapshot_paths
                        ),
                        voxel_count=len(points),
                        current_point_count=(
                            current_point_count
                        ),
                    )
                )

            writer.append_data(
                video_frame
            )

            last_video_frame = (
                video_frame
            )

            unknown_ratio = (
                np.mean(
                    display_labels
                    == UNKNOWN_LABEL_ID
                )
                if len(display_labels) > 0
                else 0.0
            )

            print(
                f"[{frame_index + 1:03d}/"
                f"{len(snapshot_paths):03d}] "
                f"voxels={len(points):,}, "
                f"unknown={unknown_ratio:.2%}"
            )

        # Hold final map for a few seconds.
        if last_video_frame is not None:
            hold_frames = int(
                round(
                    args.final_hold_seconds
                    * args.fps
                )
            )

            for _ in range(hold_frames):
                writer.append_data(
                    last_video_frame
                )

    finally:
        writer.close()

    print()
    print(
        f"Video saved to: {output_path}"
    )


if __name__ == "__main__":
    main()