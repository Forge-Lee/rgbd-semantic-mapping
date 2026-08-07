import numpy as np


COARSE_CLASS_NAMES = [
    "unknown",
    "other",
    "wall_structure",
    "ground_surface",
    "ceiling",
    "opening",
    "work_surface",
    "seating",
    "storage",
    "display_device",
    "person",
    "appliance_fixture",
    "small_object",
    "decor_soft",
    "navigation_structure",
]


COARSE_NAME_TO_ID = {
    class_name: class_id
    for class_id, class_name
    in enumerate(COARSE_CLASS_NAMES)
}

COARSE_GROUPS = {
    "wall_structure": {
        "wall",
        "column",
    },

    "ground_surface": {
        "floor",
        "rug",
    },

    "ceiling": {
        "ceiling",
    },

    "opening": {
        "door",
        "windowpane",
        "screen door",
    },

    "work_surface": {
        "table",
        "desk",
        "counter",
        "countertop",
        "coffee table",
        "kitchen island",
        "bar",
        "pool table",
    },

    "seating": {
        "chair",
        "sofa",
        "armchair",
        "seat",
        "bench",
        "swivel chair",
        "ottoman",
        "stool",
    },

    "storage": {
        "cabinet",
        "shelf",
        "wardrobe",
        "chest of drawers",
        "case",
        "bookcase",
        "buffet",
    },

    "display_device": {
        "computer",
        "television receiver",
        "screen",
        "crt screen",
        "monitor",
        "arcade machine",
    },

    "person": {
        "person",
    },

    "appliance_fixture": {
        "lamp",
        "light",
        "chandelier",
        "sconce",
        "sink",
        "bathtub",
        "toilet",
        "refrigerator",
        "stove",
        "oven",
        "microwave",
        "dishwasher",
        "washer",
        "shower",
        "radiator",
        "fan",
        "fireplace",
    },

    "small_object": {
        "box",
        "book",
        "bottle",
        "bag",
        "glass",
        "basket",
        "tray",
        "plate",
        "pot",
        "food",
        "towel",
        "apparel",
        "plaything",
        "ball",
        "ashcan",
    },

    "decor_soft": {
        "painting",
        "plant",
        "curtain",
        "mirror",
        "cushion",
        "pillow",
        "flower",
        "blind",
        "blanket",
        "sculpture",
        "poster",
        "bulletin board",
        "vase",
        "clock",
        "flag",
    },

    "navigation_structure": {
        "railing",
        "stairs",
        "stairway",
        "escalator",
        "bannister",
        "step",
    },
}

def normalize_label_name(
    label_name: str,
) -> str:
    return label_name.strip().lower()


def build_fine_to_coarse_mapping(
    coarse_groups: dict[str, set[str]],
) -> dict[str, str]:
    fine_to_coarse: dict[str, str] = {}

    for coarse_name, fine_names in (
        coarse_groups.items()
    ):
        if coarse_name not in COARSE_NAME_TO_ID:
            raise ValueError(
                f"Unknown coarse class: {coarse_name}"
            )

        for fine_name in fine_names:
            normalized_name = normalize_label_name(
                fine_name
            )

            if normalized_name in fine_to_coarse:
                previous_group = fine_to_coarse[
                    normalized_name
                ]

                raise ValueError(
                    "Fine label is mapped more than once: "
                    f"{normalized_name!r} belongs to both "
                    f"{previous_group!r} and "
                    f"{coarse_name!r}"
                )

            fine_to_coarse[normalized_name] = (
                coarse_name
            )

    return fine_to_coarse


FINE_NAME_TO_COARSE_NAME = (
    build_fine_to_coarse_mapping(
        COARSE_GROUPS
    )
)

class CoarseLabelRemapper:
    def __init__(
        self,
        fine_id_to_label: dict[int, str],
    ) -> None:
        if not fine_id_to_label:
            raise ValueError(
                "fine_id_to_label cannot be empty."
            )

        fine_ids = [
            int(fine_id)
            for fine_id
            in fine_id_to_label.keys()
        ]

        if min(fine_ids) < 0:
            raise ValueError(
                "Fine label IDs cannot be negative."
            )

        # More robust than len(fine_id_to_label).
        # This still works if label IDs are not contiguous.
        lookup_table_size = max(fine_ids) + 1

        self.unknown_id = (
            COARSE_NAME_TO_ID["unknown"]
        )

        self.other_id = (
            COARSE_NAME_TO_ID["other"]
        )

        self.fine_to_coarse = np.full(
            lookup_table_size,
            fill_value=self.other_id,
            dtype=np.int64,
        )

        for fine_id, fine_name in (
            fine_id_to_label.items()
        ):
            fine_id = int(fine_id)

            if not isinstance(fine_name, str):
                raise TypeError(
                    "Fine label names must be strings, "
                    f"got {type(fine_name)} "
                    f"for ID {fine_id}."
                )

            normalized_name = (
                normalize_label_name(
                    fine_name
                )
            )

            coarse_name = (
                FINE_NAME_TO_COARSE_NAME.get(
                    normalized_name,
                    "other",
                )
            )

            coarse_id = (
                COARSE_NAME_TO_ID[
                    coarse_name
                ]
            )

            self.fine_to_coarse[
                fine_id
            ] = coarse_id

    def remap(
        self,
        fine_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Convert ADE20K fine labels into coarse robotics labels.

        Args:
            fine_labels:
                Integer fine-label array of any shape.

        Returns:
            Coarse-label array with the same shape.
        """
        fine_labels = np.asarray(
            fine_labels
        )

        if not np.issubdtype(
            fine_labels.dtype,
            np.integer,
        ):
            raise ValueError(
                "Fine labels must contain integers."
            )

        if fine_labels.size == 0:
            return np.empty_like(
                fine_labels,
                dtype=np.int64,
            )

        minimum_label = int(
            fine_labels.min()
        )

        maximum_label = int(
            fine_labels.max()
        )

        if minimum_label < 0:
            raise ValueError(
                "Fine labels cannot be negative."
            )

        if maximum_label >= len(
            self.fine_to_coarse
        ):
            raise ValueError(
                "Fine label exceeds lookup-table size: "
                f"max label is {maximum_label}, "
                f"but table size is "
                f"{len(self.fine_to_coarse)}."
            )

        coarse_labels = (
            self.fine_to_coarse[
                fine_labels
            ]
        )

        return coarse_labels

    def get_coarse_name(
        self,
        coarse_id: int,
    ) -> str:
        coarse_id = int(coarse_id)

        if not (
            0
            <= coarse_id
            < len(COARSE_CLASS_NAMES)
        ):
            raise ValueError(
                f"Invalid coarse label ID: {coarse_id}"
            )

        return COARSE_CLASS_NAMES[
            coarse_id
        ]