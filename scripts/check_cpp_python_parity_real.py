import numpy as np

python_map = np.loadtxt(
    "outputs/parity/python_final_map.csv",
    delimiter=",",
    skiprows=1,
)

cpp_map = np.loadtxt(
    "outputs/parity/cpp_final_map.csv",
    delimiter=",",
    skiprows=1,
)

print("Python shape:", python_map.shape)
print("C++ shape:", cpp_map.shape)

assert python_map.shape == cpp_map.shape

comparison = np.isclose(
    python_map,
    cpp_map,
    # rtol=1e-9,
    # atol=1e-12,
)

print("All close:", np.all(comparison))

column_names = [
    "x",
    "y",
    "z",
    "r",
    "g",
    "b",
    "label",
    "semantic_agreement",
    "mean_model_confidence",
    "observation_count",
]

diff = np.abs(python_map - cpp_map)

for i, name in enumerate(column_names):
    print(
        f"{name:24s} "
        f"max_abs_diff={np.max(diff[:, i]):.12e} "
        f"mean_abs_diff={np.mean(diff[:, i]):.12e}"
    )

print(
    "Labels exactly equal:",
    np.array_equal(
        python_map[:, 6].astype(np.int64),
        cpp_map[:, 6].astype(np.int64),
    )
)

print(
    "Counts exactly equal:",
    np.array_equal(
        python_map[:, 9].astype(np.int64),
        cpp_map[:, 9].astype(np.int64),
    )
)

strict_close = np.isclose(
    python_map,
    cpp_map,
    rtol=1e-9,
    atol=1e-12,
)

print(
    "Strict match ratio:",
    np.mean(strict_close)
)

print(
    "Number of strict mismatches:",
    np.count_nonzero(~strict_close)
)