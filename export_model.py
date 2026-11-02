"""Export scikit-learn RandomForest model and PCA from model.p to JSON for browser use."""

import json
import os
import pickle

import numpy as np


def numpy_to_python(obj):
    """Recursively convert numpy types to plain Python types."""
    if isinstance(obj, np.ndarray):
        return [numpy_to_python(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(x) for x in obj]
    return obj


def export_model(input_path: str, output_path: str) -> None:
    # Load the pickle
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    pca = data["pca"]

    # ---------- PCA ----------
    pca_json = {
        "mean": numpy_to_python(pca.mean_),
        "components": numpy_to_python(pca.components_),
    }

    # ---------- Classes ----------
    classes = [str(c) for c in model.classes_]

    # ---------- Trees ----------
    trees_json = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        # tree.value has shape (n_nodes, 1, n_classes) – squeeze the middle dim
        value = tree.value.squeeze(axis=1)  # -> (n_nodes, n_classes)

        trees_json.append({
            "children_left": numpy_to_python(tree.children_left),
            "children_right": numpy_to_python(tree.children_right),
            "feature": numpy_to_python(tree.feature),
            "threshold": numpy_to_python(tree.threshold),
            "value": numpy_to_python(value),
        })

    # ---------- Assemble & write ----------
    output = {
        "pca": pca_json,
        "classes": classes,
        "trees": trees_json,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    # Print summary
    print(f"Exported {len(trees_json)} trees")
    print(f"Classes: {classes}")
    print(f"PCA mean length: {len(pca_json['mean'])}")
    print(f"PCA components shape: {len(pca_json['components'])}x{len(pca_json['components'][0])}")
    file_size = os.path.getsize(output_path)
    print(f"Output file size: {file_size:,} bytes")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    base = r"c:\Users\Pranav Gour\AI based sign intrepreter 4"
    export_model(
        input_path=os.path.join(base, "model.p"),
        output_path=os.path.join(base, "web", "model.json"),
    )
