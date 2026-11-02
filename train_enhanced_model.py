import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load original data
d = pickle.load(open('c:/Users/Pranav Gour/AI based sign intrepreter 4/data.pickle', 'rb'))

def normalize_landmarks(row):
    r = row[:42]
    xs = r[0::2]
    ys = r[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y, 1e-4)
    # Relative to wrist (index 0) and scaled
    norm_x = (np.array(xs) - xs[0]) / scale
    norm_y = (np.array(ys) - ys[0]) / scale
    out = np.empty(42, dtype=np.float32)
    out[0::2] = norm_x
    out[1::2] = norm_y
    return out

X = np.array([normalize_landmarks(x) for x in d['data']])
y = np.array(d['labels'])

# Train RandomForest on full normalized 42 features
clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)
clf.fit(X, y)

print(f"Model trained! Score on train set: {clf.score(X, y):.4f}")
print(f"Classes ({len(clf.classes_)}): {list(clf.classes_)}")

# Export to clean JSON format for JavaScript
trees_data = []
for tree in clf.estimators_:
    tree_dict = {
        "children_left": tree.tree_.children_left.tolist(),
        "children_right": tree.tree_.children_right.tolist(),
        "feature": tree.tree_.feature.tolist(),
        "threshold": tree.tree_.threshold.tolist(),
        "value": tree.tree_.value[:, 0, :].tolist()
    }
    trees_data.append(tree_dict)

model_export = {
    "normalized": True,
    "n_features": 42,
    "classes": clf.classes_.tolist(),
    "trees": trees_data
}

output_path = 'c:/Users/Pranav Gour/AI based sign intrepreter 4/web/model.json'
with open(output_path, 'w') as f:
    json.dump(model_export, f)

print(f"Successfully exported enhanced model to {output_path}")

