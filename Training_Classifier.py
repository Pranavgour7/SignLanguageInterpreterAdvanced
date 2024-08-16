import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Define a fixed length (e.g., for 21 hand landmarks with 2 coordinates each: 21 * 2 = 42)
fixed_length = 24  # Adjust this based on the expected number of coordinates

# Pad or truncate each data sequence to the fixed length
def pad_or_truncate(sequence, length):
    if len(sequence) > length:
        return sequence[:length]  # Truncate
    else:
        return sequence + [0] * (length - len(sequence))  # Pad with zeros

# Apply padding/truncating to all data
data = np.array([pad_or_truncate(seq, fixed_length) for seq in data])

# Proceed with the rest of the pipeline
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Apply PCA
pca = PCA(n_components=24)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Train the model
model = RandomForestClassifier()
model.fit(x_train_pca, y_train)

# Predict on the test set
y_predict = model.predict(x_test_pca)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model and PCA
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'pca': pca}, f)
