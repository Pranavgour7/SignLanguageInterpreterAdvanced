/**
 * Random Forest classifier ported from scikit-learn to JavaScript.
 * Loads the exported model.json and performs PCA + tree-based inference.
 */
class RandomForestClassifier {
    constructor() {
        this.trees = null;
        this.classes = null;
        this.pcaMean = null;
        this.pcaComponents = null;
        this.loaded = false;
    }

    async load(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load model: ${response.statusText}`);
        const data = await response.json();

        this.classes = data.classes;
        this.pcaMean = data.pca.mean;
        this.pcaComponents = data.pca.components;
        this.trees = data.trees;
        this.loaded = true;
        console.log(`Model loaded: ${this.trees.length} trees, ${this.classes.length} classes`);
    }

    /**
     * Apply PCA transformation: (x - mean) @ components.T
     * Since components is already (n_components x n_features),
     * the transform is (x - mean) . components^T, but sklearn stores
     * components as (n_components, n_features), so the dot product
     * of (1, n_features) x (n_features, n_components) = matmul with transposed components.
     * Actually: transform = (X - mean) @ components.T  but components shape is (n_components, n_features)
     * So: result[i] = sum_j (x[j] - mean[j]) * components[i][j]
     */
    applyPCA(features) {
        const centered = features.map((v, i) => v - this.pcaMean[i]);
        const result = new Array(this.pcaComponents.length);
        for (let i = 0; i < this.pcaComponents.length; i++) {
            let sum = 0;
            for (let j = 0; j < centered.length; j++) {
                sum += centered[j] * this.pcaComponents[i][j];
            }
            result[i] = sum;
        }
        return result;
    }

    /**
     * Traverse a single decision tree and return the class vote array.
     */
    traverseTree(tree, features) {
        let node = 0;
        while (tree.children_left[node] !== -1) { // -1 = leaf in sklearn
            if (features[tree.feature[node]] <= tree.threshold[node]) {
                node = tree.children_left[node];
            } else {
                node = tree.children_right[node];
            }
        }
        return tree.value[node]; // array of length n_classes
    }

    /**
     * Predict the class label and confidence for a feature vector.
     * @param {number[]} rawFeatures - 25-element array (padded/truncated landmark coords)
     * @returns {{ label: string, confidence: number, votes: Object }}
     */
    predict(rawFeatures) {
        if (!this.loaded) throw new Error('Model not loaded');

        // Apply PCA
        const pcaFeatures = this.applyPCA(rawFeatures);

        // Aggregate votes from all trees
        const totalVotes = new Array(this.classes.length).fill(0);

        for (const tree of this.trees) {
            const leafVotes = this.traverseTree(tree, pcaFeatures);
            for (let i = 0; i < this.classes.length; i++) {
                totalVotes[i] += leafVotes[i];
            }
        }

        // Find the class with the most votes
        let maxIdx = 0;
        let maxVotes = totalVotes[0];
        let sumVotes = totalVotes[0];
        for (let i = 1; i < totalVotes.length; i++) {
            sumVotes += totalVotes[i];
            if (totalVotes[i] > maxVotes) {
                maxVotes = totalVotes[i];
                maxIdx = i;
            }
        }

        const confidence = sumVotes > 0 ? maxVotes / sumVotes : 0;

        return {
            label: this.classes[maxIdx],
            confidence: confidence,
            votes: Object.fromEntries(this.classes.map((c, i) => [c, totalVotes[i] / sumVotes]))
        };
    }
}

