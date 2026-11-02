/**
 * SignLens — Enhanced Position- & Scale-Invariant Random Forest Classifier
 * Evaluates 100 decision trees in <0.3ms with wrist-relative 42-feature normalization.
 */
class RandomForestClassifier {
    constructor() {
        this.trees = null;
        this.classes = null;
        this.loaded = false;
    }

    async load(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load model: ${response.statusText}`);
        const data = await response.json();

        this.classes = data.classes;
        this.trees = data.trees;
        this.loaded = true;
        console.log(`[SignLens] Enhanced Model loaded: ${this.trees.length} trees, ${this.classes.length} classes`);
    }

    /**
     * Position- & scale-invariant normalization:
     * Centers landmarks relative to the wrist (landmark 0) and scales by hand size.
     * Allows user to sign anywhere in the webcam view, close or far!
     */
    normalize(rawFeatures) {
        const xs = new Float32Array(21);
        const ys = new Float32Array(21);

        for (let i = 0; i < 21; i++) {
            xs[i] = rawFeatures[i * 2];
            ys[i] = rawFeatures[i * 2 + 1];
        }

        let minX = xs[0], maxX = xs[0];
        let minY = ys[0], maxY = ys[0];
        for (let i = 1; i < 21; i++) {
            if (xs[i] < minX) minX = xs[i];
            if (xs[i] > maxX) maxX = xs[i];
            if (ys[i] < minY) minY = ys[i];
            if (ys[i] > maxY) maxY = ys[i];
        }

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const scale = rangeX > rangeY ? (rangeX > 1e-4 ? rangeX : 1e-4) : (rangeY > 1e-4 ? rangeY : 1e-4);

        const wristX = xs[0];
        const wristY = ys[0];
        const normalized = new Float32Array(42);

        for (let i = 0; i < 21; i++) {
            normalized[i * 2] = (xs[i] - wristX) / scale;
            normalized[i * 2 + 1] = (ys[i] - wristY) / scale;
        }

        return normalized;
    }

    /**
     * Traverse a decision tree to retrieve class votes
     */
    traverseTree(tree, features) {
        let node = 0;
        const left = tree.children_left;
        const right = tree.children_right;
        const feature = tree.feature;
        const threshold = tree.threshold;

        while (left[node] !== -1) {
            node = features[feature[node]] <= threshold[node] ? left[node] : right[node];
        }
        return tree.value[node];
    }

    /**
     * Real-time inference across all 100 trees
     * @param {number[]} rawFeatures - 42 elements (21 x,y pairs)
     */
    predict(rawFeatures) {
        if (!this.loaded) throw new Error('Model not loaded');

        // Normalize features
        const features = this.normalize(rawFeatures);
        const numClasses = this.classes.length;
        const totalVotes = new Float32Array(numClasses);

        for (let t = 0; t < this.trees.length; t++) {
            const votes = this.traverseTree(this.trees[t], features);
            for (let c = 0; c < numClasses; c++) {
                totalVotes[c] += votes[c];
            }
        }

        let maxIdx = 0;
        let maxVotes = totalVotes[0];
        let sumVotes = totalVotes[0];

        for (let i = 1; i < numClasses; i++) {
            sumVotes += totalVotes[i];
            if (totalVotes[i] > maxVotes) {
                maxVotes = totalVotes[i];
                maxIdx = i;
            }
        }

        const confidence = sumVotes > 0 ? maxVotes / sumVotes : 0;

        return {
            prediction: this.classes[maxIdx],
            label: this.classes[maxIdx],
            confidence: confidence
        };
    }
}

if (typeof window !== 'undefined') {
    window.RandomForestClassifier = RandomForestClassifier;
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RandomForestClassifier };
}
