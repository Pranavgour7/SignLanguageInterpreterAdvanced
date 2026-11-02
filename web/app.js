/**
 * Main application: connects webcam → MediaPipe Hands → RandomForest classifier → UI
 */
(function () {
    'use strict';

    // ── DOM Elements ──
    const videoEl       = document.getElementById('webcam');
    const canvasEl      = document.getElementById('overlay');
    const placeholder   = document.getElementById('videoPlaceholder');
    const startBtn      = document.getElementById('startBtn');
    const stopBtn       = document.getElementById('stopBtn');
    const statusDot     = document.getElementById('statusDot');
    const statusText    = document.getElementById('statusText');
    const predValue     = document.getElementById('predictionValue');
    const predConf      = document.getElementById('predictionConfidence');
    const sentenceValue = document.getElementById('sentenceValue');
    const clearBtn      = document.getElementById('clearSentence');
    const fpsValue      = document.getElementById('fpsValue');

    const ctx = canvasEl.getContext('2d');

    // ── State ──
    let stream = null;
    let camera = null;
    let handsDetector = null;
    let animationId = null;
    let running = false;

    // Sentence builder
    let sentence = [];
    let lastPrediction = null;
    let stableCount = 0;
    const STABLE_THRESHOLD = 25; // ~1.5s at 16ms/frame

    // FPS counter
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // ── Model ──
    const model = new RandomForestClassifier();

    // ── Hand drawing config ──
    const HAND_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],       // thumb
        [0,5],[5,6],[6,7],[7,8],       // index
        [0,9],[9,10],[10,11],[11,12],  // middle (5→9 correction: 0→9)
        [0,13],[13,14],[14,15],[15,16],// ring
        [0,17],[17,18],[18,19],[19,20],// pinky
        [5,9],[9,13],[13,17]           // palm
    ];

    // ── Init ──
    async function init() {
        setStatus('loading', 'Loading ML model...');

        try {
            await model.load('model.json');
            setStatus('loading', 'Initializing hand detector...');
            await initHands();
            setStatus('ready', 'Ready — click "Start Camera"');
            startBtn.disabled = false;
        } catch (err) {
            setStatus('error', `Error: ${err.message}`);
            console.error(err);
        }
    }

    function initHands() {
        return new Promise((resolve, reject) => {
            try {
                handsDetector = new Hands({
                    locateFile: (file) =>
                        `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
                });

                handsDetector.setOptions({
                    maxNumHands: 1,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                handsDetector.onResults(onHandResults);

                // Send a tiny dummy frame to trigger WASM initialization
                const dummyCanvas = document.createElement('canvas');
                dummyCanvas.width = 2;
                dummyCanvas.height = 2;
                handsDetector.send({ image: dummyCanvas }).then(resolve).catch(reject);
            } catch (err) {
                reject(err);
            }
        });
    }

    // ── Camera Control ──
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 },
                audio: false
            });
            videoEl.srcObject = stream;
            await videoEl.play();

            canvasEl.width = videoEl.videoWidth || 640;
            canvasEl.height = videoEl.videoHeight || 480;

            placeholder.classList.add('hidden');
            startBtn.style.display = 'none';
            stopBtn.style.display = '';
            running = true;

            setStatus('ready', '🟢 Detecting — show a sign!');
            processFrame();
        } catch (err) {
            setStatus('error', `Camera error: ${err.message}`);
        }
    }

    function stopCamera() {
        running = false;
        if (animationId) cancelAnimationFrame(animationId);
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        videoEl.srcObject = null;
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        placeholder.classList.remove('hidden');
        startBtn.style.display = '';
        stopBtn.style.display = 'none';
        setStatus('ready', 'Camera stopped. Click "Start Camera" to resume.');
        predValue.textContent = '—';
        predConf.textContent = '';
    }

    // ── Frame Processing Loop ──
    async function processFrame() {
        if (!running) return;

        try {
            await handsDetector.send({ image: videoEl });
        } catch (e) {
            // Occasionally MediaPipe throws on dropped frames; just skip
        }

        // FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            fpsValue.textContent = frameCount;
            frameCount = 0;
            lastFpsTime = now;
        }

        animationId = requestAnimationFrame(processFrame);
    }

    // ── MediaPipe Results Callback ──
    function onHandResults(results) {
        const w = canvasEl.width;
        const h = canvasEl.height;
        ctx.clearRect(0, 0, w, h);

        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            predValue.textContent = '—';
            predConf.textContent = 'No hand detected';
            lastPrediction = null;
            stableCount = 0;
            return;
        }

        const landmarks = results.multiHandLandmarks[0];

        // Draw landmarks & connections
        drawHand(landmarks, w, h);

        // Extract features (same as Python: x,y pairs flattened)
        const dataAux = [];
        for (const lm of landmarks) {
            dataAux.push(lm.x);
            dataAux.push(lm.y);
        }

        // Pad or truncate to 25
        const FIXED_LENGTH = 25;
        let features;
        if (dataAux.length < FIXED_LENGTH) {
            features = [...dataAux, ...new Array(FIXED_LENGTH - dataAux.length).fill(0)];
        } else {
            features = dataAux.slice(0, FIXED_LENGTH);
        }

        // Classify
        const result = model.predict(features);
        const pct = (result.confidence * 100).toFixed(1);

        predValue.textContent = result.label;
        predConf.textContent = `Confidence: ${pct}%`;

        // Draw bounding box + label on canvas
        drawBoundingBox(landmarks, result.label, pct, w, h);

        // Sentence builder: if same sign is held for STABLE_THRESHOLD frames, add it
        if (result.label === lastPrediction) {
            stableCount++;
            if (stableCount === STABLE_THRESHOLD) {
                addToSentence(result.label);
            }
        } else {
            lastPrediction = result.label;
            stableCount = 0;
        }
    }

    // ── Drawing Helpers ──
    function drawHand(landmarks, w, h) {
        // Draw connections
        ctx.strokeStyle = 'rgba(108, 92, 231, 0.7)';
        ctx.lineWidth = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
            ctx.beginPath();
            ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
            ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
            ctx.stroke();
        }

        // Draw landmarks
        for (const lm of landmarks) {
            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#a29bfe';
            ctx.fill();
            ctx.strokeStyle = '#6c5ce7';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    function drawBoundingBox(landmarks, label, confidence, w, h) {
        const xs = landmarks.map(l => l.x * w);
        const ys = landmarks.map(l => l.y * h);
        const pad = 20;
        const x1 = Math.min(...xs) - pad;
        const y1 = Math.min(...ys) - pad;
        const x2 = Math.max(...xs) + pad;
        const y2 = Math.max(...ys) + pad;

        // Box
        ctx.strokeStyle = '#6c5ce7';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);

        // Label background
        const text = `${label}  ${confidence}%`;
        ctx.font = 'bold 16px Inter, sans-serif';
        const tm = ctx.measureText(text);
        const lbH = 28;
        ctx.fillStyle = 'rgba(108, 92, 231, 0.85)';
        ctx.beginPath();
        ctx.roundRect(x1, y1 - lbH - 4, tm.width + 16, lbH, 6);
        ctx.fill();

        // Label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 8, y1 - 10);
    }

    // ── Sentence Builder ──
    function addToSentence(label) {
        // Don't add duplicates consecutively
        if (sentence.length > 0 && sentence[sentence.length - 1] === label) return;
        sentence.push(label);
        renderSentence();
    }

    function renderSentence() {
        if (sentence.length === 0) {
            sentenceValue.innerHTML = '<span class="sentence-placeholder">Signs will appear here as you make them...</span>';
        } else {
            sentenceValue.textContent = sentence.join(' ');
        }
    }

    function clearSentenceHandler() {
        sentence = [];
        renderSentence();
    }

    // ── Status Helper ──
    function setStatus(type, text) {
        statusDot.className = 'status-dot';
        if (type === 'ready') statusDot.classList.add('ready');
        else if (type === 'error') statusDot.classList.add('error');
        statusText.textContent = text;
    }

    // ── Event Listeners ──
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    clearBtn.addEventListener('click', clearSentenceHandler);

    // ── Boot ──
    init();
})();

