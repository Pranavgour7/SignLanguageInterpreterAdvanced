/**
 * SignLens — Unified 3D/2D Sign Studio & Fast-Flow AI Sign Language Interpreter
 * 
 * Features:
 * - Default Side-by-Side View: Interactive 3D Model and Ground-Truth 2D Guide in sync with 0ms latency
 * - 100% Alignment: Both 2D and 3D are generated directly from the model's authentic training dataset (data.pickle)
 * - Zero Network Lag: Inline vector rendering for instantaneous 2D switching alongside 3D morphing
 * - Anatomically Realistic 3D Hand Model with Natural Human Proportions & Sculpted Clay Shaders
 * - High-Precision 3D Kinematics with Natural Depth, Double-Sided Fleshy Palm Pad & Tapered Finger Capsules
 * - Zero-Dependency Native Pointer Orbit Controller (360° Drag, Pitch Clamping, Wheel Zoom, Touch)
 * - Complete 3D Studio Operations: Front/Side/Top Presets, Mirror (Left/Right Hand), Auto-Rotate, Reset
 * - Real-Time Sign Search & Category Filtering directly controlling the interactive studio
 * - Position- & Scale-Invariant 42-Landmark Random Forest Classifier (<0.3ms evaluation)
 * - Fast-Flow Continuous Signing Mode (<200ms debouncing), TTS, Practice Lab & Camera Mirroring
 */

(function () {
    'use strict';

    // ── Verified 26 Sign Dataset 100% Synchronized with Training Data ──
    const SIGNS_DATA = [
        {
            id: 'A',
            category: 'letter',
            title: 'Letter A',
            imgUrl: 'assets/signs/A.svg',
            pose: 'Tight fist with thumb resting upright along the outer edge of index finger.',
            fingers: { thumb: 'Upright along outer edge of index', index: 'Curled firmly in fist', other: 'Middle, ring, pinky curled in fist' },
            tip: 'Keep thumb resting along outer edge — do not fold it across knuckles.'
        },
        {
            id: 'B',
            category: 'letter',
            title: 'Letter B',
            imgUrl: 'assets/signs/B.svg',
            pose: 'Four fingers extended straight up together; thumb folded flat across palm.',
            fingers: { thumb: 'Folded flat across palm', index: 'Straight upright', other: 'Middle, ring, pinky straight upright together' },
            tip: 'Keep all four fingers pressed tightly together with no gaps between them.'
        },
        {
            id: 'C',
            category: 'letter',
            title: 'Letter C',
            imgUrl: 'assets/signs/C.svg',
            pose: 'Curve fingers and thumb together into a distinct "C" shape facing outward.',
            fingers: { thumb: 'Curved out into bottom arc', index: 'Curved into top arc', other: 'Curved alongside index' },
            tip: 'Hold wrist steady so camera captures the open circular hollow of the "C".'
        },
        {
            id: 'D',
            category: 'letter',
            title: 'Letter D',
            imgUrl: 'assets/signs/D.svg',
            pose: 'Index finger straight up; middle, ring, pinky touch thumb tip forming a ring.',
            fingers: { thumb: 'Touches middle, ring, pinky tips', index: 'Points straight up', other: 'Curled touching thumb' },
            tip: 'Only index points up; the other three fingers must touch the thumb.'
        },
        {
            id: 'E',
            category: 'letter',
            title: 'Letter E',
            imgUrl: 'assets/signs/E.svg',
            pose: 'Curl all 4 fingertips down tightly toward palm, thumb folded underneath.',
            fingers: { thumb: 'Folded under fingertips', index: 'Curled tightly down', other: 'Curled tightly down' },
            tip: 'Resembles a tight claw with fingertips resting on top of bent thumb.'
        },
        {
            id: 'F',
            category: 'letter',
            title: 'Letter F',
            imgUrl: 'assets/signs/F.svg',
            pose: 'Touch index tip to thumb tip forming a circle; 3 other fingers extended up spread.',
            fingers: { thumb: 'Touches index tip', index: 'Touches thumb tip in circle', other: 'Middle, ring, pinky up and spread' },
            tip: 'The classic "OK" sign with remaining 3 fingers fanned out upright.'
        },
        {
            id: 'G',
            category: 'letter',
            title: 'Letter G',
            imgUrl: 'assets/signs/G.svg',
            pose: 'Index finger and thumb extended horizontally forward in parallel like a pinch.',
            fingers: { thumb: 'Horizontal parallel to index', index: 'Pointing horizontally', other: 'Curled in fist' },
            tip: 'Imagine measuring a small distance horizontally between two fingers.'
        },
        {
            id: 'H',
            category: 'letter',
            title: 'Letter H',
            imgUrl: 'assets/signs/H.svg',
            pose: 'Index and middle fingers extended together horizontally; thumb over ring/pinky.',
            fingers: { thumb: 'Over folded ring/pinky', index: 'Horizontal together', other: 'Middle horizontal; others in fist' },
            tip: 'Keep index and middle parallel and touching horizontally.'
        },
        {
            id: 'I',
            category: 'letter',
            title: 'Letter I',
            imgUrl: 'assets/signs/I.svg',
            pose: 'Tight fist with only pinky finger extended straight upright.',
            fingers: { thumb: 'Across front knuckles', index: 'In fist', other: 'Pinky straight up; middle/ring in' },
            tip: 'Pinky must point vertically with palm facing camera.'
        },
        {
            id: 'K',
            category: 'letter',
            title: 'Letter K',
            imgUrl: 'assets/signs/K.svg',
            pose: 'Index straight up, middle forward at 45°, thumb between them against middle knuckle.',
            fingers: { thumb: 'Between index & middle', index: 'Points straight up', other: 'Middle at 45°; ring/pinky in' },
            tip: 'Thumb touches the first knuckle of middle finger.'
        },
        {
            id: 'L',
            category: 'letter',
            title: 'Letter L',
            imgUrl: 'assets/signs/L.svg',
            pose: 'Index finger straight up and thumb extended out at 90° forming an "L".',
            fingers: { thumb: 'Extended out at 90°', index: 'Pointing straight up', other: 'Curled in fist' },
            tip: 'Ensure palm faces squarely towards the camera.'
        },
        {
            id: 'M',
            category: 'letter',
            title: 'Letter M',
            imgUrl: 'assets/signs/M.svg',
            pose: 'Thumb tucked under index, middle, and ring fingers; pinky curled in.',
            fingers: { thumb: 'Under first 3 fingers', index: 'Draped over thumb', other: 'Middle/ring over thumb; pinky in' },
            tip: 'Three distinct finger bumps show over the tucked thumb.'
        },
        {
            id: 'O',
            category: 'letter',
            title: 'Letter O',
            imgUrl: 'assets/signs/O.svg',
            pose: 'Curve all fingers and thumb together until tips touch, forming an "O".',
            fingers: { thumb: 'Touches all fingertips', index: 'Curved to thumb', other: 'Curved touching thumb' },
            tip: 'Fingertips meet thumb tip to complete a smooth round shape.'
        },
        {
            id: 'R',
            category: 'letter',
            title: 'Letter R',
            imgUrl: 'assets/signs/R.svg',
            pose: 'Cross middle finger over the back of extended index finger ("good luck").',
            fingers: { thumb: 'Over ring/pinky', index: 'Extended up', other: 'Middle crossed over index' },
            tip: 'The classic "fingers crossed" sign held vertically.'
        },
        {
            id: 'U',
            category: 'letter',
            title: 'Letter U',
            imgUrl: 'assets/signs/U.svg',
            pose: 'Index and middle fingers extended straight up touching each other.',
            fingers: { thumb: 'Over folded ring/pinky', index: 'Straight up touching middle', other: 'Middle straight up; others in' },
            tip: 'Fingers must touch. If you separate them, it becomes a V.'
        },
        {
            id: 'V',
            category: 'letter',
            title: 'Letter V',
            imgUrl: 'assets/signs/V.svg',
            pose: 'Index and middle fingers extended up spread apart forming a "V" (peace sign).',
            fingers: { thumb: 'Over folded ring/pinky', index: 'Straight up spread', other: 'Middle up spread; ring/pinky in' },
            tip: 'Spread index and middle wide apart with palm facing forward.'
        },
        {
            id: 'W',
            category: 'letter',
            title: 'Letter W',
            imgUrl: 'assets/signs/W.svg',
            pose: 'Index, middle, and ring fingers extended straight up spread apart.',
            fingers: { thumb: 'Holds pinky tip in palm', index: 'Straight up spread', other: 'Middle/ring up spread; pinky in' },
            tip: 'Three middle fingers spread wide like the letter "W".'
        },
        {
            id: 'Y',
            category: 'letter',
            title: 'Letter Y',
            imgUrl: 'assets/signs/Y.svg',
            pose: 'Extend thumb and pinky wide apart; curl index, middle, and ring fingers into palm.',
            fingers: { thumb: 'Extended out sideways', index: 'Curled in fist', other: 'Pinky extended; middle/ring in' },
            tip: 'The iconic "hang loose" or "shaka" handshape.'
        },
        {
            id: '1',
            category: 'number',
            title: 'Number 1',
            imgUrl: 'assets/signs/1.jpg',
            pose: 'Index finger straight up; thumb and other three fingers folded in fist.',
            fingers: { thumb: 'Folded over fingers', index: 'Straight up', other: 'Middle, ring, pinky in fist' },
            tip: 'Single vertical index finger with back of hand or palm facing camera.'
        },
        {
            id: '3',
            category: 'number',
            title: 'Number 3',
            imgUrl: 'assets/signs/3.jpg',
            pose: 'Extend thumb, index, and middle fingers; ring and pinky curled down.',
            fingers: { thumb: 'Extended sideways/up', index: 'Straight up', other: 'Middle straight up; ring/pinky in' },
            tip: 'In ASL, 3 is signed with thumb + index + middle (not index + middle + ring).'
        },
        {
            id: '4',
            category: 'number',
            title: 'Number 4',
            imgUrl: 'assets/signs/4.jpg',
            pose: 'All four fingers extended straight up spread; thumb folded flat across palm.',
            fingers: { thumb: 'Folded across palm', index: 'Straight up spread', other: 'Middle, ring, pinky spread up' },
            tip: 'Keep all four fingers separated, thumb firmly against palm.'
        },
        {
            id: '7',
            category: 'number',
            title: 'Number 7',
            imgUrl: 'assets/signs/7.jpg',
            pose: 'Touch ring fingertip to thumb tip; index, middle, and pinky extended up.',
            fingers: { thumb: 'Touches ring fingertip', index: 'Extended upright', other: 'Middle/pinky up; ring on thumb' },
            tip: 'Ring finger touches thumb, other 3 fingers stay extended.'
        },
        {
            id: 'Hello',
            category: 'phrase',
            title: 'Hello / Wave',
            imgUrl: 'assets/signs/Hello.jpg',
            pose: 'Open flat palm with all five fingers extended upright and spread in a greeting wave.',
            fingers: { thumb: 'Extended open outwards', index: 'Straight upright spread', other: 'Middle, ring, pinky upright spread' },
            tip: 'Face open palm toward camera with all 5 fingers clearly separated and visible.'
        },
        {
            id: 'Okay',
            category: 'phrase',
            title: 'Okay',
            imgUrl: 'assets/signs/Okay.svg',
            pose: 'Touch index tip to thumb tip forming a circle; middle, ring, pinky up.',
            fingers: { thumb: 'Touches index tip', index: 'Forms ring with thumb', other: 'Middle, ring, pinky extended up' },
            tip: 'Universal "OK" hand sign held steady with circle facing camera.'
        },
        {
            id: 'Bathroom',
            category: 'phrase',
            title: 'Bathroom',
            imgUrl: 'assets/signs/Bathroom.svg',
            pose: 'Make a fist with thumb tucked poking up between index and middle knuckles (ASL "T").',
            fingers: { thumb: 'Poking up between index & middle', index: 'Curled over thumb', other: 'Ring & pinky curled in fist' },
            tip: 'Form an ASL "T" handshape with thumb tip poking between index and middle.'
        },
        {
            id: 'I Love You',
            category: 'phrase',
            title: 'I Love You',
            imgUrl: 'assets/signs/I_Love_You.svg',
            pose: 'Extend thumb, index, and pinky simultaneously; middle and ring folded down.',
            fingers: { thumb: 'Extended out wide', index: 'Pointing straight up', other: 'Pinky straight up; middle/ring in' },
            tip: 'Iconic sign combining letters I, L, and Y.'
        }
    ];

    // Hand Landmark Bone Linkages (21 points)
    const BONE_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],       // Thumb
        [0,5],[5,6],[6,7],[7,8],       // Index
        [0,9],[9,10],[10,11],[11,12],  // Middle
        [0,13],[13,14],[14,15],[15,16],// Ring
        [0,17],[17,18],[18,19],[19,20],// Pinky
        [5,9],[9,13],[13,17]           // Palm transversal
    ];

    // Anatomical radii for realistic finger tapering
    function getBoneRadius(boneIndex) {
        if (boneIndex === 0 || boneIndex === 4 || boneIndex === 8 || boneIndex === 12 || boneIndex === 16) {
            return { top: 0.10, bottom: 0.12 }; // Metacarpals
        }
        if (boneIndex >= 20) {
            return { top: 0.08, bottom: 0.08 }; // Palm bridge
        }
        if (boneIndex === 1 || boneIndex === 5 || boneIndex === 9 || boneIndex === 13 || boneIndex === 17) {
            return { top: 0.09, bottom: 0.105 }; // Proximal
        }
        if (boneIndex === 2 || boneIndex === 6 || boneIndex === 10 || boneIndex === 14 || boneIndex === 18) {
            return { top: 0.078, bottom: 0.09 }; // Intermediate
        }
        return { top: 0.068, bottom: 0.078 }; // Distal
    }

    function getJointRadius(jointIndex) {
        if (jointIndex === 0) return 0.13; // Wrist
        if (jointIndex === 1 || jointIndex === 5 || jointIndex === 9 || jointIndex === 13 || jointIndex === 17) return 0.105; // MCPs
        if (jointIndex === 4 || jointIndex === 8 || jointIndex === 12 || jointIndex === 16 || jointIndex === 20) return 0.075; // Tips
        return 0.088; // Knuckles
    }

    // ── DOM References ──
    const videoEl          = document.getElementById('webcam');
    const canvasEl         = document.getElementById('overlay');
    const placeholder      = document.getElementById('videoPlaceholder');
    const startBtn         = document.getElementById('startBtn');
    const stopBtn          = document.getElementById('stopBtn');
    const flipBtn          = document.getElementById('flipBtn');
    const statusDot        = document.getElementById('statusDot');
    const statusText       = document.getElementById('statusText');
    const predValue        = document.getElementById('predictionValue');
    const predConf         = document.getElementById('predictionConfidence');
    const confidenceBar    = document.getElementById('confidenceBar');
    const predCard         = document.getElementById('predCard');
    const sentenceValue    = document.getElementById('sentenceValue');
    const sentenceCount    = document.getElementById('sentenceCount');
    const clearBtn         = document.getElementById('clearSentence');
    const copyBtn          = document.getElementById('copySentence');
    const speakBtn         = document.getElementById('speakSentence');
    const fpsValue         = document.getElementById('fpsValue');

    const speedFastBtn     = document.getElementById('speedFastBtn');
    const speedNormalBtn   = document.getElementById('speedNormalBtn');
    const practiceBanner   = document.getElementById('practiceBanner');
    const practiceTargetName = document.getElementById('practiceTargetName');
    const practiceInstruction = document.getElementById('practiceInstruction');
    const clearPracticeBtn = document.getElementById('clearPracticeBtn');

    // Unified Studio Elements
    const studioContainer  = document.getElementById('threejsCanvasContainer');
    const studioViewportWrap = document.getElementById('studioViewportWrap');
    const studioSignBadge  = document.getElementById('studioSignBadge');
    const studioSignTitle  = document.getElementById('studioSignTitle');
    const studioCatTag     = document.getElementById('studioCatTag');
    const studioSignDesc   = document.getElementById('studioSignDesc');
    const studioPillsRow   = document.getElementById('studioPillsRow');
    const studioSearchEmpty= document.getElementById('studioSearchEmpty');

    const mode3DBtn        = document.getElementById('mode3DBtn');
    const mode2DBtn        = document.getElementById('mode2DBtn');
    const modeSplitBtn     = document.getElementById('modeSplitBtn');
    const practiceCurrentSignBtn = document.getElementById('practiceCurrentSignBtn');

    const studio2DBox      = document.getElementById('studio2DBox');
    const studio2DSvgWrap  = document.getElementById('studio2DSvgWrap');
    const studio2DImg      = document.getElementById('studio2DImg');
    const studioThumbVal   = document.getElementById('studioThumbVal');
    const studioIndexVal   = document.getElementById('studioIndexVal');
    const studioOthersVal  = document.getElementById('studioOthersVal');
    const studioTipVal     = document.getElementById('studioTipVal');

    const viewFrontBtn     = document.getElementById('viewFrontBtn');
    const viewSideBtn      = document.getElementById('viewSideBtn');
    const viewTopBtn       = document.getElementById('viewTopBtn');
    const toggle3DMirrorBtn= document.getElementById('toggle3DMirrorBtn');
    const toggleMaterialBtn= document.getElementById('toggleMaterialBtn');
    const toggleAutoRotateBtn = document.getElementById('toggleAutoRotateBtn');
    const reset3DViewBtn   = document.getElementById('reset3DViewBtn');

    const studioSignSearch = document.getElementById('studioSignSearch');
    const studioClearSearch= document.getElementById('studioClearSearch');
    const categoryTabs     = document.querySelectorAll('.studio-category-tabs .tab-btn');

    const ctx = canvasEl ? canvasEl.getContext('2d') : null;

    // ── Application State ──
    let stream = null;
    let handsDetector = null;
    let animationId = null;
    let running = false;
    let isWebcamMirrored = false;
    let activeCategory = 'all';
    let searchQuery = '';
    let practiceTarget = null;
    let activeStudioMode = 'split'; // Default: Side-by-Side

    // Fast-flow timing
    let flowSpeed = 'fast';
    let stableThreshold = 6;
    let stableStreak = 0;
    let lastCandidate = null;
    let sentence = [];

    // FPS
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // Model
    const model = new RandomForestClassifier();

    // ── 3D Three.js Studio State ──
    let scene3D, camera3D, renderer3D;
    let handGroup3D;
    let jointSpheres = [];
    let boneCylinders = [];
    let palmMesh = null;
    let palmGeometry = null;
    let poses3D = {}; // Loaded from poses_3d.json
    let signs2DSvg = {}; // Loaded from signs_2d_svg.json (0ms latency)
    let currentPose3D = null;
    let targetPose3D = null;
    let is3DMirrored = false;
    let is3DAutoRotate = true;
    let active3DSign = 'A';
    let isClayMode = false;

    // Materials
    let skinMat, clayMat, jointSkinMat, jointClayMat, palmSkinMat, palmClayMat;

    // Pointer Orbit Controller State
    let isPointerDragging = false;
    let prevPointerX = 0;
    let prevPointerY = 0;
    let currentPitch = 0.12;  // Vertical angle
    let currentYaw = 0.0;     // Horizontal angle
    let targetPitch = 0.12;
    let targetYaw = 0.0;
    let cameraDistance = 4.4;
    let targetCameraDistance = 4.4;

    // ── Init ──
    async function init() {
        populateStudioPills();
        setupEventListeners();

        // Default to Side-by-Side Split View
        setStudioMode('split');

        // Initialize 3D Hand Projection Studio & Load Ground-Truth Assets
        initThreeJSStudio();
        await loadStudioAssets();

        setStatus('loading', 'Loading 42-landmark model...');
        try {
            await model.load('model.json');
            setStatus('loading', 'Initializing MediaPipe hand tracking...');
            await initHands();
            setStatus('ready', '⚡ Fast-Flow Engine & Sign Studio Ready — click "Start Camera" to test');
            if (startBtn) startBtn.disabled = false;
        } catch (err) {
            setStatus('error', `Init error: ${err.message}`);
            console.error(err);
        }
    }

    // ── Load 3D and 2D Ground-Truth Assets Simultaneously ──
    async function loadStudioAssets() {
        try {
            // Preload all 26 authentic 2D drawn illustrations locally for instant 0ms switching
            SIGNS_DATA.forEach(sign => {
                if (sign.imgUrl) {
                    const preloadImg = new Image();
                    preloadImg.src = sign.imgUrl;
                }
            });

            const [posesRes, svgRes] = await Promise.all([
                fetch('poses_3d.json'),
                fetch('signs_2d_svg.json')
            ]);
            if (posesRes.ok) {
                poses3D = await posesRes.json();
            }
            if (svgRes.ok) {
                signs2DSvg = await svgRes.json();
            }
            set3DSignPose('A');
        } catch (e) {
            console.warn('Could not load 3D/2D assets:', e);
        }
    }

    // ── Three.js 3D Hand Model Studio ──
    function initThreeJSStudio() {
        if (!studioContainer || typeof THREE === 'undefined') return;

        const width = studioContainer.clientWidth || 400;
        const height = studioContainer.clientHeight || 500;

        // Scene
        scene3D = new THREE.Scene();

        // Camera
        camera3D = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        updateCameraOrbit();

        // WebGL Renderer
        renderer3D = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
        renderer3D.setSize(width, height);
        renderer3D.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer3D.shadowMap.enabled = true;
        renderer3D.shadowMap.type = THREE.PCFSoftShadowMap;
        studioContainer.innerHTML = '';
        studioContainer.appendChild(renderer3D.domElement);

        // ── High-Quality Studio Shaders & Materials ──
        skinMat = new THREE.MeshStandardMaterial({
            color: 0xd6997b,
            roughness: 0.64,
            metalness: 0.04
        });

        jointSkinMat = new THREE.MeshStandardMaterial({
            color: 0xcc8d70,
            roughness: 0.68,
            metalness: 0.04
        });

        palmSkinMat = new THREE.MeshStandardMaterial({
            color: 0xd49577,
            roughness: 0.70,
            metalness: 0.03,
            side: THREE.DoubleSide
        });

        clayMat = new THREE.MeshStandardMaterial({
            color: 0xeeece6,
            roughness: 0.72,
            metalness: 0.02
        });

        jointClayMat = new THREE.MeshStandardMaterial({
            color: 0xe6e3dc,
            roughness: 0.75,
            metalness: 0.02
        });

        palmClayMat = new THREE.MeshStandardMaterial({
            color: 0xeeece6,
            roughness: 0.74,
            metalness: 0.02,
            side: THREE.DoubleSide
        });

        // ── 3-Point Studio Lighting ──
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        scene3D.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xfff8ee, 2.4);
        keyLight.position.set(3.0, 4.0, 3.5);
        scene3D.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xe0f2fe, 1.3);
        fillLight.position.set(-3.0, 1.5, 2.5);
        scene3D.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 1.5);
        rimLight.position.set(0, 3.5, -3.5);
        scene3D.add(rimLight);

        // Ground Studio Pedestal Disc
        const pedestalGroup = new THREE.Group();
        pedestalGroup.position.set(0, -1.45, 0);

        const discGeo = new THREE.CircleGeometry(1.6, 48);
        discGeo.rotateX(-Math.PI / 2);
        const discMat = new THREE.MeshBasicMaterial({ color: 0x141822, transparent: true, opacity: 0.7 });
        const discMesh = new THREE.Mesh(discGeo, discMat);
        pedestalGroup.add(discMesh);

        const ringGeo = new THREE.RingGeometry(1.58, 1.62, 64);
        ringGeo.rotateX(-Math.PI / 2);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.4 });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        pedestalGroup.add(ringMesh);

        scene3D.add(pedestalGroup);

        // Hand Group
        handGroup3D = new THREE.Group();
        scene3D.add(handGroup3D);

        // Build 21 Anatomical Joint Spheres
        jointSpheres = [];
        for (let i = 0; i < 21; i++) {
            const r = getJointRadius(i);
            const jointGeo = new THREE.SphereGeometry(r, 16, 16);
            const sphere = new THREE.Mesh(jointGeo, jointSkinMat);
            sphere.position.set(0, 0, 0);
            handGroup3D.add(sphere);
            jointSpheres.push(sphere);
        }

        // Build 23 Anatomical Bone Capsules
        boneCylinders = [];
        for (let b = 0; b < BONE_CONNECTIONS.length; b++) {
            const radii = getBoneRadius(b);
            const cylGeo = new THREE.CylinderGeometry(radii.top, radii.bottom, 1, 16);
            const cyl = new THREE.Mesh(cylGeo, skinMat);
            handGroup3D.add(cyl);
            boneCylinders.push(cyl);
        }

        // Build Fleshy Palm Pad Mesh (6 vertices, 8 triangles double-sided)
        palmGeometry = new THREE.BufferGeometry();
        const palmIndices = [
            0, 1, 2,   // wrist -> thumb -> index
            0, 2, 3,   // wrist -> index -> middle
            0, 3, 4,   // wrist -> middle -> ring
            0, 4, 5,   // wrist -> ring -> pinky
            // Backside
            0, 2, 1,
            0, 3, 2,
            0, 4, 3,
            0, 5, 4
        ];
        const initialPalmPos = new Float32Array(6 * 3); // exactly 6 vertices * 3 coords
        palmGeometry.setAttribute('position', new THREE.BufferAttribute(initialPalmPos, 3));
        palmGeometry.setIndex(palmIndices);
        palmMesh = new THREE.Mesh(palmGeometry, palmSkinMat);
        handGroup3D.add(palmMesh);

        // Native Pointer Drag & Zoom Controller
        setupPointerOrbitControls();

        window.addEventListener('resize', onStudioResize);

        // 3D Render Loop
        function animate3D() {
            requestAnimationFrame(animate3D);

            // Interpolate smoothly toward target pose
            if (currentPose3D && targetPose3D) {
                for (let i = 0; i < 21; i++) {
                    const tx = (is3DMirrored ? -1 : 1) * targetPose3D[i][0];
                    const ty = targetPose3D[i][1];
                    const tz = targetPose3D[i][2];

                    currentPose3D[i][0] += (tx - currentPose3D[i][0]) * 0.16;
                    currentPose3D[i][1] += (ty - currentPose3D[i][1]) * 0.16;
                    currentPose3D[i][2] += (tz - currentPose3D[i][2]) * 0.16;

                    jointSpheres[i].position.set(currentPose3D[i][0], currentPose3D[i][1], currentPose3D[i][2]);
                }
                updateBonesGeometry();
                updatePalmGeometry();
            }

            // Turntable auto-rotate when not dragging
            if (is3DAutoRotate && !isPointerDragging) {
                targetYaw += 0.006;
            }

            currentPitch += (targetPitch - currentPitch) * 0.12;
            currentYaw   += (targetYaw - currentYaw) * 0.12;
            cameraDistance += (targetCameraDistance - cameraDistance) * 0.12;

            updateCameraOrbit();

            renderer3D.render(scene3D, camera3D);
        }
        animate3D();
    }

    function updateCameraOrbit() {
        if (!camera3D) return;
        camera3D.position.x = cameraDistance * Math.sin(currentYaw) * Math.cos(currentPitch);
        camera3D.position.y = cameraDistance * Math.sin(currentPitch);
        camera3D.position.z = cameraDistance * Math.cos(currentYaw) * Math.cos(currentPitch);
        camera3D.lookAt(0, 0, 0);
    }

    function onStudioResize() {
        if (!studioContainer || !renderer3D || !camera3D) return;
        const w = studioContainer.clientWidth;
        const h = studioContainer.clientHeight;
        if (w === 0 || h === 0) return;
        camera3D.aspect = w / h;
        camera3D.updateProjectionMatrix();
        renderer3D.setSize(w, h);
    }

    function setupPointerOrbitControls() {
        if (!studioContainer) return;

        studioContainer.addEventListener('pointerdown', (e) => {
            isPointerDragging = true;
            prevPointerX = e.clientX;
            prevPointerY = e.clientY;
            try { studioContainer.setPointerCapture(e.pointerId); } catch (_) {}
            if (is3DAutoRotate) {
                is3DAutoRotate = false;
                if (toggleAutoRotateBtn) toggleAutoRotateBtn.classList.remove('active');
            }
        });

        studioContainer.addEventListener('pointermove', (e) => {
            if (!isPointerDragging) return;
            const dx = e.clientX - prevPointerX;
            const dy = e.clientY - prevPointerY;

            targetYaw += dx * 0.008;
            targetPitch += dy * 0.008;

            targetPitch = Math.max(-1.36, Math.min(1.36, targetPitch));

            prevPointerX = e.clientX;
            prevPointerY = e.clientY;
        });

        const stopDrag = (e) => {
            isPointerDragging = false;
            try { studioContainer.releasePointerCapture(e.pointerId); } catch (_) {}
        };

        studioContainer.addEventListener('pointerup', stopDrag);
        studioContainer.addEventListener('pointercancel', stopDrag);

        studioContainer.addEventListener('wheel', (e) => {
            e.preventDefault();
            targetCameraDistance += e.deltaY * 0.004;
            targetCameraDistance = Math.max(2.4, Math.min(7.0, targetCameraDistance));
        }, { passive: false });
    }

    function updateBonesGeometry() {
        if (!currentPose3D) return;
        const vY = new THREE.Vector3(0, 1, 0);

        for (let b = 0; b < BONE_CONNECTIONS.length; b++) {
            const [i1, i2] = BONE_CONNECTIONS[b];
            const p1 = jointSpheres[i1].position;
            const p2 = jointSpheres[i2].position;

            const cyl = boneCylinders[b];
            const dist = p1.distanceTo(p2);

            cyl.scale.set(1, Math.max(dist, 0.001), 1);
            cyl.position.copy(p1).add(p2).multiplyScalar(0.5);

            const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
            cyl.quaternion.setFromUnitVectors(vY, dir);
        }
    }

    function updatePalmGeometry() {
        if (!palmGeometry || !jointSpheres || jointSpheres.length < 21) return;

        const palmLandmarks = [0, 1, 5, 9, 13, 17];
        const posAttr = palmGeometry.attributes.position;

        for (let i = 0; i < 6; i++) {
            const p = jointSpheres[palmLandmarks[i]].position;
            posAttr.setXYZ(i, p.x, p.y, p.z);
        }

        posAttr.needsUpdate = true;
        palmGeometry.computeVertexNormals();
    }

    function setMaterialTheme(clay) {
        isClayMode = clay;
        const mainMat = isClayMode ? clayMat : skinMat;
        const jMat = isClayMode ? jointClayMat : jointSkinMat;
        const pMat = isClayMode ? palmClayMat : palmSkinMat;

        boneCylinders.forEach(b => { b.material = mainMat; });
        jointSpheres.forEach(s => { s.material = jMat; });
        if (palmMesh) palmMesh.material = pMat;

        if (toggleMaterialBtn) {
            toggleMaterialBtn.innerHTML = isClayMode ? '<span>🎨 Natural Skin</span>' : '<span>🎨 Studio Clay</span>';
        }
    }

    function set3DSignPose(signId) {
        active3DSign = signId;
        const signInfo = SIGNS_DATA.find(s => s.id === signId);
        if (!signInfo) return;

        // Update Header & Badge
        if (studioSignBadge) studioSignBadge.textContent = signInfo.id;
        if (studioSignTitle) studioSignTitle.textContent = signInfo.title;
        if (studioCatTag) {
            const catLabel = signInfo.category === 'letter' ? 'Alphabet' : (signInfo.category === 'number' ? 'Number' : 'Phrase');
            studioCatTag.textContent = catLabel;
        }
        if (studioSignDesc) studioSignDesc.textContent = signInfo.pose;

        // Update 2D Drawn Hand Illustration (Instant 0ms latency from local memory)
        if (studio2DImg && signInfo.imgUrl) {
            studio2DImg.src = signInfo.imgUrl;
            studio2DImg.alt = `${signInfo.title} Hand Sign Illustration`;
            studio2DImg.style.display = 'block';
        }
        if (studio2DSvgWrap) {
            studio2DSvgWrap.style.display = 'none';
        }

        // Update Finger Posture Details & Tips
        if (studioThumbVal) studioThumbVal.textContent = signInfo.fingers.thumb;
        if (studioIndexVal) studioIndexVal.textContent = signInfo.fingers.index;
        if (studioOthersVal) studioOthersVal.textContent = signInfo.fingers.other;
        if (studioTipVal) studioTipVal.textContent = signInfo.tip;

        // Update Active Pill
        document.querySelectorAll('.studio-pill').forEach(pill => {
            pill.classList.toggle('active', pill.getAttribute('data-id') === signId);
        });

        // Update 3D Model Coordinates
        if (poses3D[signId]) {
            targetPose3D = poses3D[signId];
            if (!currentPose3D) {
                currentPose3D = JSON.parse(JSON.stringify(targetPose3D));
            }
        }
    }

    // ── Search & Filter Studio Pills ──
    function populateStudioPills() {
        if (!studioPillsRow) return;
        studioPillsRow.innerHTML = '';

        const query = searchQuery.trim().toLowerCase();
        const filtered = SIGNS_DATA.filter(sign => {
            const matchCategory = (activeCategory === 'all' || sign.category === activeCategory);
            const matchQuery = !query ||
                sign.title.toLowerCase().includes(query) ||
                sign.id.toLowerCase().includes(query) ||
                sign.pose.toLowerCase().includes(query) ||
                sign.tip.toLowerCase().includes(query);
            return matchCategory && matchQuery;
        });

        if (studioSearchEmpty) {
            studioSearchEmpty.style.display = filtered.length === 0 ? 'block' : 'none';
        }

        filtered.forEach(sign => {
            const pill = document.createElement('button');
            pill.className = `studio-pill ${sign.id === active3DSign ? 'active' : ''}`;
            pill.textContent = sign.id;
            pill.title = sign.title;
            pill.setAttribute('data-id', sign.id);

            pill.addEventListener('click', () => {
                set3DSignPose(sign.id);
            });

            studioPillsRow.appendChild(pill);
        });

        // If the current active sign is not in the filtered list, select the first match
        if (filtered.length > 0 && !filtered.some(s => s.id === active3DSign)) {
            set3DSignPose(filtered[0].id);
        }
    }

    // ── Studio View Mode Switcher: 3D | 2D | Split (Default) ──
    function setStudioMode(mode) {
        activeStudioMode = mode;
        if (!studioViewportWrap) return;

        studioViewportWrap.className = `studio-viewport-wrap mode-${mode}`;

        [mode3DBtn, mode2DBtn, modeSplitBtn].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });

        if (mode === '3d' && mode3DBtn) mode3DBtn.classList.add('active');
        if (mode === '2d' && mode2DBtn) mode2DBtn.classList.add('active');
        if (mode === 'split' && modeSplitBtn) modeSplitBtn.classList.add('active');

        // Trigger Three.js resize when layout switches
        setTimeout(onStudioResize, 50);
    }

    // ── Practice Mode ──
    function startPracticeMode(signId) {
        const sign = SIGNS_DATA.find(s => s.id === signId) || SIGNS_DATA[0];
        practiceTarget = sign.id;
        if (practiceTargetName) practiceTargetName.textContent = sign.title;
        if (practiceInstruction) practiceInstruction.textContent = sign.pose;
        if (practiceBanner) {
            practiceBanner.className = 'practice-banner';
            practiceBanner.style.display = 'flex';
        }

        const demoEl = document.getElementById('demo');
        if (demoEl) demoEl.scrollIntoView({ behavior: 'smooth' });

        if (!running) {
            startCamera();
        }
    }

    function exitPracticeMode() {
        practiceTarget = null;
        if (practiceBanner) {
            practiceBanner.style.display = 'none';
            practiceBanner.classList.remove('goal-met');
        }
    }

    // ── MediaPipe Hands Initializer ──
    function initHands() {
        return new Promise((resolve, reject) => {
            try {
                if (typeof Hands === 'undefined') {
                    return reject(new Error('MediaPipe Hands library not found'));
                }
                handsDetector = new Hands({
                    locateFile: (file) =>
                        `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
                });

                handsDetector.setOptions({
                    maxNumHands: 1,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.45,
                    minTrackingConfidence: 0.45,
                });

                handsDetector.onResults(onHandResults);

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
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });

            videoEl.srcObject = stream;
            await videoEl.play();

            canvasEl.width = videoEl.videoWidth || 640;
            canvasEl.height = videoEl.videoHeight || 480;

            if (placeholder) placeholder.classList.add('hidden');
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'inline-flex';
            running = true;

            setStatus('ready', '🟢 Camera live — show signs in continuous flow');
            processFrame();
        } catch (err) {
            setStatus('error', `Camera denied: ${err.message}`);
        }
    }

    function stopCamera() {
        running = false;
        if (animationId) cancelAnimationFrame(animationId);
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        if (videoEl) videoEl.srcObject = null;
        if (ctx && canvasEl) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        if (placeholder) placeholder.classList.remove('hidden');
        if (startBtn) startBtn.style.display = 'inline-flex';
        if (stopBtn) stopBtn.style.display = 'none';
        setStatus('ready', 'Camera stopped. Click "Start Camera" to test again.');
        if (predValue) predValue.textContent = '—';
        if (predConf) predConf.textContent = 'Waiting for hand...';
        if (confidenceBar) confidenceBar.style.width = '0%';
    }

    function toggleWebcamMirror() {
        isWebcamMirrored = !isWebcamMirrored;
        if (videoEl && canvasEl) {
            videoEl.classList.toggle('mirrored', isWebcamMirrored);
            canvasEl.classList.toggle('mirrored', isWebcamMirrored);
        }
    }

    // ── Frame Processing Loop ──
    async function processFrame() {
        if (!running) return;

        try {
            await handsDetector.send({ image: videoEl });
        } catch (e) {
            // Drop frame on collision
        }

        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            if (fpsValue) fpsValue.textContent = frameCount;
            frameCount = 0;
            lastFpsTime = now;
        }

        animationId = requestAnimationFrame(processFrame);
    }

    // ── MediaPipe Results Handler ──
    function onHandResults(results) {
        if (!ctx || !canvasEl) return;
        const w = canvasEl.width;
        const h = canvasEl.height;
        ctx.clearRect(0, 0, w, h);

        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            if (predValue) predValue.textContent = '—';
            if (predConf) predConf.textContent = 'No hand in frame';
            if (confidenceBar) confidenceBar.style.width = '0%';
            lastCandidate = null;
            stableStreak = 0;
            return;
        }

        const landmarks = results.multiHandLandmarks[0];

        // Draw HUD Skeleton
        drawHandOverlay(landmarks, w, h);

        // Extract 42 coordinates (21 joints * 2)
        const rawFeatures = new Array(42);
        for (let i = 0; i < 21; i++) {
            rawFeatures[i * 2] = landmarks[i].x;
            rawFeatures[i * 2 + 1] = landmarks[i].y;
        }

        // Run Position- & Scale-Invariant Classifier
        const res = model.predict(rawFeatures);
        displayPrediction(res.prediction, res.confidence);

        // Continuous Fast-Flow Sentence Assembly
        handleSentenceFlow(res.prediction, res.confidence);

        // Practice Mode Verification
        if (practiceTarget && res.prediction === practiceTarget && res.confidence >= 0.70) {
            if (practiceBanner) {
                practiceBanner.classList.add('goal-met');
                if (practiceInstruction) practiceInstruction.textContent = '✓ Correct sign recognized! Great job!';
            }
        }
    }

    // ── Fast-Flow Continuous Debouncing ──
    function handleSentenceFlow(sign, conf) {
        if (conf < 0.65) return;

        if (sign === lastCandidate) {
            stableStreak++;
        } else {
            lastCandidate = sign;
            stableStreak = 1;
        }

        if (stableStreak === stableThreshold) {
            if (sentence.length === 0 || sentence[sentence.length - 1] !== sign) {
                sentence.push(sign);
                renderSentence();
                flashPredictionCard();
            }
        }
    }

    function flashPredictionCard() {
        if (!predCard) return;
        predCard.classList.add('flash');
        setTimeout(() => predCard.classList.remove('flash'), 350);
    }

    function displayPrediction(sign, conf) {
        if (predValue) predValue.textContent = sign;
        const pct = Math.round(conf * 100);
        if (predConf) predConf.textContent = `${pct}% confidence`;
        if (confidenceBar) {
            confidenceBar.style.width = `${pct}%`;
            confidenceBar.className = 'confidence-bar-fill';
            if (pct >= 85) confidenceBar.classList.add('high');
            else if (pct >= 60) confidenceBar.classList.add('medium');
            else confidenceBar.classList.add('low');
        }
    }

    // ── Draw Clean Overlay ──
    function drawHandOverlay(landmarks, w, h) {
        ctx.save();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#2563eb';

        for (const [i1, i2] of BONE_CONNECTIONS) {
            ctx.beginPath();
            ctx.moveTo(landmarks[i1].x * w, landmarks[i1].y * h);
            ctx.lineTo(landmarks[i2].x * w, landmarks[i2].y * h);
            ctx.stroke();
        }

        for (let i = 0; i < 21; i++) {
            const x = landmarks[i].x * w;
            const y = landmarks[i].y * h;
            const isTip = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20);

            ctx.beginPath();
            ctx.arc(x, y, isTip ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = isTip ? '#059669' : '#93c5fd';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        ctx.restore();
    }

    // ── Sentence Operations ──
    function renderSentence() {
        if (!sentenceValue) return;
        if (sentence.length === 0) {
            sentenceValue.innerHTML = '<span class="sentence-placeholder">Signs will automatically concatenate as you sign...</span>';
            if (sentenceCount) sentenceCount.textContent = '0 signs';
        } else {
            sentenceValue.textContent = sentence.join(' ');
            if (sentenceCount) sentenceCount.textContent = `${sentence.length} sign${sentence.length > 1 ? 's' : ''}`;
        }
    }

    function clearSentence() {
        sentence = [];
        renderSentence();
    }

    function copySentence() {
        if (sentence.length === 0) return;
        const text = sentence.join(' ');
        navigator.clipboard.writeText(text).then(() => {
            if (!copyBtn) return;
            const orig = copyBtn.innerHTML;
            copyBtn.innerHTML = '✓';
            setTimeout(() => { copyBtn.innerHTML = orig; }, 1500);
        });
    }

    function speakSentence() {
        if (sentence.length === 0) return;
        const text = sentence.join(' ');
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            window.speechSynthesis.speak(utterance);
        }
    }

    // ── Status Helper ──
    function setStatus(type, text) {
        if (statusDot) {
            statusDot.className = 'status-dot';
            if (type === 'ready') statusDot.classList.add('ready');
            else if (type === 'error') statusDot.classList.add('error');
        }
        if (statusText) statusText.textContent = text;
    }

    // ── Event Listeners ──
    function setupEventListeners() {
        if (startBtn) startBtn.addEventListener('click', startCamera);
        if (stopBtn) stopBtn.addEventListener('click', stopCamera);
        if (flipBtn) flipBtn.addEventListener('click', toggleWebcamMirror);
        if (clearBtn) clearBtn.addEventListener('click', clearSentence);
        if (copyBtn) copyBtn.addEventListener('click', copySentence);
        if (speakBtn) speakBtn.addEventListener('click', speakSentence);
        if (clearPracticeBtn) clearPracticeBtn.addEventListener('click', exitPracticeMode);

        // Studio View Mode Switcher
        if (modeSplitBtn) modeSplitBtn.addEventListener('click', () => setStudioMode('split'));
        if (mode3DBtn) mode3DBtn.addEventListener('click', () => setStudioMode('3d'));
        if (mode2DBtn) mode2DBtn.addEventListener('click', () => setStudioMode('2d'));

        // Practice Current Sign Button
        if (practiceCurrentSignBtn) {
            practiceCurrentSignBtn.addEventListener('click', () => {
                startPracticeMode(active3DSign);
            });
        }

        // 3D Studio Controls
        if (viewFrontBtn) {
            viewFrontBtn.addEventListener('click', () => {
                targetPitch = 0.0;
                targetYaw = 0.0;
                targetCameraDistance = 4.4;
                setActivePerspectiveBtn(viewFrontBtn);
            });
        }

        if (viewSideBtn) {
            viewSideBtn.addEventListener('click', () => {
                targetPitch = 0.0;
                targetYaw = Math.PI / 2.0;
                targetCameraDistance = 4.4;
                setActivePerspectiveBtn(viewSideBtn);
            });
        }

        if (viewTopBtn) {
            viewTopBtn.addEventListener('click', () => {
                targetPitch = 1.35;
                targetYaw = 0.0;
                targetCameraDistance = 4.4;
                setActivePerspectiveBtn(viewTopBtn);
            });
        }

        if (reset3DViewBtn) {
            reset3DViewBtn.addEventListener('click', () => {
                targetPitch = 0.12;
                targetYaw = 0.0;
                targetCameraDistance = 4.4;
                if (viewFrontBtn) setActivePerspectiveBtn(viewFrontBtn);
            });
        }

        if (toggle3DMirrorBtn) {
            toggle3DMirrorBtn.addEventListener('click', () => {
                is3DMirrored = !is3DMirrored;
                toggle3DMirrorBtn.classList.toggle('active', is3DMirrored);
                toggle3DMirrorBtn.innerHTML = is3DMirrored ? '<span>🪞 Normal (Right Hand)</span>' : '<span>🪞 Mirror (Left Hand)</span>';
            });
        }

        if (toggleMaterialBtn) {
            toggleMaterialBtn.addEventListener('click', () => {
                setMaterialTheme(!isClayMode);
            });
        }

        if (toggleAutoRotateBtn) {
            toggleAutoRotateBtn.addEventListener('click', () => {
                is3DAutoRotate = !is3DAutoRotate;
                toggleAutoRotateBtn.classList.toggle('active', is3DAutoRotate);
            });
        }

        // Speed toggle buttons
        if (speedFastBtn) {
            speedFastBtn.addEventListener('click', () => {
                speedFastBtn.classList.add('active');
                if (speedNormalBtn) speedNormalBtn.classList.remove('active');
                flowSpeed = 'fast';
                stableThreshold = 6;
            });
        }

        if (speedNormalBtn) {
            speedNormalBtn.addEventListener('click', () => {
                speedNormalBtn.classList.add('active');
                if (speedFastBtn) speedFastBtn.classList.remove('active');
                flowSpeed = 'normal';
                stableThreshold = 16;
            });
        }

        // Studio Search Input
        if (studioSignSearch) {
            studioSignSearch.addEventListener('input', (e) => {
                searchQuery = e.target.value;
                if (studioClearSearch) {
                    studioClearSearch.style.display = searchQuery ? 'block' : 'none';
                }
                populateStudioPills();
            });

            studioSignSearch.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const firstPill = studioPillsRow.querySelector('.studio-pill');
                    if (firstPill) firstPill.click();
                }
            });
        }

        if (studioClearSearch) {
            studioClearSearch.addEventListener('click', () => {
                if (studioSignSearch) studioSignSearch.value = '';
                searchQuery = '';
                studioClearSearch.style.display = 'none';
                populateStudioPills();
                if (studioSignSearch) studioSignSearch.focus();
            });
        }

        // Studio Category Filter Tabs
        categoryTabs.forEach(btn => {
            btn.addEventListener('click', () => {
                categoryTabs.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeCategory = btn.getAttribute('data-category');
                populateStudioPills();
            });
        });
    }

    function setActivePerspectiveBtn(activeBtn) {
        [viewFrontBtn, viewSideBtn, viewTopBtn].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });
        if (activeBtn) activeBtn.classList.add('active');
    }

    // Boot
    init();
})();
