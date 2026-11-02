import pickle
import json
import numpy as np

d = pickle.load(open('c:/Users/Pranav Gour/AI based sign intrepreter 4/data.pickle', 'rb'))
labels = d['labels']
data = [x[:42] for x in d['data']]

poses = {}
for label in set(labels):
    samples = [data[i] for i, l in enumerate(labels) if l == label]
    avg = np.mean(samples, axis=0) # 42 floats
    
    # Extract 21 points
    xs = avg[0::2]
    ys = avg[1::2]
    
    # Center relative to wrist (index 0)
    wrist_x = xs[0]
    wrist_y = ys[0]
    
    cx = xs - wrist_x
    cy = ys - wrist_y
    
    # In screen coordinates, y points down. In 3D, y points up, so invert y
    cy = -cy
    
    # Normalize scale
    scale = max(np.max(np.abs(cx)), np.max(np.abs(cy)), 1e-4)
    cx = (cx / scale) * 2.0
    cy = (cy / scale) * 2.0
    
    # Calculate approximate anatomical depth z based on finger bending
    # Metacarpals (1,5,9,13,17) stay close to plane z=0
    # Tips when curled have negative z (curling towards palm/viewer)
    cz = np.zeros(21)
    
    # Thumb depth
    cz[1] = 0.1
    cz[2] = 0.2
    cz[3] = 0.15
    cz[4] = 0.05
    
    # Fingers 5-8 (index), 9-12 (middle), 13-16 (ring), 17-20 (pinky)
    # If the distance from wrist to tip is short compared to open hand, finger is curled forward
    for base in [5, 9, 13, 17]:
        tip = base + 3
        # curl depth estimation based on vertical extension
        extension = cy[tip] - cy[base]
        if extension < 0.5: # curled finger
            cz[base+1] = 0.25
            cz[base+2] = 0.4
            cz[tip] = 0.35
        else: # extended finger
            cz[base+1] = 0.05
            cz[base+2] = 0.08
            cz[tip] = 0.1
            
    # Combine into 21 x 3 array
    points_3d = []
    for i in range(21):
        points_3d.append([float(round(cx[i], 3)), float(round(cy[i], 3)), float(round(cz[i], 3))])
        
    poses[label] = points_3d

output_path = 'c:/Users/Pranav Gour/AI based sign intrepreter 4/web/poses_3d.json'
with open(output_path, 'w') as f:
    json.dump(poses, f)

print(f"Exported 3D coordinates for {len(poses)} signs to {output_path}")

