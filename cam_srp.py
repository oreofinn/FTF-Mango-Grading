import time
import cv2
import joblib
import numpy as np
from picamera2 import Picamera2
from datetime import datetime
import os
import sys
import threading
from collections import Counter, defaultdict

# Flask imports
from flask import Flask, request, redirect, jsonify

# TensorFlow imports
import tensorflow as tf

# Fix numpy import on Raspberry Pi
import numpy.core
sys.modules['numpy._core'] = numpy.core

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATASET_ROOT     = 'mango_dataset'
MODEL_TMPL       = 'cnn_{}.keras'   # e.g. cnn_apple_mango.keras
GRADES           = ['Grade A', 'Grade B', 'Grade C', 'Rejected']

# ---------------------------------------------------------------------
# SRP Recommendations (PHP per kg)
# ---------------------------------------------------------------------
SRP_PER_KG = {
    'Grade A': 120.0,
    'Grade B':  90.0,
    'Grade C':  60.0,
    'Rejected':   0.0,
}

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
app = Flask(__name__)
grade_counts    = defaultdict(int)
last_grade      = None
last_defect     = None
current_variety = None
current_model   = None
graded_once     = False

# ---------------------------------------------------------------------
# Flask Endpoints
# ---------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    varieties = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])
    btns = ''
    for v in varieties:
        border = '3px solid #000' if v == current_variety else '1px solid #888'
        btns += (
            f"<form style='display:inline' method='POST' action='/select'>"
            f"<input type='hidden' name='var' value='{v}'/>"
            f"<button style='margin:5px;padding:8px;border:{border};'>{v}</button>"
            "</form>"
        )

    color_map = {
        'Grade A':'#3B7A57','Grade B':'#0066cc',
        'Grade C':'#ffcc00','Rejected':'#d9534f'
    }
    cards = ''
    for g in GRADES:
        short = 'R' if g=='Rejected' else g.split()[-1]
        cards += (
            f"<div style='display:inline-block;width:160px;height:100px;"
            f"background:{color_map[g]};margin:8px;border-radius:8px;"
            f"color:black;font-size:20px;font-weight:bold;"
            f"text-align:center;line-height:100px;'>"
            f"{g}<br><span id='count-{short}'>{grade_counts[g]}</span>"
            "</div>"
        )

    last_html = (
        f"Last Mango � Grade: <strong><span id='last-grade'>"
        f"{last_grade or 'None'}</span></strong>, Defect: <strong>"
        f"<span id='last-defect'>{(last_defect or 0.0):.1f}%</span>"
        f"</strong><br/>SRP Recommendation: <strong>"
        f"<span id='srp'>{SRP_PER_KG.get(last_grade, 0.0):.1f}"
        f"</span></strong> PHP/kg"
    )

    return f"""
<html>
<head><title>Mango Grading</title></head>
<body>
  <h1>Select variety:</h1>
  {btns}
  <hr/>
  <h2>Current: <em>{current_variety or 'none'}</em></h2>
  <div>{cards}</div>
  <div style='clear:both;margin-top:60px;font-size:18px;' id='last-info'>
    {last_html}
  </div>
  <script>
    async function fetchStatus() {{
      try {{
        let resp = await fetch('/status');
        let js   = await resp.json();
        document.getElementById('count-A').innerText = js.grade_counts['Grade A'];
        document.getElementById('count-B').innerText = js.grade_counts['Grade B'];
        document.getElementById('count-C').innerText = js.grade_counts['Grade C'];
        document.getElementById('count-R').innerText = js.grade_counts['Rejected'];
        document.getElementById('last-grade').innerText  = js.last_grade || 'None';
        document.getElementById('last-defect').innerText = js.last_defect.toFixed(1) + '%';
        document.getElementById('srp').innerText         = js.srp.toFixed(1);
      }} catch(e) {{ console.warn('Status fetch failed:', e); }}
    }}
    setInterval(fetchStatus, 500);
  </script>
</body>
</html>
"""

@app.route('/select', methods=['POST'])
def select_variety():
    global current_variety, current_model, grade_counts
    v = request.form.get('var')
    folder = os.path.join(DATASET_ROOT, v)
    if not os.path.isdir(folder):
        return 'Unknown variety', 400

    safe = v.lower().replace(' ', '_')
    model_file = os.path.join(folder, MODEL_TMPL.format(safe))
    if not os.path.exists(model_file):
        return f'Model not found: {model_file}', 404

    try:
        current_model = tf.keras.models.load_model(model_file)
        current_model.class_names_ = GRADES
    except Exception as e:
        return f'Failed to load model: {e}', 500

    current_variety = v
    grade_counts    = defaultdict(int)
    return redirect('/')

@app.route('/status', methods=['GET'])
def status():
    srp = SRP_PER_KG.get(last_grade, 0.0)
    return jsonify({
        'grade_counts': grade_counts,
        'last_grade':   last_grade,
        'last_defect':  last_defect or 0.0,
        'srp':          srp,
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

# ---------------------------------------------------------------------
# Camera Setup
# ---------------------------------------------------------------------
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(
    main={'size': (1280, 720), 'format': 'YUV420'}
)
picam2.configure(cfg)
picam2.start()
time.sleep(2)

cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Preview', 1280, 720)
cv2.moveWindow('Camera Preview', 50, 50)

cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mask', 640, 360)
cv2.moveWindow('Mask', 1400, 50)

stable_threshold = 3
last_pred        = None
stable_cnt       = 0

VARIETY_COLOR_RANGES = {
    'APPLE MANGO':   (np.array([20,100,100]), np.array([40,255,255])),
    'CARABAO MANGO': (np.array([20,100,100]), np.array([40,255,255])),
    'INDIAN MANGO':  (np.array([30,50,50]),    np.array([85,255,255])),
    'PICO MANGO':    (np.array([20,100,100]), np.array([40,255,255])),
}

# ---------------------------------------------------------------------
# Detection Function
# ---------------------------------------------------------------------
def detect_mango(image):
    # 1) pick HSV range
    lower, upper = VARIETY_COLOR_RANGES.get(
        current_variety,
        VARIETY_COLOR_RANGES['APPLE MANGO']
    )
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2) mask + closing to fill any holes in the fruit silhouette
    raw_mask = cv2.inRange(hsv, lower, upper)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask     = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    # 3) find the mango contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, mask, 0, 0, 0, 0
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 2000:   # raised from 500 to 2000 for stability
        return False, mask, 0, 0, 0, 0

    # 4) ROI mask & fruit area
    roi = np.zeros_like(mask)
    cv2.drawContours(roi, [cnt], -1, 255, -1)
    area = np.count_nonzero(roi)

    # 5) color-based defect %
    defect_mask = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    defect_pct  = np.count_nonzero(defect_mask) / area * 100

    # 6) black/brown spot %
    bad_mask  = cv2.inRange(hsv, np.array([0,0,0]), np.array([50,255,120]))
    bad_roi   = cv2.bitwise_and(roi, bad_mask)
    bad_close = cv2.morphologyEx(bad_roi, cv2.MORPH_CLOSE, kernel)
    spot_pct  = np.count_nonzero(bad_close) / area * 100

    # 7) wrinkle detection - blur first to reduce LED glare noise
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9,9), 0)
    lap       = cv2.Laplacian(gray_blur, cv2.CV_64F)
    lap_roi   = lap * (roi/255)
    wrinkle_var = lap_roi.var()

    # 8) count large defect blobs (optional, you can drop if unused)
    dcnts, _    = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_defs  = sum(1 for c in dcnts if cv2.contourArea(c) > 300)

    return True, mask, defect_pct, large_defs, spot_pct, wrinkle_var

# ---------------------------------------------------------------------
# Camera & Grading Loop
# ---------------------------------------------------------------------
def camera_loop():
    global last_pred, stable_cnt, last_grade, last_defect, graded_once

    while True:
        votes, last_frame = [], None
        det_count = 0

        # 1) collect a few frames for voting
        for _ in range(5):
            yuv   = picam2.capture_array()
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            # rotate frame upside-down
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            last_frame = frame

            ok, *_ = detect_mango(frame)
            if ok and current_model:
                det_count += 1
                crop = cv2.resize(frame, (100, 100))[None] / 255.0
                idx  = current_model.predict(crop, verbose=0).argmax()
                votes.append(current_model.class_names_[idx])
            time.sleep(0.2)

        # 2) majority vote
        pred = Counter(votes).most_common(1)[0][0] if det_count >= 3 and votes else None

        # 3) full detection on the last rotated frame
        ok, mask, defect_pct, bigs, spot_pct, wrinkle_var = detect_mango(last_frame)

        print(f"DEF={defect_pct:.1f}%, SPOT={spot_pct:.1f}%, WRINK={wrinkle_var:.0f}")

        if ok and pred and not graded_once:
        # 1) REJECT on massive black/brown spots
            if spot_pct > 10.0:
                grade = 'Rejected'

            # 2) REJECT on very severe wrinkles
            elif wrinkle_var > 1600:
                grade = 'Rejected'

            # 3) GRADE C on moderate wrinkles (even if spot_pct is low)
            elif wrinkle_var > 800:
                grade = 'Grade C'

            # 4) GRADE C also on medium spots
            elif spot_pct > 5.0:
                grade = 'Grade C'

            # 5) GRADE B on mild wrinkles or small hue defects
            elif wrinkle_var > 300 or defect_pct > 4.0 or spot_pct > 2.0:
                grade = 'Grade B'

            # 6) otherwise it�s Grade A
            else:
                grade = 'Grade A'

            if grade == last_pred:
                stable_cnt += 1
            else:
                stable_cnt = 1
            last_pred = grade

            if stable_cnt >= stable_threshold:
                grade_counts[grade] += 1

                # annotate on screen
                h, w = last_frame.shape[:2]
                cv2.rectangle(last_frame, (0, 0), (w-1, h-1), (0, 255, 0), 3)

                cv2.putText(
                    last_frame, grade, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2
                )
                cv2.putText(
                    last_frame, f"{defect_pct:.1f}% defect", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2
                )

                last_grade  = grade
                last_defect = defect_pct
                graded_once = True
                stable_cnt  = 0

        # display rotated preview & mask
        cv2.imshow('Camera Preview', last_frame)
        cv2.imshow('Mask', mask)

        # reset when no mango
        if not ok:
            graded_once = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    camera_loop()
