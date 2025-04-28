import time
import cv2
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
DATASET_ROOT   = 'mango_dataset'
MODEL_TMPL     = 'cnn_{}.keras'       # e.g. cnn_apple_mango.keras
GRADES         = ['Grade A','Grade B','Grade C','Rejected']
SRP_PER_KG     = {'Grade A':120.0,'Grade B':90.0,'Grade C':60.0,'Rejected':0.0}

# Black/brown spot thresholds (% of area) for default (yellow) mangoes
SPOT_THRESH_REJECT   = 5.0    # spot >5% -> Rejected
SPOT_THRESH_C        = 2.0    # 2ï¿½5% -> Grade C
SPOT_THRESH_B        = 0.5    # 0.5ï¿½2% -> Grade B

# Wrinkle variance thresholds for default mangoes
WRINKLE_THRESH_REJECT = 1200.0  # variance >1200 -> Rejected
WRINKLE_THRESH_C      = 700.0   # >700 -> Grade C
WRINKLE_THRESH_B      = 300.0   # >300 -> Grade B

# Hue-defect fallback (% of area)
DEFECT_THRESH_B       = 3.0     # >3% -> Grade B

# Indian-mango lenient ï¿½natural spotï¿½ allowances
SPOT_IND_REJECT      = 8.0    # spot >8% -> Rejected
SPOT_IND_C           = 4.0    # 4ï¿½8%  -> Grade C
SPOT_IND_B           = 1.0    # 1ï¿½4%  -> Grade B

WRINKLE_IND_REJECT   = 1500.0
WRINKLE_IND_C        = 900.0
WRINKLE_IND_B        = 400.0

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
app             = Flask(__name__)
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
    varieties = sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT,d))
    )
    btns = ''
    for v in varieties:
        border = '3px solid #000' if v==current_variety else '1px solid #888'
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
        f"Last Mango ï¿½ Grade: <strong><span id='last-grade'>{last_grade or 'None'}</span></strong>, "
        f"Defect: <strong><span id='last-defect'>{(last_defect or 0.0):.1f}%</span></strong><br/>"
        f"SRP: <strong><span id='srp'>{SRP_PER_KG.get(last_grade,0.0):.1f}</span></strong> PHP/kg"
    )
    return f"""
<html><head><title>Mango Grading</title></head><body>
  <h1>Select variety:</h1>
  {btns}
  <hr/>
  <h2>Current: <em>{current_variety or 'none'}</em></h2>
  <div>{cards}</div>
  <div style='clear:both;margin-top:60px;font-size:18px;' id='last-info'>{last_html}</div>
  <script>
    async function fetchStatus() {{
      let resp = await fetch('/status');
      let js   = await resp.json();
      document.getElementById('count-A').innerText = js.grade_counts['Grade A'];
      document.getElementById('count-B').innerText = js.grade_counts['Grade B'];
      document.getElementById('count-C').innerText = js.grade_counts['Grade C'];
      document.getElementById('count-R').innerText = js.grade_counts['Rejected'];
      document.getElementById('last-grade').innerText  = js.last_grade || 'None';
      document.getElementById('last-defect').innerText = js.last_defect.toFixed(1)+'%';
      document.getElementById('srp').innerText         = js.srp.toFixed(1);
    }}
    setInterval(fetchStatus,500);
  </script>
</body></html>
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
    grade_counts = defaultdict(int)
    return redirect('/')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'grade_counts': grade_counts,
        'last_grade':   last_grade,
        'last_defect':  last_defect or 0.0,
        'srp':          SRP_PER_KG.get(last_grade, 0.0),
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
    'APPLE MANGO':   (np.array([22,120,120]), np.array([35,255,255])),
    'CARABAO MANGO': (np.array([18,100,100]), np.array([30,255,255])),
    'INDIAN MANGO':  (np.array([35, 80,  80]), np.array([85,255,255])),
    'PICO MANGO':    (np.array([20,120,120]), np.array([38,255,255])),
}

# ---------------------------------------------------------------------
# Detection Function
# ---------------------------------------------------------------------
def detect_mango(image):
    lower, upper = VARIETY_COLOR_RANGES.get(
        current_variety,
        VARIETY_COLOR_RANGES['APPLE MANGO']
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, mask, 0, 0, 0, 0
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return False, mask, 0, 0, 0, 0

    roi = np.zeros_like(mask)
    cv2.drawContours(roi, [cnt], -1, 255, -1)
    area = np.count_nonzero(roi)

    # color-defects %
    defect_mask = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    defect_pct = np.count_nonzero(defect_mask) / area * 100

    # black/brown spots %
    bad_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([50,255,120]))
    bad_roi = cv2.bitwise_and(roi, bad_mask)
    bad_close = cv2.morphologyEx(bad_roi, cv2.MORPH_CLOSE, kernel)
    spot_pct = np.count_nonzero(bad_close) / area * 100

    # wrinkle variance
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9,9), 0)
    lap_roi = cv2.Laplacian(gray, cv2.CV_64F) * (roi/255)
    wrinkle_var = lap_roi.var()

    # count large defect blobs
    dcnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_defs = sum(1 for c in dcnts if cv2.contourArea(c) > 300)

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

        print(f"RAW ? spot={spot_pct:.1f}%, wrinkle={wrinkle_var:.0f}, defect={defect_pct:.1f}%")

        # 4) annotate and count
        if ok and pred and not graded_once:

            if current_variety == 'INDIAN MANGO':
                # --- Indian-mango thresholds ---
                # Black/brown spots
                if spot_pct   > SPOT_IND_REJECT:
                    grade = 'Rejected'
                elif spot_pct > SPOT_IND_C:
                    grade = 'Grade C'
                elif spot_pct > SPOT_IND_B:
                    grade = 'Grade B'

                # Wrinkles
                elif wrinkle_var > WRINKLE_IND_REJECT:
                    grade = 'Rejected'
                elif wrinkle_var > WRINKLE_IND_C:
                    grade = 'Grade C'
                elif wrinkle_var > WRINKLE_IND_B:
                    grade = 'Grade B'

                # Hue-defect fallback
                elif defect_pct > DEFECT_THRESH_B:
                    grade = 'Grade B'

                # Otherwise perfectly smooth
                else:
                    grade = 'Grade A'
            
            else:
                # --- Default (yellow mango) thresholds ---
                # Black/brown spots
                if spot_pct   > SPOT_THRESH_REJECT:
                    grade = 'Rejected'
                elif spot_pct > SPOT_THRESH_C:
                    grade = 'Grade C'
                elif spot_pct > SPOT_THRESH_B:
                    grade = 'Grade B'

                # Wrinkles
                elif wrinkle_var > WRINKLE_THRESH_REJECT:
                    grade = 'Rejected'
                elif wrinkle_var > WRINKLE_THRESH_C:
                    grade = 'Grade C'
                elif wrinkle_var > WRINKLE_THRESH_B:
                    grade = 'Grade B'

                # Hue-defect fallback
                elif defect_pct > DEFECT_THRESH_B:
                    grade = 'Grade B'

                # Otherwise perfectly smooth
                else:
                    grade = 'Grade A'

            # stability check & commit 
            if grade == last_pred:
                stable_cnt += 1
            else:
                stable_cnt = 1
            last_pred = grade

            if stable_cnt >= stable_threshold:
                grade_counts[grade] += 1
                cv2.rectangle(last_frame, (0,0), (639,479), (0,255,0), 3)
                cv2.putText(last_frame, grade, (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.putText(last_frame, f"{defect_pct:.1f}% defect", (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                last_grade   = grade
                last_defect  = defect_pct
                graded_once  = True
                stable_cnt   = 0

        # 5) display & reset
        cv2.imshow('Camera Preview', last_frame)
        cv2.imshow('Mask', mask)
        if not ok:
            graded_once = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    camera_loop()
