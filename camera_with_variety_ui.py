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
from flask import Flask, request, redirect

# TensorFlow for .keras models
import tensorflow as tf

# fix for numpy import on rpi
import numpy.core
sys.modules["numpy._core"] = numpy.core

# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------
DATASET_ROOT   = "mango_dataset"
MODEL_TMPL     = "cnn_{}.keras"   # e.g. cnn_apple_mango.keras
GRADES         = ["Grade A", "Grade B", "Grade C", "Rejected"]

# ---------------------------------------------------------------------
#  Globals
# ---------------------------------------------------------------------
app = Flask(__name__)
grade_counts    = defaultdict(int)
last_grade: str = None
last_defect: float = None
current_variety = None
current_model   = None
graded_once     = False

# ---------------------------------------------------------------------
#  FLASK UI
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # list all variety folders
    varieties = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])

    # build the variety-select buttons
    btns = ""
    for v in varieties:
        border = "3px solid #000" if v == current_variety else "1px solid #888"
        btns += (
            f'<form style="display:inline" method="POST" action="/select">'
            f'<input type="hidden" name="var" value="{v}"/>'
            f'<button style="margin:5px;padding:8px;border:{border};">'
            f'{v}</button></form>'
        )

    # build the grade cards
    cards = ""
    color_map = {
        "Grade A":"#3B7A57", "Grade B":"#0066cc",
        "Grade C":"#ffcc00", "Rejected":"#d9534f"
    }
    for g in GRADES:
        cards += (
            f'<div style="display:inline-block;width:160px;height:100px;'
            f'background:{color_map[g]};margin:8px;border-radius:8px;'
            f'color:black;font-size:20px;font-weight:bold;'
            f'text-align:center;line-height:100px;">'
            f'{g}<br>{grade_counts[g]}</div>'
        )

    # build the �Last Mango� line
    if last_grade is not None:
        last_info = (
            f'<div style="clear:both;margin-top:50px;font-size:18px;">'
            f'Last Mango � Grade: <strong>{last_grade}</strong>, '
            f'Defect: <strong>{last_defect:.1f}%</strong>'
            f'</div>'
        )
    else:
        last_info = (
            '<div style="clear:both;margin-top:30px;font-size:18px;">'
            'Last Mango � <em>None yet</em>'
            '</div>'
        )

    return f"""
    <html><head><title>Mango Grading</title></head><body>
      <h1>Select variety:</h1>
      {btns}
      <hr/>
      <h2>Current: <em>{current_variety or "none"}</em></h2>
      <div>{cards}</div>

      {last_info}

    </body></html>
    """

@app.route("/select", methods=["POST"])
def select_variety():
    global current_variety, current_model, grade_counts

    v = request.form.get("var")
    folder = os.path.join(DATASET_ROOT, v)
    if not os.path.isdir(folder):
        return "? Unknown variety", 400

    # load model
    safe = v.lower().replace(" ", "_")
    model_file = os.path.join(folder, MODEL_TMPL.format(safe))
    if not os.path.exists(model_file):
        return f"? Model not found: {model_file}", 404

    try:
        current_model = tf.keras.models.load_model(model_file)
        current_model.class_names_ = GRADES
    except Exception as e:
        return f"? Failed to load model: {e}", 500

    current_variety = v
    grade_counts    = defaultdict(int)
    return redirect("/")


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)

# ---------------------------------------------------------------------
#  CAMERA + PREDICTION
# ---------------------------------------------------------------------
# configure PiCamera2
picam2 = Picamera2()
cfg    = picam2.create_preview_configuration(main={"size":(640,480),"format":"YUV420"})
picam2.configure(cfg)
picam2.start()
time.sleep(2)

cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera Preview", 50, 50)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.moveWindow("Mask", 700, 50)

stable_threshold = 3
last_pred        = None
stable_cnt       = 0

# at module top:
VARIETY_COLOR_RANGES = {
    "APPLE MANGO":   (np.array([20,100,100]), np.array([40,255,255])),
    "CARABAO MANGO": (np.array([20,100,100]), np.array([40,255,255])),
    "INDIAN MANGO":  (np.array([30, 50, 50]),   np.array([85,255,255])),
    "PICO MANGO":    (np.array([20,100,100]),  np.array([40,255,255])),
}

# ---------------------------------------------------------------------
#  DETECTION FUNCTION
# ---------------------------------------------------------------------
def detect_mango(image):
    # pick colour range for current variety
    lower, upper = VARIETY_COLOR_RANGES.get(
        current_variety,
        VARIETY_COLOR_RANGES["APPLE MANGO"]
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, mask, 0.0, 0, 0.0, 0.0

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 500:
        return False, mask, 0.0, 0, 0.0, 0.0

    # build ROI mask
    roi_mask = np.zeros_like(mask)
    cv2.drawContours(roi_mask, [cnt], -1, 255, -1)

    # color-based defect
    defect_mask   = cv2.bitwise_and(roi_mask, cv2.bitwise_not(mask))
    defect_area   = np.count_nonzero(defect_mask)
    mango_area    = np.count_nonzero(roi_mask)
    defect_pct    = (defect_area / mango_area) * 100
    dcontours, _  = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_defects = sum(1 for c in dcontours if cv2.contourArea(c) > 300)

    # black-spot detection
    black_low     = np.array([0, 0, 0])
    black_high    = np.array([180, 255, 50])
    black_mask    = cv2.inRange(hsv, black_low, black_high)
    black_spot    = cv2.bitwise_and(roi_mask, black_mask)
    black_pct     = (np.count_nonzero(black_spot) / mango_area) * 100

    # wrinkle detection via Laplacian variance
    gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap         = cv2.Laplacian(gray, cv2.CV_64F)
    wrinkle_var = lap.var()

    return True, mask, defect_pct, large_defects, black_pct, wrinkle_var

# ---------------------------------------------------------------------
#  TRAINING BLOCK (optional: can be separated into its own script)
# ---------------------------------------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

# Config for training
IMG_SIZE = (100, 100)
BATCH    = 32
EPOCHS   = 10
augmenter = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)

for variety in os.listdir(DATASET_ROOT):
    path = os.path.join(DATASET_ROOT, variety)
    if not os.path.isdir(path):
        continue

    print(f"\n?? Training variety: {variety}")
    train_gen = augmenter.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="training"
    )
    val_gen = augmenter.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = train_gen.num_classes
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, (3,3), activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128,(3,3), activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    safe   = variety.lower().replace(" ", "_")
    outname = f"cnn_{safe}.keras"
    model.save(outname)
    print(f"? Saved model for {variety}: {outname}")

# ---------------------------------------------------------------------
#  CAMERA + GRADING LOOP
# ---------------------------------------------------------------------
def camera_loop():
    global last_pred, stable_cnt, last_grade, last_defect, graded_once

    while True:
        votes, last_frame = [], None
        det_count = 0
        for _ in range(5):
            yuv   = picam2.capture_array()
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            last_frame = frame
            ok, *_ = detect_mango(frame)  # ignore other outputs
            if ok and current_model:
                det_count += 1
                crop = cv2.resize(frame, IMG_SIZE)[None]/255.0
                idx  = current_model.predict(crop, verbose=0).argmax()
                votes.append(current_model.class_names_[idx])
            time.sleep(0.2)

        if det_count >= 3 and votes:
            pred = Counter(votes).most_common(1)[0][0]
        else:
            pred = None

        ok, mask, pct, bigs, black_pct, wrinkle_var = detect_mango(last_frame)
        cv2.imshow("Camera Preview", last_frame)
        cv2.imshow("Mask", mask)

        if not ok:
            graded_once = False

        if ok and pred and not graded_once:
            # override for defects or black spots
            if (pct > 20 and bigs >= 3) or black_pct > 2.0:
                pred = "Rejected"
            # override for wrinkles
            if wrinkle_var > 1500:
                pred = "Rejected"

            # stability check
            if pred == last_pred:
                stable_cnt += 1
            else:
                stable_cnt = 1
            last_pred = pred

            if stable_cnt >= stable_threshold:
                grade_counts[pred] += 1
                cv2.rectangle(last_frame, (0,0), (639,479), (0,255,0), 3)
                cv2.putText(last_frame, pred, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.putText(last_frame, f"{pct:.1f}% defect", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                fn = f"captured_{pred}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(fn, last_frame)

                last_grade  = pred
                last_defect = pct
                graded_once  = True
                stable_cnt   = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    camera_loop()
