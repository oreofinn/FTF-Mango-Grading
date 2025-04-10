import os
import shutil

# 1) point this at your main �Grade A� folder:
root = r"/home/capstone/Desktop/mango_grading/mango_dataset/CARABAO_MANGO/Grade C"

# 2) make a temporary list of all subfolders (ANGLE 1, ANGLE 2, �)
for sub in os.listdir(root):
    subdir = os.path.join(root, sub)
    if not os.path.isdir(subdir):
        continue

    # 3) process each image file in that sub-folder
    for fname in os.listdir(subdir):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue

        src = os.path.join(subdir, fname)

        # 4) build a new filename that includes the angle
        #    e.g. "Grade A (ANGLE 1)_IMG_0001.JPG"
        angle = sub.replace(' ', '')            # collapse spaces if you like
        base, ext = os.path.splitext(fname)
        new_fname = f"Grade C ({angle})_{base}{ext}"

        dst = os.path.join(root, new_fname)

        # 5) move it into the main folder (avoid overwrites)
        if os.path.exists(dst):
            print(f"??  SKIP (already exists): {new_fname}")
        else:
            shutil.move(src, dst)
            print(f"?  Moved & renamed: {src!r} ? {new_fname!r}")

# (Optionally) clean up empty sub-folders:
for sub in os.listdir(root):
    subdir = os.path.join(root, sub)
    if os.path.isdir(subdir) and not os.listdir(subdir):
        os.rmdir(subdir)
        print(f"???  Removed empty folder: {sub!r}")
