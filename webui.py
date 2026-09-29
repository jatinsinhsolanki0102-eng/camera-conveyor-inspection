"""Web UI for the Camera-Based Conveyor Inspection & Tracking System.

Run from the project root:

    python webui.py            # opens http://127.0.0.1:5000

The page accepts:
  - a conveyor video (required),
  - checkerboard photos for Module B calibration (optional),
  - per-class box crops for Module E recognition (optional),
and runs either the full pipeline (frame stream -> A -> D -> B -> C -> E)
or every module script (A..E + pipeline) as a complete demo run, showing
per-module logs, the inspection report and all fresh figures for THAT video.

Each run starts from a clean outputs/ folder so results always belong to
the video you just uploaded (previous runs are archived in outputs_archive/).
"""
import time
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / 'data'
VIDEO_DIR = DATA / 'video'
CALIB_DIR = DATA / 'calibration'
BOXES_DIR = DATA / 'boxes'
UPLOAD_DIR = PROJECT_ROOT / 'uploads'
OUTPUTS = PROJECT_ROOT / 'outputs'
ARCHIVE = PROJECT_ROOT / 'outputs_archive'
WEBUI = PROJECT_ROOT / 'webui'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB uploads

ALLOWED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv'}
ALLOWED_IMG = {'.png', '.jpg', '.jpeg'}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def safe_name(name):
    name = Path(name).name
    return ''.join(c for c in name if c.isalnum() or c in '._- ').replace(' ', '_')


def save_file(file_storage, dest_dir, prefix=''):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f'{prefix}{safe_name(file_storage.filename)}'
    file_storage.save(str(path))
    return path


def save_images(files, dest_dir, clear=False):
    if clear and dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files or []:
        if f and f.filename:
            if Path(f.filename).suffix.lower() in ALLOWED_IMG:
                save_file(f, dest_dir)
                n += 1
    return n


def archive_outputs():
    """Move current outputs/ aside so the run starts clean."""
    if OUTPUTS.exists():
        ts = time.strftime('%Y%m%d_%H%M%S')
        dest = ARCHIVE / ts
        dest.mkdir(parents=True, exist_ok=True)
        for item in OUTPUTS.iterdir():
            shutil.move(str(item), str(dest / item.name))


def make_working_video(src, max_width, rid):
    work = UPLOAD_DIR / f'work_{rid}.mp4'
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_width and w > max_width:
        scale = max_width / w
        nw, nh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    else:
        nw, nh = w, h
    out = cv2.VideoWriter(str(work), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (nw, nh))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.write(cv2.resize(fr, (nw, nh)))
        n += 1
    out.release()
    cap.release()
    if n == 0:
        try:
            work.unlink()
        except OSError:
            pass
        return None
    return work, n, nw, nh


def run(cmd, timeout=3600):
    """Run a subprocess from the project root; returns (rc, output)."""
    try:
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                           text=True, encoding='utf-8', errors='replace',
                           timeout=timeout)
        return p.returncode, p.stdout + '\n' + p.stderr
    except subprocess.TimeoutExpired:
        return 124, f'TIMEOUT after {timeout} s\n'


def collect_outputs():
    items = []
    if OUTPUTS.exists():
        for p in sorted(OUTPUTS.rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg',
                                                    '.mp4'):
                items.append({
                    'url': '/outputs/' + p.relative_to(OUTPUTS).as_posix(),
                    'folder': str(p.parent.relative_to(OUTPUTS)),
                    'name': p.name,
                })
    return items


def read_report():
    rep = OUTPUTS / 'inspection_results.txt'
    if rep.exists():
        return rep.read_text(encoding='utf-8', errors='replace')
    return None


def save_box_crops():
    saved = {}
    for letter in ('A', 'B', 'C'):
        files = request.files.getlist(f'crops_{letter}')
        if files and any(f.filename for f in files):
            d = BOXES_DIR / f'box_type_{letter}'
            saved[letter] = save_images(files, d, clear=True)
    return saved


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.route('/')
def index():
    return send_from_directory(str(WEBUI), 'index.html')


@app.route('/outputs/<path:sub>')
def serve_output(sub):
    return send_from_directory(str(OUTPUTS), sub)


@app.route('/api/run', methods=['POST'])
def api_run():
    rid = uuid.uuid4().hex[:8]
    logs = {}
    t0 = time.time()

    # -- 1. parameters ------------------------------------------------------- #
    def _int(name, default):
        try:
            return int(request.form.get(name, default))
        except (TypeError, ValueError):
            return default

    def _float(name, default):
        try:
            return float(request.form.get(name, default))
        except (TypeError, ValueError):
            return default

    frames = _int('frames', 120)
    start = _int('start_frame', 0)
    max_width = _int('max_width', 960)
    square_cm = _float('square_cm', 2.5)
    mode = request.form.get('mode', 'pipeline')

    video = request.files.get('video')
    if video is None or not video.filename:
        return jsonify({'rid': 'none', 'ok': False,
                        'error': 'No video uploaded. Provide a conveyor video file.'}), 400
    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO:
        return jsonify({'rid': rid, 'ok': False,
                        'error': f'Unsupported video type "{ext}" (use mp4/avi/mov/mkv).'}), 400

    # -- 2. hold + downscale video --------------------------------------------
    raw = save_file(video, UPLOAD_DIR, prefix=f'{rid}_')
    work_info = make_working_video(raw, max_width, rid)
    if work_info is None:
        return jsonify({'rid': rid, 'ok': False,
                        'error': 'Could not decode the uploaded video. '
                                 'Try an .mp4 (H.264) file.'}), 400
    work, n_frames, ww, wh = work_info
    cap_frames = max(10, min(frames or n_frames, n_frames))
    video_name = Path(video.filename).name

    # -- 3. archive previous outputs so this run is self-contained ------------
    archive_outputs()
    logs['_meta'] = (f'video: {video_name} ({n_frames} frames) | working copy '
                     f'{ww}x{wh} | frames processed: {cap_frames} | mode: {mode}')

    # Standard location all module scripts read; modules that can take an
    # explicit path also get one.
    (VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    std_video = VIDEO_DIR / 'conveyor.mp4'
    shutil.copy2(str(work), str(std_video))

    py = sys.executable
    failed = []

    # -- 4. Module B (optional): checkerboard calibration ---------------------
    cal_files = request.files.getlist('checkerboards') or []
    if any(f.filename for f in cal_files):
        n_saved = save_images(cal_files, CALIB_DIR, clear=True)
        rc, out = run([py, 'module_B_calibration/calibrate.py',
                       '--square-size-cm', str(square_cm),
                       '--checkerboard', '9x6'])
        logs['B_calibration'] = (f'{n_saved} checkerboard image(s) -> '
                                 f'returncode={rc}\n{out[-3000:]}')
        if rc != 0:
            failed.append('Module B (calibration failed; continuing without scale)')
    else:
        logs['B_calibration'] = ('skipped (no checkerboard photos) - belt '
                                 'speed uses the assumed 0.05 cm/px scale.')

    # -- 5. Module E data (optional): per-class box crops ---------------------
    crops = save_box_crops()
    if crops:
        logs['E_crops'] = '; '.join(f'{k}: {v} images' for k, v in crops.items())
    else:
        logs['E_crops'] = ('none uploaded - using the bundled synthetic '
                           'box_type_A/B/C templates.')

    # -- 6. full-demo mode: run every module script A..E ----------------------
    if mode == 'all':
        steps = [
            ('A_segmentation', ['module_A_segmentation/segmentation.py']),
            ('C_optical_flow', ['module_C_optical_flow/optical_flow.py']),
            ('D_tracking', ['module_D_tracking/kalman_tracker.py',
                            '--video', str(work),
                            '--frames', str(cap_frames)]),
            ('E_recognition', ['module_E_recognition/recognize.py']),
        ]
        for key, cmd in steps:
            rc, out = run([py, *cmd])
            logs[key] = f'returncode={rc}\n{out[-3500:]}'
            if rc != 0 and key != 'E_recognition':
                failed.append(key)
        if 'C_optical_flow' in logs and 'failed' not in logs['C_optical_flow']:
            pass  # speed comes from pipeline anyway

    # -- 7. the integrated pipeline (single run, per doc) ---------------------
    rc, out = run([py, 'pipeline.py', '--video', str(work),
                   '--frames', str(cap_frames), '--start-frame', str(start)])
    logs['pipeline'] = f'returncode={rc}\n{out[-5000:]}'
    if rc != 0:
        failed.append('pipeline')

    elapsed = time.time() - t0
    has_checkerboard = bool(cal_files and any(True for _ in cal_files))
    return jsonify({
        'rid': rid,
        'ok': True,
        'video': video_name,
        'elapsed': round(elapsed, 1),
        'logs': logs,
        'report': read_report(),
        'images': collect_outputs(),
        'failed': failed,
        'warning': ("No checkerboard photos were uploaded, so pixel->cm uses "
                    "the assumed 0.05 cm/px scale; upload checkerboard photos "
                    "next time to get real sizes and speeds.") if not has_checkerboard
                    else None,
    })


if __name__ == '__main__':
    print('Conveyor Inspection UI:  http://127.0.0.1:5000')
    print('Upload a conveyor video (+ optional checkerboard photos and box ')
    print('crops) to get the inspection report for THAT video.')
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        in_use = probe.connect_ex(('127.0.0.1', 5000)) == 0
    if in_use:
        print('Port 5000 is already in use - a second server instance would')
        print('silently double-bind it. Starting a second copy without a')
        print('custom port is not supported; open the running instance at')
        print('http://127.0.0.1:5000 instead.')
        raise SystemExit(1)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)