"""
TİD — Türk İşaret Dili Tanıma  (Production v5 — Dual-Hand)
============================================================
v4 fixes preserved:
  FIX 1 : do_space() deadlock                → FIXED
  FIX 2 : latest_jpeg None guard             → FIXED
  FIX 3 : XSS in renderHist / renderSent     → FIXED
  FIX 4 : Camera spin-loop on failure        → FIXED

v5 new feature:
  ★ DUAL-HAND SUPPORT
    • max_num_hands raised to 2
    • MediaPipe handedness used to sort hands: LEFT always occupies
      feature slots 0–62, RIGHT always occupies slots 63–125.
    • Missing hand → zero-padded (63 zeros), keeping vector length
      consistent for the model.
    • Each hand drawn in its own colour on the video feed.
    • UI shows two independent hand-indicator pills (Sol El / Sağ El).
    • hand_count field added to /state for the frontend.

  ⚠️  MODEL NOTE
    If your model was trained on 63-feature (single-hand) vectors it
    will still work for single-hand signs via zero-padding.
    To get the best accuracy on two-handed signs, retrain on 126-feature
    vectors built with the same LEFT-first ordering used here.

Çalıştır : python app_production_v5.py
Tarayıcı : http://127.0.0.1:5000
"""

import os, time, threading, queue, pickle
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, render_template_string, jsonify, request

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "letter_model_rf.pkl")

# ── Config ──────────────────────────────────────────────────────────
HOLD_DURATION        = 0.9
NO_HAND_TIMEOUT      = 2.0
CONFIDENCE_THRESHOLD = 0.60
CAM_WIDTH, CAM_HEIGHT = 1280, 720
STREAM_FPS           = 30
MAX_HISTORY          = 20
PRED_BUFFER_SIZE     = 12
JPEG_QUALITY         = 80
MAX_CAM_FAILURES     = 100

# ── Visual colours for each hand on the video overlay ───────────────
# LEFT  hand: teal  (matches the existing single-hand colour)
# RIGHT hand: amber (clearly distinct)
COLOUR_LEFT  = (60,  200, 170)   # BGR teal
COLOUR_RIGHT = (40,  170, 255)   # BGR amber-orange
DOT_COLOUR   = (255, 255, 255)

# ── Model ────────────────────────────────────────────────────────────
print("[1/3] Model yükleniyor...")
with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
rf_model = data["model"]
classes  = list(data["classes"])

# Detect whether the model expects 63 (single-hand) or 126 (dual-hand)
# features so we can zero-pad correctly in both cases.
n_features = rf_model.n_features_in_
print(f"    OK — {len(classes)} sınıf: {classes}")
print(f"    Model giriş boyutu: {n_features} özellik "
      f"({'çift el' if n_features >= 126 else 'tek el — sıfır dolgu aktif'})")

# ── MediaPipe ────────────────────────────────────────────────────────
print("[2/3] MediaPipe başlatılıyor...")
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
hands_sol = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,                  # ★ DUAL-HAND: allow up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
print("    OK")

# ── Shared State ─────────────────────────────────────────────────────
state = {
    "letter":       "",
    "confidence":   0.0,
    "hand_detected": False,
    "hand_count":   0,          # ★ new: 0 / 1 / 2
    "left_detected":  False,    # ★ new: individual hand flags
    "right_detected": False,    # ★ new
    "hold_progress": 0.0,
    "word":         "",
    "sentence":     "",
    "history":      [],
    "auto_enabled": True,
}
state_lock  = threading.Lock()
latest_jpeg = None
jpeg_lock   = threading.Lock()
frame_queue = queue.Queue(maxsize=2)

# ── State helpers ────────────────────────────────────────────────────
def do_add_letter(letter):
    with state_lock:
        state["word"] += letter

def do_space():
    """Save current word to sentence. Must NOT be called while holding state_lock."""
    with state_lock:
        w = state["word"].strip()
        if w:
            state["sentence"] += w + " "
            state["history"].insert(0, {"word": w, "time": time.strftime("%H:%M")})
            state["history"] = state["history"][:MAX_HISTORY]
        state["word"] = ""

# ── Feature extraction helpers ───────────────────────────────────────
_SINGLE_LEN = 63   # 21 landmarks × 3
_DUAL_LEN   = 126  # 42 landmarks × 3

def _landmarks_to_array(hand_landmarks):
    """Flatten a single hand's landmarks to a 63-float numpy array."""
    return np.array(
        [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)],
        dtype=np.float32
    )

def build_feature_vector(left_lm, right_lm):
    """
    Build the feature vector fed to the RF model.

    Ordering is always:  [LEFT(63)] + [RIGHT(63)]
    Missing hand → zero block of 63 floats.

    If the model was trained on only 63 features (single-hand) we fall
    back to whichever hand is present (prefer the dominant/right hand).
    """
    left_arr  = _landmarks_to_array(left_lm)  if left_lm  else np.zeros(_SINGLE_LEN, np.float32)
    right_arr = _landmarks_to_array(right_lm) if right_lm else np.zeros(_SINGLE_LEN, np.float32)

    if n_features >= _DUAL_LEN:
        # Model trained on dual-hand vectors
        return np.concatenate([left_arr, right_arr]).reshape(1, -1)
    else:
        # Legacy single-hand model: use whichever hand is available,
        # prefer right (dominant), fall back to left
        arr = right_arr if right_lm else left_arr
        return arr.reshape(1, -1)

# ══════════════════════════════════════════════════════════════════════
# THREAD-1: Kamera (sadece okur)
# ══════════════════════════════════════════════════════════════════════
def capture_thread():
    print("[3/3] Kamera açılıyor...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!")
        return

    print("    OK\n")
    print("✅  http://127.0.0.1:5000  adresini tarayıcıda açın\n")

    fail_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count >= MAX_CAM_FAILURES:
                print(f"[HATA] Kamera {MAX_CAM_FAILURES} ardışık hatadan sonra kayboldu.")
                cap.release()
                break
            time.sleep(0.01)
            continue

        fail_count = 0
        frame = cv2.flip(frame, 1)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

# ══════════════════════════════════════════════════════════════════════
# THREAD-2: Inference (MediaPipe + RF)  ★ DUAL-HAND REWRITE
# ══════════════════════════════════════════════════════════════════════
def inference_thread():
    global latest_jpeg

    pred_buffer   = deque(maxlen=PRED_BUFFER_SIZE)
    stable_letter = ""
    stable_start  = None
    letter_added  = False
    no_hand_start = None

    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        h_frame, w_frame = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands_sol.process(rgb)
        rgb.flags.writeable = True

        detected_letter = ""
        confidence_val  = 0.0
        hand_found      = False
        hold_progress   = 0.0
        left_found      = False
        right_found     = False
        hand_count      = 0
        now             = time.time()

        if result.multi_hand_landmarks and result.multi_handedness:
            hand_found = True
            no_hand_start = None

            # ── Separate hands by MediaPipe's handedness label ──────────
            # NOTE: After cv2.flip(frame, 1) the image is mirrored, so
            # MediaPipe's "Right" label corresponds to the user's actual
            # right hand (the mirroring re-aligns labels with reality).
            left_lm  = None
            right_lm = None

            for hlm, handedness in zip(result.multi_hand_landmarks,
                                        result.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"

                if label == "Left":
                    left_lm   = hlm
                    left_found = True
                else:
                    right_lm   = hlm
                    right_found = True

                # ── Draw skeleton in the hand's own colour ───────────────
                colour = COLOUR_LEFT if label == "Left" else COLOUR_RIGHT
                for conn in mp_hands.HAND_CONNECTIONS:
                    x1 = int(hlm.landmark[conn[0]].x * w_frame)
                    y1 = int(hlm.landmark[conn[0]].y * h_frame)
                    x2 = int(hlm.landmark[conn[1]].x * w_frame)
                    y2 = int(hlm.landmark[conn[1]].y * h_frame)
                    cv2.line(frame, (x1, y1), (x2, y2), colour, 2)
                for lm in hlm.landmark:
                    cx, cy = int(lm.x * w_frame), int(lm.y * h_frame)
                    cv2.circle(frame, (cx, cy), 4, DOT_COLOUR, -1)

                # ── Label badge (SOL / SAĞ) near the wrist ───────────────
                wrist = hlm.landmark[0]
                wx = int(wrist.x * w_frame)
                wy = int(wrist.y * h_frame) + 22
                badge_text = "SOL" if label == "Left" else "SAG"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame,
                              (wx - 4, wy - th - 4),
                              (wx + tw + 4, wy + 4),
                              colour, -1)
                cv2.putText(frame, badge_text,
                            (wx, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (15, 15, 15), 1, cv2.LINE_AA)

            hand_count = (1 if left_found else 0) + (1 if right_found else 0)

            # ── Build feature vector and run RF ─────────────────────────
            row   = build_feature_vector(left_lm, right_lm)
            proba = rf_model.predict_proba(row)[0]
            cid   = int(np.argmax(proba))
            conf  = float(proba[cid])
            confidence_val = conf

            if conf > CONFIDENCE_THRESHOLD:
                pred_buffer.append(classes[cid])

            if len(pred_buffer) > 5:
                detected_letter = Counter(pred_buffer).most_common(1)[0][0]

            # ── Auto-hold timer ──────────────────────────────────────────
            with state_lock:
                auto_on = state["auto_enabled"]

            if auto_on and detected_letter:
                if detected_letter == stable_letter:
                    if stable_start is None:
                        stable_start = now
                    elapsed       = now - stable_start
                    hold_progress = min(elapsed / HOLD_DURATION, 1.0)
                    if elapsed >= HOLD_DURATION and not letter_added:
                        do_add_letter(detected_letter)
                        letter_added  = True
                        stable_start  = None
                        hold_progress = 0.0
                        pred_buffer.clear()
                else:
                    stable_letter = detected_letter
                    stable_start  = now
                    letter_added  = False
                    hold_progress = 0.0
            else:
                stable_letter = ""
                stable_start  = None
                letter_added  = False
                hold_progress = 0.0

        else:
            # No hands detected
            stable_letter = ""
            stable_start  = None
            letter_added  = False
            hold_progress = 0.0
            pred_buffer.clear()

            # Auto-space after timeout (deadlock-safe: read flag+word inside
            # lock, then release before calling do_space)
            should_space = False
            with state_lock:
                auto_on = state["auto_enabled"]
                if auto_on:
                    if no_hand_start is None:
                        no_hand_start = now
                    elif now - no_hand_start >= NO_HAND_TIMEOUT:
                        if state["word"]:
                            should_space  = True
                        no_hand_start = None

            if should_space:
                do_space()

        # ── Update shared state ──────────────────────────────────────────
        with state_lock:
            state["letter"]         = detected_letter
            state["confidence"]     = round(confidence_val * 100, 1)
            state["hand_detected"]  = hand_found
            state["hand_count"]     = hand_count
            state["left_detected"]  = left_found
            state["right_detected"] = right_found
            state["hold_progress"]  = round(hold_progress * 100, 1)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with jpeg_lock:
            latest_jpeg = buf.tobytes()

# ══════════════════════════════════════════════════════════════════════
# FLASK
# ══════════════════════════════════════════════════════════════════════
app = Flask(__name__)

def gen_stream():
    waited = 0.0
    while latest_jpeg is None and waited < 5.0:
        time.sleep(0.05)
        waited += 0.05

    interval = 1.0 / STREAM_FPS
    while True:
        with jpeg_lock:
            frame = latest_jpeg
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(interval)

@app.route('/video_feed')
def video_feed():
    return Response(gen_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(state)

@app.route('/word/add_letter', methods=['POST'])
def api_add():
    letter = (request.get_json() or {}).get('letter', '')
    if letter:
        do_add_letter(letter)
    return jsonify(ok=True)

@app.route('/word/space',     methods=['POST'])
def api_space():
    do_space()
    return jsonify(ok=True)

@app.route('/word/backspace', methods=['POST'])
def api_bksp():
    with state_lock:
        state["word"] = state["word"][:-1]
    return jsonify(ok=True)

@app.route('/word/clear', methods=['POST'])
def api_clear():
    with state_lock:
        state["word"] = ""
    return jsonify(ok=True)

@app.route('/sentence/clear', methods=['POST'])
def api_clr_sent():
    with state_lock:
        state["sentence"] = ""
        state["history"]  = []
    return jsonify(ok=True)

@app.route('/toggle_auto', methods=['POST'])
def api_toggle():
    with state_lock:
        state["auto_enabled"] = (request.get_json() or {}).get('enabled', True)
    return jsonify(ok=True)

@app.route('/')
def index():
    return render_template_string(HTML)

# ══════════════════════════════════════════════════════════════════════
# HTML  (v5 — dual-hand UI)
# ══════════════════════════════════════════════════════════════════════
HTML = r'''<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TİD — Türk İşaret Dili Tanıma</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080c10;--s:#0d1117;--s2:#131920;
  --ac:#00e5c0;--ac2:#0091ff;
  --left:#00e5c0;    /* teal  — Sol El */
  --right:#ffaa2a;   /* amber — Sağ El */
  --danger:#ff4d6d;--text:#e2eaf4;--muted:#4a5568;--border:#1e2d3d
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),
                   linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:40px 40px;opacity:.35;pointer-events:none;z-index:0}

.wrap{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:22px}

/* ── Header ── */
header{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:28px;padding-bottom:18px;border-bottom:1px solid var(--border);flex-wrap:wrap;gap:12px}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:42px;height:42px;background:linear-gradient(135deg,var(--ac),var(--ac2));
  border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}
.logo h1{font-size:1.35rem;font-weight:800;
  background:linear-gradient(90deg,var(--ac),var(--ac2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo p{font-size:.68rem;color:var(--muted);font-family:'Space Mono',monospace;
  letter-spacing:1px;text-transform:uppercase}
.hdr-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.badge{display:flex;align-items:center;gap:8px;background:var(--s2);
  border:1px solid var(--border);padding:7px 14px;border-radius:100px;
  font-family:'Space Mono',monospace;font-size:.72rem;color:var(--ac)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ac);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.atoggle{display:flex;align-items:center;gap:9px;background:var(--s2);
  border:1px solid var(--border);padding:7px 14px;border-radius:100px;cursor:pointer;
  font-family:'Space Mono',monospace;font-size:.7rem;color:var(--muted);
  user-select:none;transition:all .2s}
.atoggle.on{border-color:var(--ac);color:var(--ac)}
.pill{width:30px;height:15px;border-radius:8px;background:var(--border);position:relative;transition:all .2s}
.pill.on{background:var(--ac)}
.thumb{width:11px;height:11px;border-radius:50%;background:#fff;
  position:absolute;top:2px;left:2px;transition:all .2s}
.thumb.on{left:17px}

/* ── Layout grid ── */
.grid{display:grid;grid-template-columns:1fr 370px;gap:20px;align-items:start}
.left-col{display:flex;flex-direction:column;gap:14px}

/* ── Camera card ── */
.cam-card{background:var(--s);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.cam-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:12px 18px;border-bottom:1px solid var(--border);background:var(--s2)}
.cam-title{font-family:'Space Mono',monospace;font-size:.68rem;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase}
.wdots{display:flex;gap:5px}
.cdot{width:10px;height:10px;border-radius:50%}
.cam-wrap{position:relative;background:#000;aspect-ratio:16/9}
.cam-feed{width:100%;height:100%;object-fit:cover;display:block}
.corner{position:absolute;width:18px;height:18px;border-color:var(--ac);border-style:solid;opacity:.7;z-index:2}
.tl{top:8px;left:8px;border-width:2px 0 0 2px}.tr{top:8px;right:8px;border-width:2px 2px 0 0}
.bl{bottom:8px;left:8px;border-width:0 0 2px 2px}.br{bottom:8px;right:8px;border-width:0 2px 2px 0}

/* ── ★ Dual-hand indicator bar (bottom of camera) ── */
.hand-bar{
  position:absolute;bottom:0;left:0;right:0;
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(8,12,16,.82);backdrop-filter:blur(6px);
  border-top:1px solid var(--border);
  padding:8px 14px;z-index:4;gap:10px
}
.hand-pill{
  display:flex;align-items:center;gap:7px;
  border:1px solid var(--border);border-radius:100px;
  padding:5px 13px;font-family:'Space Mono',monospace;font-size:.63rem;
  color:var(--muted);transition:all .35s;flex:1;justify-content:center
}
.hand-pill .hicon{font-size:1rem;filter:grayscale(1) opacity(.35);transition:filter .35s}
.hand-pill.left-on {border-color:var(--left);  color:var(--left)}
.hand-pill.left-on  .hicon{filter:none}
.hand-pill.right-on{border-color:var(--right); color:var(--right)}
.hand-pill.right-on .hicon{filter:none}
.hand-count-badge{
  font-family:'Space Mono',monospace;font-size:.62rem;color:var(--muted);
  background:var(--s2);border:1px solid var(--border);
  padding:4px 10px;border-radius:100px;white-space:nowrap;transition:all .35s
}
.hand-count-badge.one {border-color:var(--ac); color:var(--ac)}
.hand-count-badge.two {border-color:var(--right);color:var(--right);
  box-shadow:0 0 10px rgba(255,170,42,.25)}

/* ── Sentence card ── */
.sent-card{background:var(--s);border:1px solid var(--border);border-radius:14px;padding:18px}
.sent-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.sec-title{font-family:'Space Mono',monospace;font-size:.63rem;letter-spacing:3px;
  text-transform:uppercase;color:var(--muted)}
.sent-box{background:var(--s2);border:1px solid var(--border);border-radius:9px;
  padding:12px 16px;font-size:1.05rem;font-weight:600;letter-spacing:2px;
  min-height:50px;display:flex;align-items:center;flex-wrap:wrap;gap:6px;color:var(--text)}

/* ── Right column ── */
.right{display:flex;flex-direction:column;gap:14px}

/* ── Letter card ── */
.letter-card{background:var(--s);border:1px solid var(--border);border-radius:14px;
  padding:24px;text-align:center;position:relative;overflow:hidden}
.letter-card::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(0,229,192,.07) 0%,transparent 70%);
  pointer-events:none}
.lbl{font-family:'Space Mono',monospace;font-size:.63rem;letter-spacing:3px;
  text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.letter-big{font-size:8rem;font-weight:800;line-height:1;
  background:linear-gradient(135deg,var(--ac),var(--ac2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  min-height:110px;display:flex;align-items:center;justify-content:center;
  filter:drop-shadow(0 0 25px rgba(0,229,192,.3));transition:all .15s}
.letter-big.dim{opacity:.15;filter:none}

/* ── ★ Dual-hand mini indicators inside letter card ── */
.hand-mini-row{display:flex;justify-content:center;gap:8px;margin:10px 0 4px}
.hand-mini{display:flex;align-items:center;gap:5px;
  border:1px solid var(--border);border-radius:6px;
  padding:4px 10px;font-family:'Space Mono',monospace;font-size:.58rem;color:var(--muted);
  transition:all .3s}
.hand-mini.lactive{border-color:var(--left); color:var(--left);
  background:rgba(0,229,192,.07)}
.hand-mini.ractive{border-color:var(--right);color:var(--right);
  background:rgba(255,170,42,.07)}
.hmini-dot{width:6px;height:6px;border-radius:50%;
  background:currentColor;opacity:.4;transition:opacity .3s}
.hand-mini.lactive .hmini-dot,
.hand-mini.ractive .hmini-dot{opacity:1}

/* ── Hold ring ── */
.ring-wrap{display:flex;justify-content:center;margin:8px 0 2px}
.ring{width:52px;height:52px;position:relative}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring-track{fill:none;stroke:var(--border);stroke-width:4}
.ring-fill{fill:none;stroke:var(--ac);stroke-width:4;stroke-linecap:round;
  stroke-dasharray:132;stroke-dashoffset:132;transition:stroke-dashoffset .1s linear}
.ring-lbl{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-family:'Space Mono',monospace;font-size:.58rem;color:var(--ac)}

/* ── Confidence bar ── */
.conf-sec{margin-top:10px}
.conf-hdr{display:flex;justify-content:space-between;margin-bottom:6px}
.conf-lbl{font-family:'Space Mono',monospace;font-size:.63rem;color:var(--muted);text-transform:uppercase}
.conf-val{font-family:'Space Mono',monospace;font-size:.72rem;color:var(--ac);font-weight:700}
.bar-bg{height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--ac2),var(--ac));transition:width .3s;width:0}

/* ── Word card ── */
.word-card{background:var(--s);border:1px solid var(--border);border-radius:14px;padding:18px}
.word-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.word-acts{display:flex;gap:6px;flex-wrap:wrap}
.btn{font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:1px;
  padding:5px 10px;border-radius:6px;border:1px solid var(--border);
  background:var(--s2);color:var(--muted);cursor:pointer;transition:all .2s;text-transform:uppercase}
.btn:hover{border-color:var(--ac);color:var(--ac)}
.btn-d:hover{border-color:var(--danger);color:var(--danger)}
.word-box{background:var(--s2);border:1px solid var(--border);border-radius:9px;
  padding:12px 16px;font-size:1.5rem;font-weight:700;letter-spacing:4px;
  min-height:54px;display:flex;align-items:center;color:var(--text);word-break:break-all}
.cursor{display:inline-block;width:2px;height:1.3rem;background:var(--ac);
  margin-left:4px;animation:blink 1s step-end infinite;flex-shrink:0}
@keyframes blink{50%{opacity:0}}

/* ── History card ── */
.hist-card{background:var(--s);border:1px solid var(--border);border-radius:14px;padding:18px}
.hist-list{display:flex;flex-direction:column;gap:7px;max-height:150px;overflow-y:auto;margin-top:12px}
.hist-item{display:flex;align-items:center;justify-content:space-between;
  background:var(--s2);border:1px solid var(--border);padding:7px 13px;border-radius:7px;
  font-family:'Space Mono',monospace;font-size:.72rem}
.hist-word{color:var(--text);letter-spacing:2px}
.hist-time{color:var(--muted);font-size:.62rem}

/* ── Shortcuts ── */
.shortcuts{background:var(--s);border:1px solid var(--border);border-radius:14px;
  padding:14px 18px;display:flex;flex-wrap:wrap;gap:8px}
.sc{display:flex;align-items:center;gap:7px;font-family:'Space Mono',monospace;font-size:.62rem;color:var(--muted)}
kbd{background:var(--s2);border:1px solid var(--border);border-radius:4px;
  padding:2px 6px;font-family:'Space Mono',monospace;font-size:.62rem;color:var(--text)}

/* ── Flash animation ── */
.anim{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(.5);
  font-size:5rem;font-weight:800;color:var(--ac);pointer-events:none;opacity:0;z-index:999;
  transition:all .4s cubic-bezier(.175,.885,.32,1.275)}
.anim.show{opacity:1;transform:translate(-50%,-50%) scale(1)}
.anim.hide{opacity:0;transform:translate(-50%,-80%) scale(1.2);transition:all .3s ease-in}

/* ── Footer ── */
footer{margin-top:28px;padding-top:18px;border-top:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.ft{font-family:'Space Mono',monospace;font-size:.63rem;color:var(--muted)}
.ft-tag{font-family:'Space Mono',monospace;font-size:.63rem;color:var(--muted);
  background:var(--s2);border:1px solid var(--border);padding:3px 9px;border-radius:4px}

@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="anim" id="anim"></div>
<div class="wrap">

  <header>
    <div class="logo">
      <div class="logo-icon">🤟</div>
      <div>
        <h1>TİD Tanıma</h1>
        <p>Türk İşaret Dili · Gerçek Zamanlı AI · Çift El</p>
      </div>
    </div>
    <div class="hdr-right">
      <div class="atoggle on" id="atog" onclick="toggleAuto()">
        <div class="pill on" id="pill"><div class="thumb on" id="thumb"></div></div>OTOMATİK
      </div>
      <div class="badge"><div class="dot"></div>CANLI · MediaPipe + RF</div>
    </div>
  </header>

  <div class="grid">
    <!-- LEFT COLUMN -->
    <div class="left-col">
      <div class="cam-card">
        <div class="cam-hdr">
          <span class="cam-title">◉ Canlı Kamera — Çift El Modu</span>
          <div class="wdots">
            <div class="cdot" style="background:#ff5f57"></div>
            <div class="cdot" style="background:#febc2e"></div>
            <div class="cdot" style="background:#28c840"></div>
          </div>
        </div>
        <div class="cam-wrap">
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          <img class="cam-feed" src="/video_feed">

          <!-- ★ Dual-hand indicator bar -->
          <div class="hand-bar">
            <div class="hand-pill" id="pillLeft">
              <span class="hicon">🫲</span>SOL EL
            </div>
            <div class="hand-count-badge" id="handCountBadge">0 EL</div>
            <div class="hand-pill" id="pillRight">
              SAĞ EL<span class="hicon">🫱</span>
            </div>
          </div>
        </div>
      </div>

      <div class="sent-card">
        <div class="sent-hdr">
          <span class="sec-title">Oluşan Cümle</span>
          <button class="btn btn-d" onclick="clearSent()">Temizle</button>
        </div>
        <div class="sent-box" id="sentBox">
          <span style="color:var(--muted);font-size:.82rem;font-family:'Space Mono',monospace;letter-spacing:1px;">
            İşaret yapın, kelimeler burada birikecek...
          </span>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div class="right">
      <div class="letter-card">
        <div class="lbl">Tanınan Harf</div>
        <div class="letter-big dim" id="lbig">—</div>

        <!-- ★ Mini dual-hand status inside the letter card -->
        <div class="hand-mini-row">
          <div class="hand-mini" id="miniLeft">
            <div class="hmini-dot"></div>SOL
          </div>
          <div class="hand-mini" id="miniRight">
            SAĞ<div class="hmini-dot"></div>
          </div>
        </div>

        <div class="ring-wrap">
          <div class="ring">
            <svg viewBox="0 0 48 48">
              <circle class="ring-track" cx="24" cy="24" r="21"/>
              <circle class="ring-fill" id="ring" cx="24" cy="24" r="21"/>
            </svg>
            <div class="ring-lbl" id="rlbl"></div>
          </div>
        </div>

        <div class="conf-sec">
          <div class="conf-hdr">
            <span class="conf-lbl">Güven</span>
            <span class="conf-val" id="cval">0%</span>
          </div>
          <div class="bar-bg"><div class="bar-fill" id="cbar"></div></div>
        </div>
      </div>

      <div class="word-card">
        <div class="word-hdr">
          <span class="sec-title">Mevcut Kelime</span>
          <div class="word-acts">
            <button class="btn" onclick="addL()">+ Harf (E)</button>
            <button class="btn" onclick="addSp()">Boşluk</button>
            <button class="btn" onclick="bksp()">← Sil</button>
            <button class="btn btn-d" onclick="clearW()">Temizle</button>
          </div>
        </div>
        <div class="word-box"><span id="wt"></span><span class="cursor"></span></div>
      </div>

      <div class="hist-card">
        <div class="sec-title">Kelime Geçmişi</div>
        <div class="hist-list" id="hist">
          <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:var(--muted);text-align:center;padding:18px 0;">
            Henüz kelime eklenmedi
          </div>
        </div>
      </div>

      <div class="shortcuts">
        <div class="sc"><kbd>E</kbd>Harf ekle</div>
        <div class="sc"><kbd>Space</kbd>Boşluk/Kaydet</div>
        <div class="sc"><kbd>⌫</kbd>Geri al</div>
        <div class="sc"><kbd>Esc</kbd>Temizle</div>
        <div class="sc"><kbd>A</kbd>Auto aç/kapat</div>
      </div>
    </div>
  </div>

  <footer>
    <span class="ft">Bitirme Projesi · TİD İşaret Dili Tanıma Sistemi · Çift El Desteği</span>
    <span class="ft-tag">v5.0 · 2025</span>
  </footer>
</div><!-- /wrap -->

<script>
const RING_C = 132;
let curLetter = "", autoOn = true, lastWord = "", lastSent = "";

/* ── XSS guard ──────────────────────────────────────────────────── */
function esc(s){
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

/* ── Hold ring ──────────────────────────────────────────────────── */
function setRing(p){
  document.getElementById('ring').style.strokeDashoffset = RING_C*(1 - p/100);
  document.getElementById('rlbl').textContent = p > 5 ? Math.round(p)+'%' : '';
}

/* ── ★ Dual-hand UI updater ─────────────────────────────────────── */
function updateHandUI(d){
  const pL   = document.getElementById('pillLeft');
  const pR   = document.getElementById('pillRight');
  const mL   = document.getElementById('miniLeft');
  const mR   = document.getElementById('miniRight');
  const badge= document.getElementById('handCountBadge');
  const cnt  = d.hand_count || 0;

  // Camera bar pills
  pL.classList.toggle('left-on',  !!d.left_detected);
  pR.classList.toggle('right-on', !!d.right_detected);

  // Letter-card mini indicators
  mL.classList.toggle('lactive', !!d.left_detected);
  mR.classList.toggle('ractive', !!d.right_detected);

  // Count badge
  badge.className = 'hand-count-badge' + (cnt===2?' two':cnt===1?' one':'');
  badge.textContent = cnt === 0 ? 'EL ALGILANMIYOR'
                    : cnt === 1 ? '1 EL ALGILANDI'
                    :             '2 EL ALGILANDI ✦';
}

/* ── Main poll loop ─────────────────────────────────────────────── */
async function poll(){
  try{
    const d = await fetch('/state').then(r => r.json());
    curLetter = d.letter;

    // Letter display
    const lb = document.getElementById('lbig');
    if(d.hand_detected && d.letter){ lb.textContent = d.letter; lb.classList.remove('dim'); }
    else                            { lb.textContent = '—';     lb.classList.add('dim');    }

    // Confidence
    document.getElementById('cval').textContent  = d.confidence + '%';
    document.getElementById('cbar').style.width  = d.confidence + '%';
    setRing(d.hold_progress);

    // ★ Dual-hand indicators
    updateHandUI(d);

    // Word
    if(d.word !== lastWord){
      if(d.word.length > lastWord.length) showAnim(d.word.slice(-1));
      lastWord = d.word;
      document.getElementById('wt').textContent = d.word;
    }

    // Sentence & history
    if(d.sentence !== lastSent){ lastSent = d.sentence; renderSent(d.sentence); }
    renderHist(d.history);

  }catch(e){}
  setTimeout(poll, 80);
}

function renderSent(s){
  const el = document.getElementById('sentBox');
  if(!s.trim()){
    el.innerHTML = '<span style="color:var(--muted);font-size:.82rem;font-family:Space Mono,monospace;letter-spacing:1px;">İşaret yapın, kelimeler burada birikecek...</span>';
    return;
  }
  el.innerHTML = s.trim().split(' ').map(w =>
    `<span style="background:var(--s2);border:1px solid var(--border);padding:3px 9px;border-radius:5px;letter-spacing:2px;">${esc(w)}</span>`
  ).join('');
}

function renderHist(h){
  const el = document.getElementById('hist');
  if(!h || !h.length){
    el.innerHTML = '<div style="font-family:Space Mono,monospace;font-size:.68rem;color:var(--muted);text-align:center;padding:18px 0;">Henüz kelime eklenmedi</div>';
    return;
  }
  el.innerHTML = h.slice(0,10).map(x =>
    `<div class="hist-item"><span class="hist-word">${esc(x.word)}</span><span class="hist-time">${esc(x.time)}</span></div>`
  ).join('');
}

function showAnim(l){
  const el = document.getElementById('anim');
  el.textContent = l; el.className = 'anim show';
  setTimeout(()=> el.className = 'anim hide', 600);
  setTimeout(()=> el.className = 'anim',      1000);
}

async function addL()  { if(curLetter){ await fetch('/word/add_letter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({letter:curLetter})}); showAnim(curLetter); } }
async function addSp() { await fetch('/word/space',     {method:'POST'}); }
async function bksp()  { await fetch('/word/backspace', {method:'POST'}); }
async function clearW(){ await fetch('/word/clear',     {method:'POST'}); }
async function clearSent(){ await fetch('/sentence/clear',{method:'POST'}); lastSent=''; renderSent(''); }

async function toggleAuto(){
  autoOn = !autoOn;
  await fetch('/toggle_auto',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:autoOn})});
  ['atog','pill','thumb'].forEach(id=>{
    const el = document.getElementById(id);
    autoOn ? el.classList.add('on') : el.classList.remove('on');
  });
}

document.addEventListener('keydown', e => {
  if     (e.key==='e'||e.key==='E')  addL();
  else if(e.key===' ')               { e.preventDefault(); addSp(); }
  else if(e.key==='Backspace')       bksp();
  else if(e.key==='Escape')          clearW();
  else if(e.key==='a'||e.key==='A')  toggleAuto();
});

poll();
</script>
</body>
</html>'''

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    threading.Thread(target=capture_thread,   daemon=True).start()
    threading.Thread(target=inference_thread, daemon=True).start()
    time.sleep(2)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)