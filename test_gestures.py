import os
import sys
import numpy as np
from meme_action import MemeTrigger

mt = MemeTrigger(hold_frames=10)

def make_hand(thumb_ext=False, index_state='folded', middle_ext=False, ring_ext=False, pinky_ext=False):
    pts = [[300, 400]] * 21
    pts[9] = [300, 300]   # Palm size = 100
    pts[2] = [260, 340]
    pts[3] = [230, 310]
    pts[4] = [170, 260] if thumb_ext else [280, 320]
    
    pts[5] = [270, 290]
    if index_state == 'extended':
        pts[6] = [260, 230]
        pts[7] = [255, 170]
        pts[8] = [250, 110]
    elif index_state == 'bent':
        pts[6] = [260, 220] # PIP projected forward
        pts[7] = [275, 230]
        pts[8] = [270, 270] # curled inward back towards MCP
    else: # folded
        pts[6] = [270, 310] # PIP not projected
        pts[7] = [270, 320]
        pts[8] = [270, 340]

    pts[10] = [300, 220]
    pts[12] = [300, 100] if middle_ext else [300, 340]

    pts[13] = [330, 290]
    pts[14] = [335, 230]
    pts[16] = [340, 110] if ring_ext else [330, 340]

    pts[17] = [360, 310]
    pts[18] = [370, 250]
    pts[20] = [385, 130] if pinky_ext else [355, 340]

    return pts

results = []
# 1 -> thumb + pinky
pts1 = make_hand(thumb_ext=True, pinky_ext=True)
gid1, name1 = mt.detect_custom_gesture(pts1, 'Stop')
results.append((1, gid1, name1))

# 2 -> index
pts2 = make_hand(index_state='extended')
gid2, name2 = mt.detect_custom_gesture(pts2, 'Stop')
results.append((2, gid2, name2))

# 3 -> index bend
pts3 = make_hand(index_state='bent')
gid3, name3 = mt.detect_custom_gesture(pts3, 'Stop')
results.append((3, gid3, name3))

# 4 -> thumb
pts4 = make_hand(thumb_ext=True)
gid4, name4 = mt.detect_custom_gesture(pts4, 'Stop')
results.append((4, gid4, name4))

# 5 -> closed
pts5 = make_hand()
gid5, name5 = mt.detect_custom_gesture(pts5, 'Stop')
results.append((5, gid5, name5))

# 6 -> clockwise motion
gid6_cw, name6_cw = mt.detect_custom_gesture(pts2, 'Clockwise')
results.append(('6 (Clockwise)', gid6_cw, name6_cw))

# 6 -> anticlockwise motion
gid6_ccw, name6_ccw = mt.detect_custom_gesture(pts2, 'Counter Clockwise')
results.append(('6 (Counter Clockwise)', gid6_ccw, name6_ccw))

# 7 -> open
pts7 = make_hand(thumb_ext=True, index_state='extended', middle_ext=True, ring_ext=True, pinky_ext=True)
gid7, name7 = mt.detect_custom_gesture(pts7, 'Stop')
results.append((7, gid7, name7))

all_pass = True
print("=" * 65)
print("GESTURE DETECTION VERIFICATION TEST SUITE")
print("=" * 65)
for expected, detected, label in results:
    is_ok = str(expected).startswith(str(detected))
    if not is_ok:
        all_pass = False
    status_str = "[PASS]" if is_ok else "[FAIL]"
    print(f"Target: {str(expected):<22} -> Detected ID: {str(detected):<3} ({label:<25}) {status_str}")

print("=" * 65)

# --- Temporal Hold & Jitter Stability Test ---
print("\nTESTING TEMPORAL STABILITY & HOLD FILTER:")
test_mt = MemeTrigger(hold_frames=10)

# Simulate 3 frames of random gesture (transient jitter)
for _ in range(3):
    test_mt.trigger(pts2, 'Stop')
print(f"After 3 frames of gesture 2 -> Active Meme: {test_mt.active_meme_id} (Expected None - Not yet held)")
assert test_mt.active_meme_id is None, "Should not trigger on transient 3 frames!"

# Complete 10 frames of gesture 2
for _ in range(7):
    test_mt.trigger(pts2, 'Stop')
print(f"After 10 frames of gesture 2 -> Active Meme: {test_mt.active_meme_id} (Expected 2 - Locked in!)")
assert test_mt.active_meme_id == 2, "Should lock in after 10 frames!"

# Simulate transient 2 frames of another gesture while moving hand
for _ in range(2):
    test_mt.trigger(pts1, 'Stop')
print(f"After 2 transient frames of gesture 1 -> Active Meme: {test_mt.active_meme_id} (Expected 2 - Maintained stability!)")
assert test_mt.active_meme_id == 2, "Active meme should not be disturbed by 2 transient frames!"

# Test Overlay Rendering
dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
rendered_frame = test_mt.draw_overlay(dummy_frame)
assert rendered_frame.shape == (540, 960, 3), "Overlay output shape mismatch!"
print("Overlay rendering verification: [PASS]")

print("=" * 65)
if all_pass:
    print("ALL 7 GESTURES & TEMPORAL STABILITY TESTS PASSED WITH 100% ACCURACY!")
else:
    print("SOME GESTURES FAILED!")
print("=" * 65)
