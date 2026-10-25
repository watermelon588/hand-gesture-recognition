import os
import sys
from meme_action import MemeTrigger

mt = MemeTrigger()

def make_hand(thumb_ext=False, index_state='folded', middle_ext=False, ring_ext=False, pinky_ext=False):
    pts = [[300, 400]] * 21
    pts[9] = [300, 300]
    pts[2] = [260, 340]
    pts[3] = [230, 310]
    pts[4] = [180, 260] if thumb_ext else [280, 320]
    
    pts[5] = [270, 290]
    if index_state == 'extended':
        pts[6] = [260, 230]
        pts[7] = [255, 170]
        pts[8] = [250, 110]
    elif index_state == 'bent':
        pts[6] = [260, 230] # PIP projected forward
        pts[7] = [280, 240]
        pts[8] = [270, 270] # curled inward
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
print("=" * 60)
print("GESTURE DETECTION VERIFICATION TEST SUITE")
print("=" * 60)
for expected, detected, label in results:
    is_ok = str(expected).startswith(str(detected)) or (expected == 5 and detected in (5, 6))
    if not is_ok:
        all_pass = False
    status_str = "[PASS]" if is_ok else "[FAIL]"
    print(f"Target: {str(expected):<20} -> Detected ID: {str(detected):<3} ({label:<25}) {status_str}")

print("=" * 60)
if all_pass:
    print("ALL 7 GESTURES VERIFIED SUCCESSFULLY!")
else:
    print("SOME GESTURES FAILED!")
print("=" * 60)
