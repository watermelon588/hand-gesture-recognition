import os
import sys
import numpy as np
from meme_action import MemeTrigger

mt = MemeTrigger(hold_frames=8)

def make_hand(thumb_ext=False, thumb_up=False, index_state='folded', middle_ext=False, ring_ext=False, pinky_ext=False, is_left=False):
    # Right hand: thumb is to the left (x < wrist), pinky to the right (x > wrist)
    # Left hand: thumb is to the right (x > wrist), pinky to the left (x < wrist)
    sign = -1 if is_left else 1
    wrist_x, wrist_y = 300, 400
    pts = [[wrist_x, wrist_y]] * 21
    pts[9] = [300, 300]   # Palm center (MCP 9), palm_size = 100

    # Thumb
    pts[2] = [wrist_x - 40 * sign, 340]
    pts[3] = [wrist_x - 70 * sign, 310]
    if thumb_up:
        pts[4] = [wrist_x - 50 * sign, 240]  # Pointing UP (y=240 < y=340)
    elif thumb_ext:
        pts[4] = [wrist_x - 120 * sign, 310] # Pointing outward
    else:
        pts[4] = [wrist_x - 20 * sign, 320]  # Folded across palm

    # Index (5, 6, 7, 8)
    pts[5] = [wrist_x - 30 * sign, 290]
    if index_state == 'extended':
        pts[6] = [wrist_x - 40 * sign, 230]
        pts[7] = [wrist_x - 45 * sign, 170]
        pts[8] = [wrist_x - 50 * sign, 110]
    elif index_state == 'bent':
        pts[6] = [wrist_x - 40 * sign, 220] # PIP projected forward
        pts[7] = [wrist_x - 25 * sign, 230]
        pts[8] = [wrist_x - 30 * sign, 260] # Tip curled inward
    elif index_state == 'inward':
        pts[6] = [wrist_x - 10 * sign, 270]
        pts[7] = [wrist_x + 30 * sign, 270]
        pts[8] = [wrist_x + 70 * sign, 270] # pointing horizontally inward at chest
    else: # folded
        pts[6] = [wrist_x - 30 * sign, 310]
        pts[7] = [wrist_x - 30 * sign, 320]
        pts[8] = [wrist_x - 30 * sign, 340]

    # Middle (9, 10, 11, 12)
    pts[10] = [300, 220]
    pts[12] = [300, 100] if middle_ext else [300, 340]

    # Ring (13, 14, 15, 16)
    pts[13] = [wrist_x + 30 * sign, 290]
    pts[14] = [wrist_x + 35 * sign, 230]
    pts[16] = [wrist_x + 40 * sign, 110] if ring_ext else [wrist_x + 30 * sign, 340]

    # Pinky (17, 18, 19, 20)
    pts[17] = [wrist_x + 60 * sign, 310]
    pts[18] = [wrist_x + 70 * sign, 250]
    pts[20] = [wrist_x + 85 * sign, 130] if pinky_ext else [wrist_x + 55 * sign, 340]

    return pts

tests = [
    # 1 -> Cat Shaka (🤙)
    ("1: Shaka (Right Hand)", make_hand(thumb_ext=True, pinky_ext=True, is_left=False), "Stop", "Unknown", "Right", 1),
    ("1: Shaka (Left Hand)", make_hand(thumb_ext=True, pinky_ext=True, is_left=True), "Stop", "Unknown", "Left", 1),

    # 2 -> 1 Finger: Monkey Pointing (☝️)
    ("2: 1 Finger Point (Right Hand)", make_hand(index_state='extended', is_left=False), "Stop", "Pointer", "Right", 2),
    ("2: 1 Finger Point (Left Hand)", make_hand(index_state='extended', is_left=True), "Stop", "Pointer", "Left", 2),

    # 4 -> Monkey Thumbs Up (👍)
    ("4: Thumbs Up (Right Hand)", make_hand(thumb_up=True, is_left=False), "Stop", "Unknown", "Right", 4),
    ("4: Thumbs Up (Left Hand)", make_hand(thumb_up=True, is_left=True), "Stop", "Unknown", "Left", 4),

    # 5 -> 4 Fingers: Cat 4-Finger Hold (🐾)
    ("5: 4 Fingers (Right Hand)", make_hand(thumb_ext=False, index_state='extended', middle_ext=True, ring_ext=True, pinky_ext=True, is_left=False), "Stop", "Unknown", "Right", 5),
    ("5: 4 Fingers (Left Hand)", make_hand(thumb_ext=False, index_state='extended', middle_ext=True, ring_ext=True, pinky_ext=True, is_left=True), "Stop", "Unknown", "Left", 5),

    # 7 -> 5 Fingers: J. Cole Open Hand (🖐️)
    ("7: 5 Fingers / Open (Right Hand)", make_hand(thumb_ext=True, index_state='extended', middle_ext=True, ring_ext=True, pinky_ext=True, is_left=False), "Stop", "Open", "Right", 7),
    ("7: 5 Fingers / Open (Left Hand)", make_hand(thumb_ext=True, index_state='extended', middle_ext=True, ring_ext=True, pinky_ext=True, is_left=True), "Stop", "Open", "Left", 7),

    # 8 -> 2 Fingers: Cool Cat Peace (✌️)
    ("8: 2 Fingers Peace (Right Hand)", make_hand(thumb_ext=False, index_state='extended', middle_ext=True, ring_ext=False, pinky_ext=False, is_left=False), "Stop", "Unknown", "Right", 8),
    ("8: 2 Fingers Peace (Left Hand)", make_hand(thumb_ext=False, index_state='extended', middle_ext=True, ring_ext=False, pinky_ext=False, is_left=True), "Stop", "Unknown", "Left", 8),

    # Removed Gestures & Dynamic Motion Overrides:
    ("Removed: Hooked Index (Old 3)", make_hand(index_state='bent', is_left=False), "Stop", "Unknown", "Right", None),
    ("Removed: Closed Fist Left (Old 5)", make_hand(is_left=True), "Stop", "Close", "Left", None),
    ("Removed: Closed Fist Right (Old 6)", make_hand(is_left=False), "Stop", "Close", "Right", None),
    ("Dynamic: Clockwise Circle Motion", make_hand(index_state='extended', is_left=False), "Clockwise", "Pointer", "Right", None),
    ("Dynamic: CCW Circle Motion", make_hand(index_state='extended', is_left=False), "Counter Clockwise", "Pointer", "Right", None),
    ("Removed: Point at Chest (Old 8)", make_hand(thumb_ext=True, index_state='inward', is_left=False), "Stop", "Unknown", "Right", None),
]

all_pass = True
print("=" * 75)
print("GESTURE & HANDEDNESS VERIFICATION TEST SUITE (LEFT + RIGHT HANDS)")
print("=" * 75)
for test_name, pts, motion, sign_label, hand, expected_id in tests:
    det_id, det_name = mt.detect_custom_gesture(pts, motion, sign_label, hand)
    is_ok = (det_id == expected_id)
    if not is_ok:
        all_pass = False
    status = "[PASS]" if is_ok else "[FAIL]"
    print(f"{test_name:<34} -> Detected ID: {str(det_id):<2} ({det_name:<24}) {status}")

print("=" * 75)

# --- Dynamic Gesture Lifecycle & Hold-to-Confirm Temporal Smoothing Test ---
print("\nVERIFYING DYNAMIC MOTION GESTURES & LIFECYCLE (START / CLOSE / STOP):")
test_mt = MemeTrigger(hold_frames=8, start_active=False)
right_point = make_hand(index_state='extended', is_left=False)

# 1. Standby check: static gestures must NOT trigger memes while standby
for _ in range(10):
    test_mt.trigger(right_point, 'Stop', 'Pointer', 'Right')
assert test_mt.active_meme_id is None, "Should NOT trigger meme while in Standby!"
assert not test_mt.detection_enabled, "Detection should be disabled initially!"
print("Standby rejection of static gestures: [PASS]")

# 2. Dynamic Start: 'Counter Clockwise' gesture starts detection
test_mt.trigger(right_point, 'Counter Clockwise', 'Pointer', 'Right')
assert test_mt.detection_enabled, "Counter Clockwise should enable detection!"
assert test_mt.active_meme_id is None, "Start gesture itself should not activate a meme!"
print("Counter Clockwise dynamic gesture starts detection: [PASS]")

# 3. 3 frames (transient) while active
for _ in range(3):
    test_mt.trigger(right_point, 'Stop', 'Pointer', 'Right')
assert test_mt.active_meme_id is None, "Should not trigger on 3 transient frames!"
print("Transient 3 frames rejected properly: [PASS]")

# 4. 8 frames (confirmed) -> locks into Meme 2
for _ in range(5):
    test_mt.trigger(right_point, 'Stop', 'Pointer', 'Right')
assert test_mt.active_meme_id == 2, "Should lock into Meme 2 after 8 frames!"
print("Hold-to-confirm locked into Meme 2: [PASS]")

# 5. Dynamic Stop / Close: 'Clockwise' gesture closes active meme and stops detection
test_mt.trigger(right_point, 'Clockwise', 'Pointer', 'Right')
assert test_mt.active_meme_id is None, "Clockwise should close active meme!"
assert not test_mt.detection_enabled, "Clockwise should stop detection!"
print("Clockwise dynamic gesture closes active meme & stops detection: [PASS]")

# 6. Re-verify standby: static gestures do not trigger after stop
for _ in range(10):
    test_mt.trigger(right_point, 'Stop', 'Pointer', 'Right')
assert test_mt.active_meme_id is None, "Should not trigger meme after stopped!"
print("Static gestures rejected after stop: [PASS]")

# 7. Test Overlay Rendering in both Standby and Active states
dummy = np.zeros((540, 960, 3), dtype=np.uint8)
rendered_standby = test_mt.draw_overlay(dummy.copy())
assert rendered_standby.shape == (540, 960, 3), "Overlay dimension mismatch in standby!"

test_mt.detection_enabled = True
test_mt.active_meme_id = 8  # Cool cat peace
rendered_active = test_mt.draw_overlay(dummy.copy())
assert rendered_active.shape == (540, 960, 3), "Overlay dimension mismatch in active!"
print("In-video card overlay render (Standby & Active): [PASS]")

print("=" * 75)
if all_pass:
    print(f"ALL {len(tests)} STATIC TESTS + ALL DYNAMIC LIFECYCLE TESTS PASSED WITH 100% SUCCESS!")
else:
    print("SOME TEST CASES FAILED!")
print("=" * 75)
