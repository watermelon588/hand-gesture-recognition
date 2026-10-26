#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meme Action Handler:
Maps real-time hand gestures and motions directly to meme reference images (1.jpg - 8.jpg).
Renders seamlessly inside the video window with hold-to-confirm temporal smoothing.

Supports BOTH Left and Right hands with hand-agnostic kinematic geometry
and synergy with KeyPointClassifier ML predictions.
"""
import os
import math
import cv2
import numpy as np


class MemeTrigger:
    def __init__(self, memes_dir='memes', hold_frames=8, start_active=False, **kwargs):
        self.memes_dir = memes_dir
        self.hold_frames = hold_frames  # Consecutive frames to confirm a gesture (~0.25 - 0.3s)
        self.detection_enabled = start_active  # Standby by default; started via Counter Clockwise dynamic gesture

        # Exact user file mapping (supports both .png and .jpg)
        self.meme_files = {}
        for mid in [1, 2, 4, 5, 7, 8]:
            png_p = os.path.join(memes_dir, f"{mid}.png")
            jpg_p = os.path.join(memes_dir, f"{mid}.jpg")
            if os.path.exists(png_p):
                self.meme_files[mid] = png_p
            elif os.path.exists(jpg_p):
                self.meme_files[mid] = jpg_p
            else:
                self.meme_files[mid] = png_p

        self.meme_labels = {
            1: "Cat Shaka (Thumb+Pinky)",
            2: "Monkey Point (1 Finger)",
            4: "Monkey Thumbs Up",
            5: "Cat 4-Finger Hold (4 Fingers)",
            7: "J. Cole Open Hand (5 Fingers)",
            8: "Cool Cat Peace (2 Fingers)",
        }

        # Cache pre-fitted card images (letterboxed to avoid distortion)
        self.card_size = (270, 315)
        self.img_box_size = (254, 230)
        self.cached_images = {}
        self._load_and_cache_images()

        # State tracking
        self.active_meme_id = None
        self.active_meme_label = ""
        self.flash_frames = 0
        self.status_message = "STANDBY" if not self.detection_enabled else "ACTIVE"

        # Temporal hold state machine
        self.current_candidate = None
        self.candidate_name = ""
        self.hold_count = 0

    def _load_and_cache_images(self):
        """Loads and prepares high quality letterboxed images for the in-video card."""
        target_w, target_h = self.img_box_size
        for gid, path in self.meme_files.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    h, w, _ = img.shape
                    scale = min(target_w / w, target_h / h)
                    nw = int(w * scale)
                    nh = int(h * scale)
                    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

                    # Create letterboxed canvas with dark background
                    canvas = np.full((target_h, target_w, 3), 22, dtype=np.uint8)
                    x_off = (target_w - nw) // 2
                    y_off = (target_h - nh) // 2
                    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
                    self.cached_images[gid] = canvas

    def detect_custom_gesture(self, landmark_list, motion_gesture, hand_sign_label="Unknown", handedness="Right"):
        """
        Classifies 21 MediaPipe hand landmarks into meme gestures (1, 2, 4, 5, 7, 8).
        Fully compatible with both Left and Right hands.
        """
        # Motion Gesture override: When drawing circles in the air, don't trigger static finger gestures
        if motion_gesture in ["Clockwise", "Counter Clockwise"]:
            return None, f"Motion: {motion_gesture}"

        if not landmark_list or len(landmark_list) < 21:
            return None, "None"

        pts = landmark_list

        def dist(i1, i2):
            return math.hypot(pts[i1][0] - pts[i2][0], pts[i1][1] - pts[i2][1])

        def get_angle(i1, i2, i3):
            a, b, c = pts[i1], pts[i2], pts[i3]
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            m1 = math.hypot(v1[0], v1[1])
            m2 = math.hypot(v2[0], v2[1])
            if m1 * m2 == 0:
                return 180.0
            cos = max(-1.0, min(1.0, dot / (m1 * m2)))
            return math.degrees(math.acos(cos))

        # Palm scale normalization (wrist 0 to middle MCP 9)
        palm_size = dist(0, 9)
        if palm_size < 1:
            palm_size = 1.0

        # --- Hand-Agnostic Finger States (Left & Right invariant) ---
        # 1. Index (5: MCP, 6: PIP, 7: DIP, 8: TIP)
        index_angle = get_angle(5, 6, 8)
        index_extended = (dist(8, 0) > dist(6, 0) * 1.15) and (dist(8, 5) > dist(6, 5) * 1.05) and (index_angle > 135)
        index_folded = not index_extended

        # 2. Middle (9: MCP, 10: PIP, 11: DIP, 12: TIP)
        middle_angle = get_angle(9, 10, 12)
        middle_extended = (dist(12, 0) > dist(10, 0) * 1.15) and (dist(12, 9) > dist(10, 9) * 1.05) and (middle_angle > 135)
        middle_folded = (dist(12, 0) <= dist(10, 0) * 1.1) or (middle_angle < 115) or (not middle_extended)

        # 3. Ring (13: MCP, 14: PIP, 15: DIP, 16: TIP)
        ring_angle = get_angle(13, 14, 16)
        ring_extended = (dist(16, 0) > dist(14, 0) * 1.15) and (dist(16, 13) > dist(14, 13) * 1.05) and (ring_angle > 135)
        ring_folded = (dist(16, 0) <= dist(14, 0) * 1.1) or (ring_angle < 115) or (not ring_extended)

        # 4. Pinky (17: MCP, 18: PIP, 19: DIP, 20: TIP)
        pinky_angle = get_angle(17, 18, 20)
        pinky_extended = (dist(20, 0) > dist(18, 0) * 1.15) and (dist(20, 17) > dist(18, 17) * 1.05) and (pinky_angle > 135)
        pinky_folded = (dist(20, 0) <= dist(18, 0) * 1.1) or (pinky_angle < 115) or (not pinky_extended)

        # 5. Thumb (0: Wrist, 2: MCP, 3: IP, 4: TIP)
        thumb_extended = (dist(4, 2) > 0.55 * palm_size) and (dist(4, 9) > 0.65 * palm_size)
        thumb_pointing_up = (pts[4][1] < pts[2][1] - 0.2 * palm_size) and thumb_extended and (abs(pts[4][1] - pts[2][1]) > abs(pts[4][0] - pts[2][0]))

        # --- Gesture Decision Rules ---

        # 7 & 5 -> 5 Fingers (Open Hand) & 4 Fingers (Cat 4-Finger Hold):
        all_four_extended = index_extended and middle_extended and ring_extended and pinky_extended
        if all_four_extended:
            if thumb_extended:
                return 7, "Open Hand (5 Fingers)"
            else:
                return 5, "4 Fingers"

        if hand_sign_label == "Open" and thumb_extended and not thumb_pointing_up:
            return 7, "Open Hand (5 Fingers)"

        # 1 -> Thumb + Pinky (Cat Shaka 🤙):
        if thumb_extended and pinky_extended and middle_folded and ring_folded and index_folded:
            return 1, "Thumb + Pinky"

        # 4 -> Thumb Up (Monkey Thumbs Up 👍):
        if thumb_pointing_up and (not pinky_extended) and index_folded and middle_folded and ring_folded:
            return 4, "Thumb Up"

        # 8 -> 2 Fingers (Cool Cat Peace ✌️):
        if index_extended and middle_extended and ring_folded and pinky_folded and (not thumb_pointing_up):
            return 8, "2 Fingers"

        # 2 -> 1 Finger (Monkey Pointing Up ☝️):
        if (index_extended or hand_sign_label == "Pointer") and middle_folded and ring_folded and pinky_folded and (not thumb_pointing_up):
            return 2, "1 Finger (Index Point)"

        return None, "Unknown"

    def trigger(self, landmark_list, motion_gesture, hand_sign_label="Unknown", handedness="Right"):
        """
        Feeds new frame landmark data and motion gesture.
        Dynamic motion gestures:
          - 'Counter Clockwise' (↺): Starts meme detection.
          - 'Clockwise' (↻): Closes the active meme and stops/pauses detection.
        Uses temporal hold smoothing to confirm gestures stably.
        """
        # 1. Dynamic Motion Gestures (Highest Priority)
        if motion_gesture == "Counter Clockwise":
            if not self.detection_enabled:
                self.detection_enabled = True
                self.flash_frames = 12
                self.status_message = "DETECTION STARTED (CCW)"
                print("[DYNAMIC GESTURE] Counter Clockwise (CCW) -> Meme Detection STARTED!")
            self.current_candidate = None
            self.hold_count = 0
            return self.active_meme_id

        elif motion_gesture == "Clockwise":
            if self.detection_enabled or (self.active_meme_id is not None):
                self.detection_enabled = False
                self.active_meme_id = None
                self.active_meme_label = ""
                self.current_candidate = None
                self.hold_count = 0
                self.flash_frames = 12
                self.status_message = "MEME CLOSED / STOPPED (CW)"
                print("[DYNAMIC GESTURE] Clockwise (CW) -> Active Meme CLOSED & Detection STOPPED!")
            return None

        # 2. If detection is paused / standby, do not evaluate static gestures
        if not self.detection_enabled:
            self.current_candidate = None
            self.hold_count = 0
            return self.active_meme_id

        # 3. Static Hand Gesture Evaluation
        raw_id, raw_name = self.detect_custom_gesture(landmark_list, motion_gesture, hand_sign_label, handedness)

        if raw_id is not None:
            if raw_id == self.current_candidate:
                self.hold_count += 1
            else:
                self.current_candidate = raw_id
                self.candidate_name = raw_name
                self.hold_count = 1

            # When gesture has been held stably for required frames
            if self.hold_count >= self.hold_frames:
                self.hold_count = self.hold_frames  # cap

                target_id = self.current_candidate

                if self.active_meme_id != target_id:
                    self.active_meme_id = target_id
                    self.active_meme_label = self.meme_labels.get(self.active_meme_id, self.candidate_name)
                    self.flash_frames = 8  # trigger visual pulse on the card
                    print(f"[MEME CONFIRMED] Switched to Meme #{self.active_meme_id}: {self.active_meme_label}")
        else:
            self.hold_count = max(0, self.hold_count - 1)
            if self.hold_count == 0:
                self.current_candidate = None
                self.candidate_name = ""

        return self.active_meme_id

    def close_active_meme(self):
        """Closes the currently displayed meme."""
        self.active_meme_id = None
        self.active_meme_label = ""
        self.current_candidate = None
        self.hold_count = 0
        self.flash_frames = 10
        self.status_message = "MEME CLOSED"
        print("[ACTION] Active meme closed.")

    def toggle_detection(self):
        """Toggles detection state between active and standby."""
        self.detection_enabled = not self.detection_enabled
        if not self.detection_enabled:
            self.active_meme_id = None
            self.active_meme_label = ""
            self.status_message = "DETECTION PAUSED"
            print("[ACTION] Meme detection PAUSED.")
        else:
            self.status_message = "DETECTION STARTED"
            print("[ACTION] Meme detection STARTED.")
        self.flash_frames = 10

    def update_no_hand(self):
        """Called when no hand is in the frame to smoothly reset candidate hold state."""
        self.hold_count = max(0, self.hold_count - 2)
        if self.hold_count == 0:
            self.current_candidate = None
            self.candidate_name = ""

    def draw_overlay(self, image):
        """
        Renders the active meme card, dynamic gesture status, and hold progress directly on the OpenCV frame.
        """
        im_h, im_w, _ = image.shape
        card_w, card_h = self.card_size

        if im_w < 720:
            scale_factor = 0.85
            card_w = int(card_w * scale_factor)
            card_h = int(card_h * scale_factor)

        margin = 15
        x1 = im_w - card_w - margin
        y1 = margin
        x2 = x1 + card_w
        y2 = y1 + card_h

        # 1. Dark frosted-glass card background
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 26), -1)
        cv2.addWeighted(overlay, 0.88, image, 0.12, 0, image)

        # 2. Glowing card border
        if self.flash_frames > 0:
            border_color = (0, 230, 255)  # Gold pulse
            border_thick = 3
            self.flash_frames -= 1
        elif self.active_meme_id:
            border_color = (0, 220, 140)  # Neon Emerald Green
            border_thick = 2
        elif self.detection_enabled:
            border_color = (0, 200, 240)  # Bright Cyan
            border_thick = 2
        else:
            border_color = (120, 100, 70)  # Subtle Standby Amber/Slate
            border_thick = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, border_thick)

        # 3. Card Header Bar
        header_h = 32
        cv2.rectangle(image, (x1, y1), (x2, y1 + header_h), (35, 35, 45), -1)
        cv2.rectangle(image, (x1, y1 + header_h), (x2, y1 + header_h), border_color, 1)

        if self.active_meme_id:
            title_text = f"MEME #{self.active_meme_id}: {self.active_meme_label}"
        elif self.detection_enabled:
            title_text = "MEME CONTROLLER: ACTIVE"
        else:
            title_text = "MEME CONTROLLER: STANDBY"

        cv2.putText(image, title_text, (x1 + 8, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Active Meme Image Display vs Guide Layout
        if self.active_meme_id and self.active_meme_id in self.cached_images:
            cached_img = self.cached_images[self.active_meme_id]
            display_w = card_w - 16
            display_h = card_h - header_h - 44
            if (cached_img.shape[1] != display_w) or (cached_img.shape[0] != display_h):
                img_to_draw = cv2.resize(cached_img, (display_w, display_h), interpolation=cv2.INTER_AREA)
            else:
                img_to_draw = cached_img

            img_x1 = x1 + 8
            img_y1 = y1 + header_h + 6
            image[img_y1:img_y1 + display_h, img_x1:img_x1 + display_w] = img_to_draw
        else:
            guide_y = y1 + 52
            if not self.detection_enabled:
                # Standby Banner
                cv2.rectangle(image, (x1 + 10, guide_y - 8), (x2 - 10, guide_y + 18), (55, 40, 20), -1)
                cv2.rectangle(image, (x1 + 10, guide_y - 8), (x2 - 10, guide_y + 18), (0, 190, 255), 1)
                cv2.putText(image, "DETECTION PAUSED", (x1 + 45, guide_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

                guide_lines = [
                    ("DYNAMIC MOTIONS:", (255, 255, 255)),
                    ("Draw CCW in air: START", (0, 255, 200)),
                    ("Draw CW in air: STOP/CLOSE", (120, 200, 255)),
                    ("", (0, 0, 0)),
                    ("Draw CCW circle with pointer", (220, 220, 220)),
                    ("finger in air to start!", (220, 220, 220)),
                    ("", (0, 0, 0)),
                    ("Key 'S': toggle | 'C': close", (160, 160, 180)),
                ]
                for idx, (gl, color) in enumerate(guide_lines):
                    if gl:
                        cv2.putText(image, gl, (x1 + 14, guide_y + 40 + idx * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            else:
                # Active Guide
                cv2.putText(image, "Dynamic: CCW Start | CW Stop/Close", (x1 + 12, guide_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1, cv2.LINE_AA)
                guide_lines = [
                    "1: Cat Shaka (Thumb+Pinky)",
                    "2: Monkey Point (1 Finger)",
                    "4: Monkey Thumbs Up",
                    "5: Cat 4-Finger Hold (4 Fingers)",
                    "7: J. Cole Open (5 Fingers)",
                    "8: Cool Cat Peace (2 Fingers)",
                ]
                for idx, gl in enumerate(guide_lines):
                    cv2.putText(image, gl, (x1 + 14, guide_y + 24 + idx * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

        # 5. Bottom Status / Progress Bar
        bar_y = y2 - 24
        bar_x1 = x1 + 10
        bar_x2 = x2 - 10
        bar_w = bar_x2 - bar_x1
        bar_h = 13

        if self.current_candidate is not None and self.current_candidate != self.active_meme_id and self.hold_count > 1 and self.detection_enabled:
            progress = min(1.0, self.hold_count / self.hold_frames)
            fill_w = int(bar_w * progress)

            cv2.rectangle(image, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 60), -1)
            cv2.rectangle(image, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h), (0, 200, 255), -1)
            cv2.rectangle(image, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (120, 120, 140), 1)

            cand_name = self.meme_labels.get(self.current_candidate, self.candidate_name)
            hold_msg = f"Holding: {cand_name} ({int(progress * 100)}%)"
            cv2.putText(image, hold_msg, (bar_x1, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 240, 255), 1, cv2.LINE_AA)
        else:
            if self.active_meme_id:
                cv2.putText(image, "[ACTIVE] Draw CW to Close Meme", (bar_x1 + 4, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 120), 1, cv2.LINE_AA)
            elif not self.detection_enabled:
                cv2.putText(image, "[STANDBY] Draw CCW to Start", (bar_x1 + 4, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, "[READY] Hold gesture to trigger", (bar_x1 + 4, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 220, 160), 1, cv2.LINE_AA)

        return image
