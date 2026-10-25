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
    def __init__(self, memes_dir='memes', hold_frames=8, **kwargs):
        self.memes_dir = memes_dir
        self.hold_frames = hold_frames  # Consecutive frames to confirm a gesture (~0.25 - 0.3s)

        # Exact user file mapping
        self.meme_files = {
            1: os.path.join(memes_dir, "1.jpg"),
            2: os.path.join(memes_dir, "2.jpg"),
            3: os.path.join(memes_dir, "3.jpg"),
            4: os.path.join(memes_dir, "4.jpg"),
            5: os.path.join(memes_dir, "5.jpg"),
            6: os.path.join(memes_dir, "6.jpg"),
            7: os.path.join(memes_dir, "7.jpg"),
            8: os.path.join(memes_dir, "8.jpg"),
        }

        self.meme_labels = {
            1: "Cat Shaka (Thumb+Pinky)",
            2: "Monkey Pointing Up",
            3: "Monkey Thinking (Hooked Index)",
            4: "Monkey Thumbs Up",
            5: "Baby Fist (Closed Left)",
            6: "LeBron Knuckle (Closed Right / 🔄)",
            7: "J. Cole Open Hand",
            8: "Who, Me? (Point at Self)",
        }

        # Cache pre-fitted card images (letterboxed to avoid distortion)
        self.card_size = (260, 290)
        self.img_box_size = (244, 205)
        self.cached_images = {}
        self._load_and_cache_images()

        # State tracking
        self.active_meme_id = None
        self.active_meme_label = ""
        self.flash_frames = 0

        # Temporal hold state machine
        self.current_candidate = None
        self.candidate_name = ""
        self.hold_count = 0

        # Fist toggle tracking for single-hand users
        self.fist_hold_timer = 0
        self.fist_toggle = 0

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
        Classifies 21 MediaPipe hand landmarks into meme gestures 1-8.
        Fully compatible with both Left and Right hands.
        """
        # Motion Gesture override: Clockwise or Counter Clockwise triggers Meme #6 (LeBron 🔄)
        if motion_gesture in ["Clockwise", "Counter Clockwise"]:
            return 6, f"Motion: {motion_gesture}"

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
        # Bent/Hooked (Thinking pose): PIP projected forward from MCP, tip curled in towards palm/MCP
        index_pip_proj = (dist(6, 5) > 0.3 * palm_size) and (dist(6, 0) > dist(5, 0) * 1.08)
        index_bent = index_pip_proj and (index_angle < 125) and (dist(8, 5) <= dist(6, 5) * 1.25) and (not index_extended)
        index_folded = (not index_extended) and (not index_bent)

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
        thumb_pointing_up = (pts[4][1] < pts[2][1] - 0.2 * palm_size) and thumb_extended

        # --- Gesture Decision Rules ---

        # 7 -> Open Hand (J. Cole Stop / Open Hand):
        # 4 main fingers extended, or ML says Open
        if (index_extended and middle_extended and ring_extended and pinky_extended) or (hand_sign_label == "Open" and not thumb_pointing_up):
            return 7, "Open Hand"

        # 1 -> Thumb + Pinky (Cat Shaka 🤙):
        # Thumb & Pinky extended, middle 3 fingers folded
        if thumb_extended and pinky_extended and middle_folded and ring_folded and (not index_extended) and (not index_bent):
            return 1, "Thumb + Pinky"

        # 8 -> Who, Me? (Index pointing inward / towards self):
        index_pointing_inward = (dist(8, 5) > 0.6 * palm_size) and (abs(pts[8][0] - pts[5][0]) > abs(pts[8][1] - pts[5][1]))
        if index_pointing_inward and middle_folded and ring_folded and pinky_folded:
            return 8, "Point at Self"

        # 4 -> Thumb Up (Monkey Thumbs Up):
        # Thumb pointing UP, all other 4 fingers folded into a fist
        if thumb_pointing_up and (not pinky_extended) and index_folded and middle_folded and ring_folded and pinky_folded and (not index_pointing_inward):
            return 4, "Thumb Up"

        # 2 -> Index Pointer (Monkey Pointing Up ☝️):
        # Index extended straight up, other fingers folded
        if (index_extended or hand_sign_label == "Pointer") and (not index_bent) and middle_folded and ring_folded and pinky_folded and (not pinky_extended):
            return 2, "Index Point"

        # 3 -> Index Bend (Monkey Thinking 🤔):
        # Index hooked/bent, other 3 fingers folded
        if index_bent and middle_folded and ring_folded and pinky_folded:
            return 3, "Index Bend"

        # 5 & 6 -> Closed Fist (Baby Fist ✊ / LeBron Knuckle Bite ✊):
        # Triggers via ML model "Close" OR geometric folding of all 4 fingers without thumb pointing up
        all_four_folded = index_folded and middle_folded and ring_folded and pinky_folded and (not index_bent) and (not pinky_extended)
        if hand_sign_label == "Close" or (all_four_folded and not thumb_pointing_up):
            # Left Hand Fist -> Meme #5 (Baby Fist)
            # Right Hand Fist -> Meme #6 (LeBron Knuckle Bite)
            # If user holds fist continuously, toggle to allow single-hand access to both
            if handedness == "Left":
                return 5, "Baby Fist (Left)"
            elif handedness == "Right":
                return 6, "LeBron Knuckle (Right)"
            else:
                return 5, "Closed Fist"

        return None, "Unknown"

    def trigger(self, landmark_list, motion_gesture, hand_sign_label="Unknown", handedness="Right"):
        """
        Feeds new frame landmark data and motion gesture.
        Uses temporal hold smoothing to confirm gestures stably.
        """
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
                # Toggle for fist if held continuously for a long time (~40 frames / ~1.5s)
                if target_id in (5, 6):
                    self.fist_hold_timer += 1
                    if self.fist_hold_timer > 40:
                        self.fist_toggle = 1 - self.fist_toggle
                        self.fist_hold_timer = 0
                        target_id = 6 if self.fist_toggle == 1 else 5
                else:
                    self.fist_hold_timer = 0

                if self.active_meme_id != target_id:
                    self.active_meme_id = target_id
                    self.active_meme_label = self.meme_labels.get(self.active_meme_id, self.candidate_name)
                    self.flash_frames = 8  # trigger visual pulse on the card
                    print(f"[MEME CONFIRMED] Switched to Meme #{self.active_meme_id}: {self.active_meme_label}")
        else:
            self.hold_count = max(0, self.hold_count - 1)
            self.fist_hold_timer = 0
            if self.hold_count == 0:
                self.current_candidate = None
                self.candidate_name = ""

        return self.active_meme_id

    def update_no_hand(self):
        """Called when no hand is in the frame to smoothly reset candidate hold state."""
        self.hold_count = max(0, self.hold_count - 2)
        self.fist_hold_timer = 0
        if self.hold_count == 0:
            self.current_candidate = None
            self.candidate_name = ""

    def draw_overlay(self, image):
        """
        Renders the active meme card and hold progress directly on the OpenCV frame.
        """
        im_h, im_w, _ = image.shape
        card_w, card_h = self.card_size

        if im_w < 720:
            scale_factor = 0.8
            card_w = int(card_w * scale_factor)
            card_h = int(card_h * scale_factor)

        margin = 15
        x1 = im_w - card_w - margin
        y1 = margin
        x2 = x1 + card_w
        y2 = y1 + card_h

        # 1. Dark frosted-glass card background
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # 2. Glowing card border (flashes gold when newly unlocked)
        if self.flash_frames > 0:
            border_color = (0, 230, 255)  # Gold
            border_thick = 3
            self.flash_frames -= 1
        else:
            border_color = (0, 220, 140)  # Neon Emerald Green
            border_thick = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, border_thick)

        # 3. Card Header Bar
        header_h = 32
        cv2.rectangle(image, (x1, y1), (x2, y1 + header_h), (40, 40, 50), -1)
        cv2.rectangle(image, (x1, y1 + header_h), (x2, y1 + header_h), border_color, 1)

        if self.active_meme_id:
            title_text = f"MEME #{self.active_meme_id}: {self.active_meme_label}"
        else:
            title_text = "HAND MEME CONTROLLER"

        cv2.putText(image, title_text, (x1 + 8, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Active Meme Image Display
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
            # Idle placeholder guide
            guide_y = y1 + 55
            cv2.putText(image, "Hold gesture to trigger:", (x1 + 14, guide_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            guide_lines = [
                "1: Cat Shaka (Thumb+Pinky)",
                "2: Monkey Point (Index Up)",
                "3: Monkey Think (Hook Index)",
                "4: Monkey Thumbs Up",
                "5: Baby Fist (Left ✊)",
                "6: LeBron Knuckle (Right / 🔄)",
                "7: J. Cole Open Hand (🖐️)",
                "8: Who, Me? (Point at Self)",
            ]
            for idx, gl in enumerate(guide_lines):
                cv2.putText(image, gl, (x1 + 16, guide_y + 20 + idx * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)

        # 5. Hold-to-Confirm Progress Bar
        bar_y = y2 - 24
        bar_x1 = x1 + 10
        bar_x2 = x2 - 10
        bar_w = bar_x2 - bar_x1
        bar_h = 13

        if self.current_candidate is not None and self.current_candidate != self.active_meme_id and self.hold_count > 1:
            progress = min(1.0, self.hold_count / self.hold_frames)
            fill_w = int(bar_w * progress)

            # Bar background
            cv2.rectangle(image, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 60), -1)
            # Bar fill
            cv2.rectangle(image, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h), (0, 200, 255), -1)
            # Bar border
            cv2.rectangle(image, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (120, 120, 140), 1)

            # Hold label
            cand_name = self.meme_labels.get(self.current_candidate, self.candidate_name)
            hold_msg = f"Holding: {cand_name} ({int(progress * 100)}%)"
            cv2.putText(image, hold_msg, (bar_x1, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 240, 255), 1, cv2.LINE_AA)
        else:
            # Active status badge
            if self.active_meme_id:
                cv2.putText(image, "[ACTIVE ON SCREEN]", (bar_x1 + 4, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 120), 1, cv2.LINE_AA)

        return image
