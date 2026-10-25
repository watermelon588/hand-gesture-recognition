#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meme Action Handler:
Maps 7 hand gestures and motions directly to meme images displayed inside the video window.

Mapping:
  1 -> Thumb + Pinky finger (memes/1.jpg - Cat Shaka 🤙)
  2 -> Index finger pointing (memes/2.jpg - Monkey Pointing ☝️)
  3 -> Index finger bent / thinking (memes/3.jpg - Monkey Thinking 🤔)
  4 -> Thumb up (memes/4.jpg - Monkey Thumbs Up 👍)
  5 -> Closed fist (memes/5.jpg - Laughing Baby ✊)
  6 -> Clockwise / Counter Clockwise motion (memes/6.jpg - LeBron Knuckle Bite 🔄)
  7 -> Open hand (memes/7.jpg - J. Cole Open Hand 🖐️)

Stability Features:
  - In-Video Card Rendering: Strictly on the OpenCV window, no external popups, no stacking.
  - Hold-to-Confirm Smoothing: Must hold gesture stably for N frames (e.g. 10 frames ≈ 0.35s)
    to confirm, eliminating transient gesture flickering and accidental triggers.
  - On-Screen Hold Progress Bar: Real-time visual feedback of hold progress before switching.
  - Active Meme Persistence: Keeps the confirmed meme visible without rapid disappearance.
"""
import os
import math
import cv2
import numpy as np


class MemeTrigger:
    def __init__(self, memes_dir='memes', hold_frames=10):
        self.memes_dir = memes_dir
        self.hold_frames = hold_frames  # Consecutive frames required to lock in a gesture

        # Exact user file mapping
        self.meme_files = {
            1: os.path.join(memes_dir, "1.jpg"),
            2: os.path.join(memes_dir, "2.jpg"),
            3: os.path.join(memes_dir, "3.jpg"),
            4: os.path.join(memes_dir, "4.jpg"),
            5: os.path.join(memes_dir, "5.jpg"),
            6: os.path.join(memes_dir, "6.jpg"),
            7: os.path.join(memes_dir, "7.jpg"),
        }

        self.meme_labels = {
            1: "Thumb + Pinky",
            2: "Index Point",
            3: "Index Bend",
            4: "Thumb Up",
            5: "Closed Fist",
            6: "Circle Motion",
            7: "Open Hand",
        }

        # Cache pre-fitted card images (letterboxed to avoid distortion)
        self.card_size = (260, 290)  # (width, height)
        self.img_box_size = (244, 210)  # image area inside card
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

    def _load_and_cache_images(self):
        """Loads and prepares high quality letterboxed images for the in-video card."""
        target_w, target_h = self.img_box_size
        for gid, path in self.meme_files.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    h, w, _ = img.shape
                    # Scale preserving aspect ratio
                    scale = min(target_w / w, target_h / h)
                    nw = int(w * scale)
                    nh = int(h * scale)
                    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

                    # Create letterboxed canvas
                    canvas = np.full((target_h, target_w, 3), 22, dtype=np.uint8)
                    x_off = (target_w - nw) // 2
                    y_off = (target_h - nh) // 2
                    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
                    self.cached_images[gid] = canvas

    def detect_custom_gesture(self, landmark_list, motion_gesture):
        """
        Robust geometric classification of 21 MediaPipe hand landmarks + motion gesture.
        Returns: (gesture_id: int or None, gesture_name: str)
        """
        # Motion Gesture override: Clockwise or Counter Clockwise triggers 6.jpg!
        if motion_gesture in ["Clockwise", "Counter Clockwise"]:
            return 6, f"Motion: {motion_gesture}"

        if not landmark_list or len(landmark_list) < 21:
            return None, "None"

        def dist(i1, i2):
            return math.hypot(landmark_list[i1][0] - landmark_list[i2][0],
                              landmark_list[i1][1] - landmark_list[i2][1])

        def get_angle(i1, i2, i3):
            a, b, c = landmark_list[i1], landmark_list[i2], landmark_list[i3]
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

        # --- Finger Extension / Folding Analysis ---
        # 1. Index (5: MCP, 6: PIP, 7: DIP, 8: TIP)
        index_angle = get_angle(5, 6, 8)
        index_pip_proj = dist(6, 0) > dist(5, 0) * 1.15
        index_extended = (dist(8, 0) > dist(6, 0) * 1.18) and (index_angle > 140) and (dist(8, 5) > 0.8 * palm_size)
        # Bent: PIP projected forward, tip curled inward/hooked
        index_bent = index_pip_proj and (index_angle < 125) and (dist(8, 0) < dist(6, 0) * 1.05) and (not index_extended)
        index_folded = (not index_bent) and (dist(8, 0) < dist(6, 0) * 1.08 or dist(8, 5) < 0.65 * palm_size)

        # 2. Middle (9: MCP, 10: PIP, 11: DIP, 12: TIP)
        middle_angle = get_angle(9, 10, 12)
        middle_extended = (dist(12, 0) > dist(10, 0) * 1.18) and (middle_angle > 135) and (dist(12, 9) > 0.8 * palm_size)
        middle_folded = dist(12, 0) < dist(10, 0) * 1.08 or dist(12, 9) < 0.65 * palm_size

        # 3. Ring (13: MCP, 14: PIP, 15: DIP, 16: TIP)
        ring_angle = get_angle(13, 14, 16)
        ring_extended = (dist(16, 0) > dist(14, 0) * 1.18) and (ring_angle > 135) and (dist(16, 13) > 0.8 * palm_size)
        ring_folded = dist(16, 0) < dist(14, 0) * 1.08 or dist(16, 13) < 0.65 * palm_size

        # 4. Pinky (17: MCP, 18: PIP, 19: DIP, 20: TIP)
        pinky_angle = get_angle(17, 18, 20)
        pinky_extended = (dist(20, 0) > dist(18, 0) * 1.18) and (pinky_angle > 135) and (dist(20, 17) > 0.8 * palm_size)
        pinky_folded = dist(20, 0) < dist(18, 0) * 1.08 or dist(20, 17) < 0.65 * palm_size

        # 5. Thumb (0: Wrist, 2: MCP, 3: IP, 4: TIP)
        thumb_extended = (dist(4, 17) > dist(2, 17) * 1.2) and (dist(4, 0) > dist(2, 0) * 1.15) and (dist(4, 9) > 0.75 * palm_size)
        thumb_folded = not thumb_extended

        # --- Gesture Decision Rules ---
        # 7 -> Open Hand: 4 main fingers extended
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return 7, "Open Hand"

        # 1 -> Thumb + Pinky: Thumb and pinky extended, middle three folded
        if thumb_extended and pinky_extended and middle_folded and ring_folded and (not index_extended) and (not index_bent):
            return 1, "Thumb + Pinky"

        # 4 -> Thumb Up: Only thumb extended, all other 4 folded
        if thumb_extended and (not pinky_extended) and index_folded and middle_folded and ring_folded and pinky_folded:
            return 4, "Thumb Up"

        # 2 -> Index Pointer: Only index extended straight
        if index_extended and (not index_bent) and thumb_folded and middle_folded and ring_folded and pinky_folded:
            return 2, "Index"

        # 3 -> Index Bend: Index hooked/bent, other 4 fingers folded
        if index_bent and middle_folded and ring_folded and pinky_folded and thumb_folded:
            return 3, "Index Bend"

        # 5 -> Closed Fist: All 5 fingers folded tight
        if thumb_folded and index_folded and middle_folded and ring_folded and pinky_folded and (not index_bent):
            return 5, "Closed Fist"

        return None, "Unknown"

    def trigger(self, landmark_list, motion_gesture):
        """
        Feed new frame landmark data and motion gesture.
        Uses temporal hold smoothing to confirm gestures stably without flickering.
        """
        raw_id, raw_name = self.detect_custom_gesture(landmark_list, motion_gesture)

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
                if self.active_meme_id != self.current_candidate:
                    self.active_meme_id = self.current_candidate
                    self.active_meme_label = self.meme_labels.get(self.active_meme_id, self.candidate_name)
                    self.flash_frames = 8  # trigger visual pulse on the card
                    print(f"[MEME CONFIRMED] Switched to Meme #{self.active_meme_id}: {self.active_meme_label}")
        else:
            # Gradually decay hold count if hand is lost or in between gestures
            self.hold_count = max(0, self.hold_count - 1)
            if self.hold_count == 0:
                self.current_candidate = None
                self.candidate_name = ""

        return self.active_meme_id

    def update_no_hand(self):
        """Called when no hand is in the frame to smoothly reset candidate hold state."""
        self.hold_count = max(0, self.hold_count - 2)
        if self.hold_count == 0:
            self.current_candidate = None
            self.candidate_name = ""

    def draw_overlay(self, image):
        """
        Renders the active meme card and hold progress directly on the OpenCV frame.
        """
        im_h, im_w, _ = image.shape
        card_w, card_h = self.card_size

        # Adapt card size if camera resolution is smaller
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Active Meme Image Display
        if self.active_meme_id and self.active_meme_id in self.cached_images:
            cached_img = self.cached_images[self.active_meme_id]
            # Resize cached image if card is scaled down
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
            guide_y = y1 + 65
            cv2.putText(image, "Hold gesture to trigger:", (x1 + 14, guide_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
            guide_lines = [
                "1: 🤙 Thumb+Pinky",
                "2: ☝️ Index Point",
                "3: 🤔 Index Bend",
                "4: 👍 Thumb Up",
                "5: ✊ Closed Fist",
                "6: 🔄 Circle Motion",
                "7: 🖐️ Open Hand",
            ]
            for idx, gl in enumerate(guide_lines):
                cv2.putText(image, gl, (x1 + 20, guide_y + 24 + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # 5. Hold-to-Confirm Progress Bar
        bar_y = y2 - 26
        bar_x1 = x1 + 10
        bar_x2 = x2 - 10
        bar_w = bar_x2 - bar_x1
        bar_h = 14

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 255), 1, cv2.LINE_AA)
        else:
            # Active status badge
            if self.active_meme_id:
                cv2.putText(image, "● ACTIVE ON SCREEN", (bar_x1 + 4, bar_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 120), 1, cv2.LINE_AA)

        return image
