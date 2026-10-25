#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meme Action Handler:
Maps specific hand gestures and motions directly to meme images 1.jpg - 7.jpg.

Mapping:
  1 -> thumb + pinky finger (1.jpg)
  2 -> index finger pointing (2.jpg)
  3 -> index finger bent / thinking (3.jpg)
  4 -> thumb / thumbs up (4.jpg)
  5 -> closed fist #1 (5.jpg)
  6 -> closed fist #2 / clockwise or anticlockwise motion (6.jpg)
  7 -> open hand (7.jpg)
"""
import os
import math
import time
import cv2


class MemeTrigger:
    def __init__(self, memes_dir='memes', cooldown=2.0):
        self.memes_dir = memes_dir
        self.cooldown = cooldown
        self.last_trigger_time = 0
        self.last_gesture_id = None
        self.closed_toggle = 0

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

        # Cache thumbnails for on-screen PiP overlay
        self.thumbnails = {}
        for gid, path in self.meme_files.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self.thumbnails[gid] = cv2.resize(img, (160, 160))

        self.active_meme_id = None
        self.active_meme_label = ""
        self.display_expire_time = 0

    def detect_custom_gesture(self, landmark_list, motion_gesture):
        """
        Classifies 21 2D landmarks + motion into gestures 1-7.
        """
        if not landmark_list or len(landmark_list) < 21:
            return None, "None"

        # Dynamic motion gesture override (Clockwise or Counter Clockwise -> 6.jpg)
        if motion_gesture in ["Clockwise", "Counter Clockwise"]:
            return 6, f"Motion: {motion_gesture}"

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

        palm_size = dist(0, 9)
        if palm_size < 1:
            palm_size = 1.0

        # Finger 1: Index (5: MCP, 6: PIP, 7: DIP, 8: TIP)
        index_angle = get_angle(5, 6, 8)
        index_pip_projected = dist(6, 0) > dist(5, 0) * 1.2
        index_extended = (dist(8, 0) > dist(6, 0) * 1.15) and (index_angle > 135)
        # Bent: PIP is projected forward from MCP, but TIP is curled in (thinking gesture)
        index_bent = index_pip_projected and (index_angle < 130) and (not index_extended)
        index_folded = (not index_pip_projected) or (dist(8, 0) < dist(6, 0) * 1.05 and not index_bent)

        # Finger 2: Middle (9: MCP, 10: PIP, 11: DIP, 12: TIP)
        middle_extended = (dist(12, 0) > dist(10, 0) * 1.15) and (get_angle(9, 10, 12) > 135)
        middle_folded = dist(12, 0) < dist(10, 0) * 1.05

        # Finger 3: Ring (13: MCP, 14: PIP, 15: DIP, 16: TIP)
        ring_extended = (dist(16, 0) > dist(14, 0) * 1.15) and (get_angle(13, 14, 16) > 135)
        ring_folded = dist(16, 0) < dist(14, 0) * 1.05

        # Finger 4: Pinky (17: MCP, 18: PIP, 19: DIP, 20: TIP)
        pinky_extended = (dist(20, 0) > dist(18, 0) * 1.15) and (get_angle(17, 18, 20) > 135)
        pinky_folded = dist(20, 0) < dist(18, 0) * 1.05

        # Finger 0: Thumb (2: MCP, 3: IP, 4: TIP)
        thumb_extended = (dist(4, 17) > dist(3, 17) * 1.15) and (dist(4, 0) > dist(2, 0) * 1.15)
        thumb_folded = not thumb_extended

        # ---------------- GESTURE RULES ---------------- #
        # 7 -> open (All fingers extended)
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return 7, "Open Hand"

        # 1 -> thumb + pinky finger (Cat Shaka 🤙)
        if thumb_extended and pinky_extended and middle_folded and ring_folded and (not index_extended):
            return 1, "Thumb + Pinky"

        # 4 -> thumb (Thumbs up: only thumb extended)
        if thumb_extended and index_folded and middle_folded and ring_folded and pinky_folded:
            return 4, "Thumb Up"

        # 2 -> index (Pointing: only index extended)
        if index_extended and middle_folded and ring_folded and pinky_folded:
            return 2, "Index"

        # 3 -> index bend (Thinking monkey 🤔)
        if index_bent and middle_folded and ring_folded and pinky_folded:
            return 3, "Index Bend"

        # 5 & 6 -> closed (Closed fist: all fingers folded, alternates between 5 and 6)
        if index_folded and middle_folded and ring_folded and pinky_folded and thumb_folded:
            gid = 5 if (self.closed_toggle % 2 == 0) else 6
            return gid, f"Closed Fist ({gid})"

        return None, "Unknown"

    def trigger(self, landmark_list, motion_gesture):
        """
        Evaluates gesture and triggers the matching meme image.
        """
        gesture_id, gesture_name = self.detect_custom_gesture(landmark_list, motion_gesture)
        if gesture_id is None:
            return None

        now = time.time()
        # Cooldown check
        if now - self.last_trigger_time < self.cooldown:
            return gesture_id

        # Avoid repeated triggers without releasing
        if gesture_id == self.last_gesture_id and (now - self.last_trigger_time < self.cooldown * 1.5):
            return gesture_id

        # If it's a closed fist, toggle for next time so both 5 and 6 are triggered
        if gesture_id in (5, 6):
            self.closed_toggle += 1

        meme_path = self.meme_files.get(gesture_id)
        if meme_path and os.path.exists(meme_path):
            print(f"[MEME TRIGGER] 🎯 Detected {gesture_name} -> Opening {meme_path}")
            try:
                os.startfile(os.path.abspath(meme_path))
            except Exception as e:
                print(f"[MEME TRIGGER ERROR] Could not open {meme_path}: {e}")

        self.last_trigger_time = now
        self.last_gesture_id = gesture_id
        self.active_meme_id = gesture_id
        self.active_meme_label = f"Meme #{gesture_id}: {gesture_name}"
        self.display_expire_time = now + 3.0
        return gesture_id

    def draw_overlay(self, image):
        """
        Overlays the triggered meme preview in the top-right corner of the video frame.
        """
        if self.active_meme_id and time.time() < self.display_expire_time:
            thumb = self.thumbnails.get(self.active_meme_id)
            if thumb is not None:
                th_h, th_w, _ = thumb.shape
                im_h, im_w, _ = image.shape
                x_offset = im_w - th_w - 15
                y_offset = 15

                # Draw border & background rectangle
                cv2.rectangle(image, (x_offset - 4, y_offset - 4),
                              (x_offset + th_w + 4, y_offset + th_h + 30),
                              (0, 255, 0), -1)
                image[y_offset:y_offset + th_h, x_offset:x_offset + th_w] = thumb

                # Caption label
                cv2.putText(image, self.active_meme_label,
                            (x_offset - 2, y_offset + th_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return image
