#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meme Action Handler:
Automatically opens memes, GIFs, or images based on classified hand gestures.
"""
import os
import time


class MemeTrigger:
    def __init__(self, memes_dir='memes', cooldown=2.5):
        """
        :param memes_dir: Path to directory containing meme files
        :param cooldown: Seconds to wait before allowing another meme trigger
        """
        self.memes_dir = memes_dir
        self.cooldown = cooldown
        self.last_trigger_time = 0
        self.last_gesture = None
        self.extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']

        # Ensure memes directory exists
        os.makedirs(self.memes_dir, exist_ok=True)

    def find_meme_file(self, gesture_name):
        """Looks for a matching image file in the memes folder."""
        clean_name = gesture_name.lower().replace(" ", "_")
        for ext in self.extensions:
            candidate = os.path.join(self.memes_dir, clean_name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def trigger(self, hand_sign, finger_gesture):
        """
        Checks gesture and opens matching meme if found and off cooldown.
        Prioritizes active dynamic finger motions over static hand signs.
        """
        now = time.time()
        if now - self.last_trigger_time < self.cooldown:
            return None

        # Prioritize finger motion if it's an active gesture (not 'Stop')
        gesture = finger_gesture if (finger_gesture and finger_gesture != "Stop") else hand_sign

        if not gesture:
            return None

        # Avoid spamming the exact same gesture repeatedly
        if gesture == self.last_gesture and (now - self.last_trigger_time < self.cooldown * 2):
            return None

        meme_file = self.find_meme_file(gesture)
        if meme_file:
            print(f"[MEME TRIGGER] 🎯 Gesture: '{gesture}' -> Opening: {meme_file}")
            try:
                os.startfile(os.path.abspath(meme_file))
                self.last_trigger_time = now
                self.last_gesture = gesture
                return meme_file
            except Exception as e:
                print(f"[MEME TRIGGER ERROR] Could not open {meme_file}: {e}")
                return None

        return None
