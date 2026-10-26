# 🎭 In-Video Hand Gesture & Meme Controller

The project includes an interactive gesture controller that maps your exact real-time hand poses and finger movements to meme reference images, rendering them directly inside the video window.

---

## 📸 Meme Reference Images & Exact Gesture Mapping

| ID | Meme Reference | Target Hand Pose | Handedness | Exact Pose Details |
| :---: | :---: | :--- | :---: | :--- |
| **1** | ![Cat Shaka](1.jpg) | 🤙 **Cat Shaka** | **Left or Right** | **Thumb & Pinky extended outwards**, middle 3 fingers curled into palm ("Call-me" / Shaka sign). |
| **2** | ![Monkey Pointing](2.jpg) | ☝️ **1 Finger: Monkey Pointing** | **Left or Right** | **Index finger pointing straight UP** (1 finger count), thumb and other 3 fingers folded down. |
| **4** | ![Monkey Thumbs Up](4.jpg) | 👍 **Monkey Thumbs Up** | **Left or Right** | **Thumb pointing vertically UP**, all four other fingers clenched tight in a fist. |
| **5** | ![Cat 4-Finger Hold](5.png) | 🐾 **4 Fingers: Cat 4-Finger Hold** | **Left or Right** | **Index, Middle, Ring & Pinky extended UP** (4 finger count), thumb tucked/folded across palm. |
| **7** | ![J. Cole Open](7.jpg) | 🖐️ **5 Fingers: J. Cole Open Hand** | **Left or Right** | **All 5 fingers extended flat out** facing the camera (5 finger count / "Stop / Wait / Chill" pose). |
| **8** | ![Cool Cat Peace](8.png) | ✌️ **2 Fingers: Cool Cat Peace** | **Left or Right** | **Index & Middle fingers extended in a V / Peace pose** (2 finger count), ring and pinky folded down. |

---

## 🔄 Dynamic Hand Gestures: Start Detection & Close Active Meme

The system integrates real-time trajectory tracking of the index fingertip (`PointHistoryClassifier`) to enable dynamic air gestures for full lifecycle control:

| Dynamic Motion | Action | Description |
| :---: | :---: | :--- |
| **↺ Counter-Clockwise (CCW)** | **START Meme Detection** | Draw a counter-clockwise circle in the air with your pointer finger to start/enable meme detection. |
| **↻ Clockwise (CW)** | **STOP Detection & CLOSE Meme** | Draw a clockwise circle in the air with your pointer finger to immediately close/dismiss the active meme and pause detection. |

* **Keyboard Fallbacks**: Press `S` to toggle detection on/off, `C` to dismiss the active meme.

---

## 🧠 System Architecture & Reliability Highlights

1. **100% In-Video Rendering**:
   - Zero external desktop windows spawned.
   - Clean, frosted-glass card rendered directly in the top-right corner of the OpenCV video window with smooth letterboxing.

2. **Hand-Agnostic Left & Right Hand Support**:
   - Kinematic analysis checks relative joint angles and bone chain vectors (`MCP` -> `PIP` -> `DIP` -> `TIP`) rather than hardcoded 2D x/y coordinates.
   - Works seamlessly regardless of whether you use your left hand or right hand, or show palm or back of hand.

3. **Intuitive Finger Count Kinematics**:
   - Accurately discriminates 1, 2, 4, and 5 finger poses by evaluating multi-finger joint extension angles and bone projection vectors.
   - 4-Finger vs 5-Finger discrimination dynamically verifies thumb extension vs inward palm fold.

4. **Hold-to-Confirm Temporal Smoothing**:
   - Requires you to hold a pose consistently for **8 frames (~0.25 - 0.30 seconds)** to lock it in.
   - Moving your hand or transitioning between poses will **never** trigger accidental memes.
   - An interactive cyan progress bar (`Holding: <Gesture> (XX%)`) provides real-time feedback while holding.
