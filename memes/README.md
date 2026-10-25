# 🎭 In-Video Hand Gesture & Meme Controller

The system maps hand poses and circular finger movements directly to your meme images displayed **strictly inside the video window**:

| Meme File | Target Gesture | Description | Visual Preview |
| :--- | :--- | :--- | :--- |
| **`1.jpg`** | 🤙 **Thumb + Pinky** | Shaka gesture (Thumb & Pinky extended, middle three folded) | Cat Shaka |
| **`2.jpg`** | ☝️ **Index Pointer** | Pointing index finger straight up | Monkey Pointing |
| **`3.jpg`** | 🤔 **Index Bend** | Index finger hooked / bent at PIP (thinking pose) | Monkey Thinking |
| **`4.jpg`** | 👍 **Thumb Up** | Thumbs up (only thumb extended, four fingers curled in fist) | Monkey Thumbs Up |
| **`5.jpg`** | ✊ **Closed Fist** | Compact closed fist (all fingers folded) | Laughing Baby |
| **`6.jpg`** | 🔄 **Clockwise / Anticlockwise** | Drawing a circle in the air with index finger | LeBron Knuckle Bite |
| **`7.jpg`** | 🖐️ **Open Hand** | All 5 fingers extended flat out | J. Cole Open Hand |

---

### Key Improvements & Stability System:
- **100% In-Video Display**: No external OS image viewer windows. Memes appear directly inside a styled, frosted-glass card in the top-right corner of the camera feed.
- **Hold-to-Confirm Temporal Smoothing**: You must deliberately hold a gesture for ~0.35s (10 consecutive frames) to lock it in. Random hand movements or transitioning between poses will **never** trigger accidental memes.
- **Real-Time Hold Progress Meter**: A dynamic cyan progress bar displays `Holding: <Gesture> (XX%)` on the card while holding.
- **Active Meme Persistence**: The confirmed meme stays visible on screen with an active status badge until you intentionally switch to another gesture.
