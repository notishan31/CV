"""
Video Colorization — frame-by-frame
Uses the Zhang et al. 2016 deep learning model via OpenCV's DNN module.

Usage:
    python colorize_video.py
    python colorize_video.py --input my_bw_video.mp4 --output colorized.mp4

The output video will have the same FPS and resolution as the input.
"""

import argparse
import os
import time
import numpy as np
import cv2

# ============================================================================
# STEP 1: Load the model and cluster centers (same as image version)
# ============================================================================

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT   = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
HULL_PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")


def load_colorization_model():
    """Load the Caffe network and inject ab cluster centers."""
    print("[INFO] Loading colorization model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Load 313 ab-gamut cluster centers and reshape to 1×1 conv kernel
    pts = np.load(HULL_PTS)                                    # (313, 2)
    pts = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)

    # Inject into network layers
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full((1, 313), 2.606, dtype=np.float32)
    ]
    print("[INFO] Model loaded successfully.")
    return net


def colorize_frame(net, frame_bgr):
    """
    Colorize a single BGR frame using the loaded network.

    Steps (same pipeline as the image script):
      1. Normalize pixels to [0, 1]
      2. Convert BGR → Lab
      3. Extract L channel, resize to 224×224, mean-center
      4. Forward pass → predicted ab at 56×56
      5. Resize ab back to original dimensions
      6. Merge original L + predicted ab → convert Lab → BGR
    """
    (H, W) = frame_bgr.shape[:2]

    # Normalize to [0, 1]
    frame_normalized = frame_bgr.astype(np.float32) / 255.0

    # BGR → Lab
    frame_lab = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2Lab)

    # Extract L channel, resize, mean-center
    L_channel = frame_lab[:, :, 0]
    L_resized = cv2.resize(L_channel, (224, 224))
    L_resized -= 50

    # Forward pass through the network
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab_predicted = net.forward()[0, :, :, :].transpose((1, 2, 0))  # (56, 56, 2)

    # Resize predicted ab back to original frame dimensions
    ab_resized = cv2.resize(ab_predicted, (W, H))

    # Merge original L with predicted ab
    L_original = frame_lab[:, :, 0:1]
    colorized_lab = np.concatenate([L_original, ab_resized], axis=2)

    # Lab → BGR, clip, convert to uint8
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_Lab2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (colorized_bgr * 255).astype(np.uint8)

    return colorized_bgr


# ============================================================================
# STEP 2: Open the input video with cv2.VideoCapture()
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Colorize a B&W video frame-by-frame")
    parser.add_argument("--input",  "-i", default="input_video.mp4", help="Path to input MP4 file")
    parser.add_argument("--output", "-o", default="colorized_video.mp4", help="Path to output MP4 file")
    args = parser.parse_args()

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.input}")
        return

    # Read video properties — we'll match these exactly in the output
    fps         = int(cap.get(cv2.CAP_PROP_FPS))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input video : {args.input}")
    print(f"[INFO] Resolution  : {width}x{height}")
    print(f"[INFO] FPS         : {fps}")
    print(f"[INFO] Total frames: {total_frames}")

    # ========================================================================
    # STEP 3: Set up cv2.VideoWriter() with same FPS and resolution
    # ========================================================================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Could not create output video: {args.output}")
        cap.release()
        return

    # Load the colorization model once (reused for every frame)
    net = load_colorization_model()

    # ========================================================================
    # STEP 4: Process every frame — colorize and write to output
    # ========================================================================
    frame_num = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_num += 1

        # Colorize this frame
        colorized = colorize_frame(net, frame)

        # Write the colorized frame to the output video
        out.write(colorized)

        # ── Progress indicator ─────────────────────────────────────────────
        elapsed = time.time() - start_time
        fps_actual = frame_num / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_num) / fps_actual if fps_actual > 0 else 0

        print(
            f"\r[PROGRESS] Frame {frame_num}/{total_frames} "
            f"({100 * frame_num / total_frames:.1f}%) | "
            f"Speed: {fps_actual:.1f} fps | "
            f"ETA: {int(eta // 60)}m {int(eta % 60)}s",
            end="", flush=True
        )

    # ========================================================================
    # Done — release resources
    # ========================================================================
    print()  # newline after the \r progress line
    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"[INFO] Colorization complete!")
    print(f"[INFO] Processed {frame_num} frames in {total_time:.1f}s ({frame_num / total_time:.1f} fps)")
    print(f"[INFO] Output saved to: {args.output}")


if __name__ == "__main__":
    main()
