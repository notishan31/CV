"""
Black & White Image Colorization
Using the Zhang et al. 2016 deep learning model via OpenCV's DNN module.

Usage:
    python colorize.py

Make sure these three files are present inside the 'models/' folder:
    - colorization_deploy_v2.prototxt      (network architecture)
    - colorization_release_v2.caffemodel   (pre-trained weights)
    - pts_in_hull.npy                      (313 ab-gamut cluster centers)
"""

import numpy as np
import cv2
import os

# ============================================================================
# STEP 1: Loading the model and the cluster centers
# ============================================================================
# The colorization model works in the CIE Lab color space. It takes the
# lightness (L) channel as input and predicts the two chrominance channels
# (a and b). The 313 cluster centers in pts_in_hull.npy represent a quantized
# version of the ab color space — the network outputs a probability
# distribution over these 313 bins, which is then converted to ab values.

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT   = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
HULL_PTS   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Load the Caffe network (architecture + weights)
print("[INFO] Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

# Load the 313 ab cluster centers and reshape them into a 1x1 convolution
# kernel shape: (2, 313, 1, 1). These are injected into the network so it
# can map its 313-class prediction back into actual ab values.
pts = np.load(HULL_PTS)                                  # shape: (313, 2)
pts = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)

# Inject the cluster centers into the "class8_ab" layer of the network.
# Also set a temperature factor (2.606) in the "conv8_313_rh" layer to
# control the sharpness of the predicted color distribution.
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
    np.full((1, 313), 2.606, dtype=np.float32)
]
print("[INFO] Model loaded successfully.")

# ============================================================================
# STEP 2: Reading the image and normalizing pixel values to [0, 1]
# ============================================================================
# cv2.imread() reads the image in BGR format with pixel values in [0, 255].
# We convert to float32 and divide by 255 so values lie in [0, 1], which is
# required before converting to the Lab color space with OpenCV.

INPUT_IMAGE = "test_input.jpg"   # <-- change this to your B&W image path

print(f"[INFO] Reading image: {INPUT_IMAGE}")
image_bgr = cv2.imread(INPUT_IMAGE)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {INPUT_IMAGE}")

# Store original dimensions for later resizing
(H, W) = image_bgr.shape[:2]

# Normalize pixel values from [0, 255] → [0, 1]
image_normalized = image_bgr.astype(np.float32) / 255.0

# ============================================================================
# STEP 3: Converting the image from BGR to LAB color space
# ============================================================================
# The Lab color space separates lightness (L) from color (a, b):
#   L  → lightness,   0 (black) to 100 (white)
#   a  → green-red axis
#   b  → blue-yellow axis
#
# This is ideal for colorization because we keep the original L (structure/
# brightness) untouched and only predict the missing a and b (color info).

image_lab = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2Lab)

# ============================================================================
# STEP 4: Extracting the L channel, resizing to 224x224, and running inference
# ============================================================================
# The network expects a single-channel 224×224 input — just the L channel,
# mean-centered by subtracting 50 (the midpoint of the L range [0, 100]).

# Extract the L channel from the Lab image
L_channel = image_lab[:, :, 0]

# Resize L to 224×224 (the model's expected spatial input size)
L_resized = cv2.resize(L_channel, (224, 224))

# Mean-center: subtract 50 so L is roughly in [-50, +50]
L_resized -= 50

# Wrap L into a 4D blob (batch=1, channels=1, h=224, w=224) and feed to network
net.setInput(cv2.dnn.blobFromImage(L_resized))

# Forward pass — the network predicts the ab chrominance channels
# Output shape: (1, 2, 56, 56) → 2 channels (a, b) at 56×56 resolution
ab_predicted = net.forward()[0, :, :, :]   # shape: (2, 56, 56)

# Rearrange from (channels, height, width) → (height, width, channels)
ab_predicted = ab_predicted.transpose((1, 2, 0))  # shape: (56, 56, 2)

print("[INFO] Network inference complete.")

# ============================================================================
# STEP 5: Resizing the predicted 'a' and 'b' channels back to original size
# ============================================================================
# The network outputs ab at 56×56 resolution. We resize these back to the
# original image dimensions (H, W) so they can be merged with the full-res
# L channel.

ab_resized = cv2.resize(ab_predicted, (W, H))  # shape: (H, W, 2)

# ============================================================================
# STEP 6: Merging original L with predicted ab, converting back to BGR, saving
# ============================================================================
# We take the original full-resolution L channel (preserving all brightness
# and structural detail) and concatenate it with the predicted ab channels
# to form a complete Lab image. Then we convert Lab → BGR for display/saving.

# Combine: original L (H, W, 1) + predicted ab (H, W, 2) → Lab (H, W, 3)
L_original = image_lab[:, :, 0:1]               # keep as (H, W, 1)
colorized_lab = np.concatenate([L_original, ab_resized], axis=2)

# Convert from Lab → BGR color space
colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_Lab2BGR)

# Clip to valid [0, 1] range and convert back to uint8 [0, 255]
colorized_bgr = np.clip(colorized_bgr, 0, 1)
colorized_bgr = (colorized_bgr * 255).astype(np.uint8)

# Save the colorized output
OUTPUT_IMAGE = "colorized_output.jpg"
cv2.imwrite(OUTPUT_IMAGE, colorized_bgr)
print(f"[INFO] Colorized image saved to: {OUTPUT_IMAGE}")

# ============================================================================
# Display: show original and colorized side by side
# ============================================================================
# Resize both to the same height for a clean side-by-side comparison
comparison = np.hstack([image_bgr, colorized_bgr])
cv2.imshow("Original (left)  |  Colorized (right)", comparison)
print("[INFO] Press any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
