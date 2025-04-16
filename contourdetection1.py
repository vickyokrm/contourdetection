import cv2
import os
import numpy as np


# === CONFIG ===
input_image_path = "tech_diagram.png"
template_dir = "templates"  # Folder with symbol template images
output_dir = "matched_symbols"
match_threshold = 0.7  # Similarity threshold

# === PREP ===
os.makedirs(output_dir, exist_ok=True)
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# === Load Templates ===
templates = {}
for filename in os.listdir(template_dir):
    if filename.endswith(".png"):
        template_img = cv2.imread(os.path.join(template_dir, filename), cv2.IMREAD_GRAYSCALE)
        templates[filename] = template_img

# === Template Matching ===
match_count = 0
for name, template in templates.items():
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= match_threshold)

    for pt in zip(*loc[::-1]):  # Switch x and y
        match_crop = image[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
        match_path = os.path.join(output_dir, f"{name}_match_{match_count+1}.png")
        cv2.imwrite(match_path, match_crop)
        match_count += 1

print(f"✅ Found {match_count} matching symbols. Saved to '{output_dir}'")