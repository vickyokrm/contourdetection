import cv2
import os

# === CONFIG ===
input_image_path = "tech_diagram.png"  # Use your PNG image here
output_dir = "symbols_extracted"
min_area = 100  # Minimum size of shape to be considered a symbol
aspect_ratio_range = (0.8, 1.2)  # For square/rectangular symbols

# === PREP ===
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess image
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Gray Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# Extract and save symbol candidates
symbol_count = 0
all_symbols = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    aspect_ratio = w / float(h)

    # extract all possible symbols
    symbol_crop = image[y:y+h, x:x+w]
    symbol_path = os.path.join(output_dir, f"symbol_{all_symbols + 1}.png")
    cv2.imwrite(symbol_path, symbol_crop)
    all_symbols += 1

    # Check if the contour meets the criteria for square/rectangular symbols
    if area > min_area and aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
        symbol_crop = image[y:y+h, x:x+w]
        symbol_path = os.path.join(output_dir, f"symbol_uni{symbol_count + 1}.png")
        cv2.imwrite(symbol_path, symbol_crop)
        symbol_count += 1

print(f"✅ Extracted {symbol_count, all_symbols} potential symbols into '{output_dir}'")
