# create_banner.py — Works with all Pillow versions (10.x compatible)
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------
size = (1600, 400)  # banner size
grad_from, grad_to = (255, 228, 236), (255, 255, 255)  # gradient soft pink to white
title = "AI Breast Cancer Predictor"
subtitle = "Developed by Amit Barik"
output_file = "breast_cancer_banner.png"

# -----------------------------------------------------
# CREATE GRADIENT BACKGROUND
# -----------------------------------------------------
img = Image.new("RGB", size, grad_from)
for x in range(size[0]):
    r = int(grad_from[0] + (grad_to[0] - grad_from[0]) * x / size[0])
    g = int(grad_from[1] + (grad_to[1] - grad_from[1]) * x / size[0])
    b = int(grad_from[2] + (grad_to[2] - grad_from[2]) * x / size[0])
    for y in range(size[1]):
        img.putpixel((x, y), (r, g, b))

# -----------------------------------------------------
# ADD TEXT (CENTERED)
# -----------------------------------------------------
draw = ImageDraw.Draw(img)
try:
    title_font = ImageFont.truetype("arial.ttf", 100)
    sub_font = ImageFont.truetype("arial.ttf", 42)
except:
    title_font = ImageFont.load_default()
    sub_font = ImageFont.load_default()

# --- Compute text bounding boxes (modern Pillow compatible) ---
def get_text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(text, font=font)

# Title position
w, h = get_text_size(draw, title, title_font)
draw.text(((size[0] - w) / 2, 120), title, fill="#db2777", font=title_font)

# Subtitle position
w2, h2 = get_text_size(draw, subtitle, sub_font)
draw.text(((size[0] - w2) / 2, 260), subtitle, fill="#6b7280", font=sub_font)

# -----------------------------------------------------
# SAVE IMAGE
# -----------------------------------------------------
img.save(output_file)
print(f"✅ Banner created successfully and saved as {output_file}")
