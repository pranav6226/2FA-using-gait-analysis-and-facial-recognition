from PIL import Image
bg = Image.open("Image1.png")
for i in range(2,10):
    overlay = Image.open(f"Image{i}.png")
    bg = bg.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(backround, overlay, 0.3)
    bg = new_img
    new_img.save("final.png","PNG")