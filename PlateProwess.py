import numpy as np
import cv2
import imutils
import pytesseract
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 150, 370)
    return gray, edged

def find_contours(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def mask_number_plate(gray, NumberPlateCnt):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [NumberPlateCnt], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(gray, gray, mask=mask)

def convert_to_binary(masked_image):
    _, binary = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def ocr_text(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(image, config=custom_config, lang='hrv').strip()

def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    gray, edged = preprocess_image(file_path)
    plateCnt = find_contours(edged)
    if plateCnt is not None:
        masked_image = mask_number_plate(gray, plateCnt)
        binary_image = convert_to_binary(masked_image)
        text = ocr_text(binary_image)
        result_label.config(text=f"OCR Result: {text}")

        binary_image = Image.fromarray(binary_image)
        binary_image = ImageTk.PhotoImage(binary_image)
        panel.config(image=binary_image)
        panel.image = binary_image
    else:
        result_label.config(text="No contour detected.")

app = tk.Tk()
app.title("License Plate OCR")
app.geometry("800x600")

button = Button(app, text="Select Image", command=select_image)
button.pack()

result_label = Label(app, text="OCR Result:")
result_label.pack()

panel = Label(app)
panel.pack()

app.mainloop()