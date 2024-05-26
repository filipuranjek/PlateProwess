import numpy as np
import cv2
import imutils
import pytesseract
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas
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
        result_label.config(text=f"OCR Result: {text}", fg='green')

        binary_image = Image.fromarray(binary_image)
        binary_image = ImageTk.PhotoImage(binary_image)
        panel.config(image=binary_image)
        panel.image = binary_image
    else:
        result_label.config(text="No contour detected.", fg='red')

app = tk.Tk()
app.title("PlateProwess")
app.geometry("900x600")
app.configure(bg='#333333')

header_frame = Frame(app, bg='#444444', pady=10)
header_frame.pack(fill='x')

header_label = Label(header_frame, text="PlateProwess", font=('Arial', 24, 'bold'), bg='#444444', fg='white')
header_label.pack()

main_frame = Frame(app, bg='#333333', pady=20)
main_frame.pack(fill='both', expand=True)

left_frame = Frame(main_frame, bg='#333333')
left_frame.pack(side='left', fill='y', padx=20)

right_frame = Frame(main_frame, bg='#333333')
right_frame.pack(side='right', fill='both', expand=True, padx=20)

drop_area = Canvas(right_frame, width=400, height=300, bg='#444444', relief='ridge', bd=2)
drop_area.pack(pady=20)
drop_area.create_text(200, 150, text="Drop files here", fill="white", font=('Arial', 18))

button = Button(right_frame, text="Choose an image", command=select_image, font=('Arial', 14), bg='#555555', fg='white', relief='raised', bd=2)
button.pack()

result_label = Label(right_frame, text="OCR Result:", font=('Arial', 16), bg='#333333', fg='white')
result_label.pack(pady=20)

panel = Label(right_frame, bg='#333333')
panel.pack()

app.mainloop()
