import numpy as np
import cv2
import imutils
import pytesseract

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

def main():
    gray, edged = preprocess_image("./data/car8.jpeg")
    plateCnt = find_contours(edged)
    if plateCnt is not None:
        masked_image = mask_number_plate(gray, plateCnt)
        binary_image = convert_to_binary(masked_image)
        text = ocr_text(binary_image)
        print(text)
        cv2.imshow("Binary Plate", binary_image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        print("No contour detected.")

if __name__ == "__main__":
    main()
