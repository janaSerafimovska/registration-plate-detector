import pytesseract
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class SobelANPR:
    def __init__(self, min_ar=2.5, max_ar=7):
        self.minAR = min_ar
        self.max_ar = max_ar

    @staticmethod
    def debug_image_show(caption, image, wait_key=False):
        cv2.imshow(caption, image)
        if wait_key:
            cv2.waitKey(0)

    @staticmethod
    def morphology_operation(gray, rect_kernel):
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kernel)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        return [black_hat, light]

    def locate_license_plate_candidates(self, gray, image, keep):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morphology = self.morphology_operation(gray, rect_kernel)
        morph = morphology[0]
        luminance = morphology[1]

        grad_x = cv2.Sobel(morph, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_x = np.absolute(grad_x)

        (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))

        grad_x = 255 * ((grad_x - minVal) / (maxVal - minVal))
        grad_x = grad_x.astype("uint8")

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=3)

        thresh = cv2.bitwise_and(thresh, thresh, mask=luminance)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:keep]

        ori_copy = image.copy()

        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(ori_copy, [box], -1, (0, 255, 0), 2)
            self.debug_image_show("Contours", ori_copy, True)

        return contours

    def locate_license_plate(self, gray, candidates):
        lp_cnt = None
        roi = None

        candidates = sorted(candidates, key=cv2.contourArea)

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if ar < self.minAR or ar > self.max_ar:
                continue
            lp_cnt = c
            license_plate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            self.debug_image_show("Licence Plate", license_plate, True)

            break

        return roi, lp_cnt

    @staticmethod
    def build_tesseract_options():
        options = "--oem 3"
        options += " --psm {}".format(6)
        options += " -c tessedit_char_whitelist={}".format("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        return options

    def find_and_ocr(self, image):
        lp_text = None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray, image, 5)
        (lp, lpCnt) = self.locate_license_plate(gray, candidates)

        if lp is not None:
            lp = cv2.bitwise_not(lp)
            lp = imutils.resize(lp, height=80, width=100)

            options = self.build_tesseract_options()
            lp_text = pytesseract.image_to_string(lp, lang='eng', config=options)

            converted_img = cv2.cvtColor(lp, cv2.COLOR_GRAY2BGR)
            plt.imshow(converted_img)
            plt.show()

        return lp_text, lpCnt
