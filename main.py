import imutils
import cv2
from anprclass import SobelANPR


def cleanup_text(text):
    text = text.replace("°", "-").replace(")", "").replace("|", "").replace("l", "I").replace("&", "6")
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


if __name__ == '__main__':

    anpr = None
    imagePaths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg", "img7.jpg", "img8.jpg"]

    f = open("output.txt", "a")

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        height, width = image.shape[0], image.shape[1]
        if width > 1000:
            image = imutils.resize(image, width=400, height=600)

        image = cv2.bilateralFilter(image, 3, 105, 105)

        anpr = SobelANPR()
        (lpText, lpCnt) = anpr.find_and_ocr(image)
        if lpText is not None and lpCnt is not None:
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

            (x, y, w, h) = cv2.boundingRect(lpCnt)
            f.write(cleanup_text(lpText) + '\n')
