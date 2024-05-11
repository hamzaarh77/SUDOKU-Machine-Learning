import cv2
import numpy as np
import matplotlib.pyplot as plt
from reconnaisance2 import *
import torch
import scipy.stats
from torchvision import transforms

# Chargement du modèle entraîné
model = model.Net()
model.load_state_dict(torch.load('/home/etud/Bureau/sudoku_9x9/reconnaisance2/mnist_model.pth'))
model.eval()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def center_digit(cell):
    # Convertir en noir et blanc inversé pour la détection de contours
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cntr = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cntr)

        startx = (cell.shape[1] - w) // 2
        starty = (cell.shape[0] - h) // 2
        result = np.zeros_like(cell)
        result[starty:starty + h, startx:startx + w] = cell[y:y + h, x:x + w]
        return result
    return cell


def process_sudoku_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error loading image.")
        return

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_g, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_contour = contour

    if best_contour is None or len(best_contour) == 0:
        print("No suitable contour found.")
        return

    warped = four_point_transform(img, best_contour.reshape(-1, 2))

    warped_inverted = cv2.bitwise_not(warped)

    sudoku_cells = []
    rows = np.array_split(warped_inverted, 9, axis=0)
    for row in rows:
        cols = np.array_split(row, 9, axis=1)
        for col in cols:
            margin = int(0.2 * col.shape[0])  # 20%
            cropped = col[margin:col.shape[0] - margin, margin:col.shape[1] - margin]
            cell = cv2.resize(cropped, (28, 28))
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)  
            centered_cell = center_digit(cell)
            sudoku_cells.append(centered_cell)

    return sudoku_cells

def predict_digit(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)  
        _, predicted = torch.max(output, 1)  
    return predicted.item()

def numerisation_image(image):
    ret=[]

    sudoku_images = process_sudoku_image(image)
    if sudoku_images:
        for image in sudoku_images:
            if image.shape != (28, 28):
                print("Error: Image is not 28x28 pixels")
                continue  #
            image = np.array(image)
            p = np.histogram(image.reshape((784)), bins=256, range=(0, 256), density=True)[0]
            e=scipy.stats.entropy(p, base=2)

            if ( e == 0.0):
                ret.append(0)
            else:
                digit = predict_digit(image)
                ret.append(digit)
        return(ret)
        

