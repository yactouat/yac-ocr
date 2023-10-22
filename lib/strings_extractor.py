import cv2
from PIL import Image
from IPython.display import display
import os

def binarize_img(grayscale_img, threshold=127, max=255):
    copy = grayscale_img.copy()
    _, binary_img = cv2.threshold(copy, threshold, max, cv2.THRESH_BINARY)
    return binary_img

def crop_img(img, x, y, width, height):
    copy = img.copy()
    cv2.rectangle(copy, (x, y), (x + width, y + height), (0, 255, 0), 3)
    return copy[y:y+height, x:x+width]

def extract_img_from_bounding_box(img, bounding_box):
    x, y, width, height = bounding_box
    return crop_img(img, x, y, width, height)

# we only retrieve the external contours, as these are the only ones we're interested in;
# we've also used `cv2.CHAIN_APPROX_SIMPLE` to compress the contouring, it reduces the number of data points to process
def get_contoured_img(inverted_binary_img):
    copy = inverted_binary_img.copy()
    contours, _ = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(copy, contours, -1, (0, 255, 0), 3)
    return contours, copy

def get_contours_bounding_boxes(contours, min_width=10, min_height=10):
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return [box for box in bounding_boxes if box[2] > min_width and box[3] > min_height]

def get_grayscale_img(path):
    return cv2.imread(path, 0)

def invert_colors(binary_img):
    copy = binary_img.copy()
    return 255 - copy

def load_raw_img(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def save_img(img, path):
    cv2.imwrite(path, img)

def strings_extractor(img_path, threshold_value, out_folder_path, threshold_max_value=255, min_contour_width=10, min_contour_height=10):
    grayscale_img = get_grayscale_img(img_path)
    binary_img = binarize_img(grayscale_img, threshold_value, threshold_max_value)
    inverted_img = invert_colors(binary_img)
    contours, _ = get_contoured_img(inverted_img)
    bounding_boxes = get_contours_bounding_boxes(contours, min_contour_width, min_contour_height)
    _, extension = os.path.splitext(img_path)
    for x, y, w, h in bounding_boxes:
        cropped_img = crop_img(grayscale_img, x, y, w, h)
        save_img(cropped_img, f'{out_folder_path}/{img_path.replace(extension, "")}-{x}-{y}-{w}-{h}{extension}')