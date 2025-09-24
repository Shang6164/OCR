import numpy as np
import cv2

def calculate_iou_for_bboxes(box1, box2):
    """
    Tính toán IoU giữa hai bounding box.
    Định dạng box: [x_min, y_min, x_max, y_max].
    """
    x_min_intersect = max(box1[0], box2[0])
    y_min_intersect = max(box1[1], box2[1])
    x_max_intersect = min(box1[2], box2[2])
    y_max_intersect = min(box1[3], box2[3])

    intersect_width = max(0, x_max_intersect - x_min_intersect)
    intersect_height = max(0, y_max_intersect - y_min_intersect)
    intersection_area = intersect_width * intersect_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou_result = intersection_area / (union_area + 1e-6)
    return iou_result

def order_corner_points(corners):
    """
    Sắp xếp 4 điểm góc của thẻ theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái.
    Input: mảng NumPy 2D của 4 điểm.
    Output: mảng NumPy chứa các điểm đã sắp xếp.
    """
    ordered_points = np.zeros((4, 2), dtype="float32")
    
    sum_coords = corners.sum(axis=1)
    ordered_points[0] = corners[np.argmin(sum_coords)]  # top-left có tổng nhỏ nhất
    ordered_points[2] = corners[np.argmax(sum_coords)]  # bottom-right có tổng lớn nhất
    
    diff_coords = np.diff(corners, axis=1)
    ordered_points[1] = corners[np.argmin(diff_coords)] # top-right có hiệu nhỏ nhất
    ordered_points[3] = corners[np.argmax(diff_coords)] # bottom-left có hiệu lớn nhất
    
    return ordered_points