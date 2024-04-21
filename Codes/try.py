import numpy as np
import cv2

def generate_segmentation_mask(segmentation_annotation):
  """Generates a segmentation mask from a list of lists of tuples of x, y coordinates.

  Args:
    segmentation_annotation: A list of lists of tuples of x, y coordinates.

  Returns:
    A segmentation mask.
  """

  mask = np.zeros((10, 12), dtype=np.uint8)
  for polygon in segmentation_annotation:
    polygon = np.array(polygon)
    cv2.fillPoly(mask, [polygon], 1)
  return mask

if __name__ == '__main__':
    s = [[(2, 3), (7, 9), (4, 1), (1, 6)]]
    m = generate_segmentation_mask(s)
    print(s)
    print("Sairammmmmmmmmmmmmmmm")
    print(m)
