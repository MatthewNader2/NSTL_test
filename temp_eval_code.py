import blockSize
import cv2
import imgRect
import minDisparity
import numberOfDisparities
import pt1
import pt2
import roi1
import roi2
import winname

input_source = 'input.jpg'
cv2_getwindowimagerect_default = cv2.getWindowImageRect({winname})
cv2_getvaliddisparityroi_default = cv2.getValidDisparityROI({roi1}, {roi2}, {minDisparity}, {numberOfDisparities}, {blockSize})
cv2_clipline_default = cv2.clipLine({imgRect}, {pt1}, {pt2})