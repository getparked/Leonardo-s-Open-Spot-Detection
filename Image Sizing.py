import cv2 as cv
import numpy as np


PL = cv.imread('./Pics/PLPi4.jpg')
PLResized = cv.resize(PL, (1500, 600), interpolation = cv.INTER_AREA)
PLOG = PLResized.copy()

cv.line(PLResized, (978, 37), (995, 36), (255, 0, 0), 2) #Blue
cv.line(PLResized, (988, 56), (1006, 55), (0, 0, 255), 2) #Red


# Sup izq --- Sup derf
# Inf izq --- Inf der

#1500
#600

cv.imshow("Funciona?", PLResized)
cv.imshow("OG", PLOG)
cv.waitKey(0)
cv.destroyAllWindows()
