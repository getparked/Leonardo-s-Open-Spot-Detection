"""
Program Name: Spot Detection from Pi Image, V-1
Author: Get Parked
Creation Date: 15/09/2023
Description: This program reads an image taken from the Pi, ideally a parking lot and will perform open stall detection by filtering by,
stall colour comparison, rectangularity, minimum area and percentage of stalls' ground pixels.

It then will gather this information in an array and transmit the data to The Things Network.
"""

# Import Libraries
import cv2 as cv
import numpy as np
import serial

print("Starting")

# Variable Initialization
OpenSpotsCtr = 0
ParkingSpotsArray = [1] * 184
ser = serial.Serial('/dev/ttyS0', 115200, timeout=3)

H_BFR = 1
S_BFR = 5
V_BFR = 18

B_h_max = 0
B_s_max = 0
B_v_max = 0

P_h_max = 0
P_s_max = 0
P_v_max = 0

R_h_max = 0
R_s_max = 0
R_v_max = 0

G_h_max = 0
G_s_max = 0
G_v_max = 0

O_h_max = 0
O_s_max = 0
O_v_max = 0

Y_h_max = 0
Y_s_max = 0
Y_v_max = 0



# Function Definitions

def process_region(img, vertices, lower_gray, upper_gray, half_side, tolerance, min_area):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_gray, upper_gray)

    # Create Poligonal mask
    poly_mask = np.zeros_like(mask)
    vertices_array = np.array([vertices], dtype=np.int32)
    cv.fillPoly(poly_mask, vertices_array, 255)

    # Apply mask to original image
    masked = cv.bitwise_and(mask, mask, mask=poly_mask)

    height, width = masked.shape
    new_mask = np.zeros_like(masked)

    for y in range(half_side, height - half_side):
        for x in range(half_side, width - half_side):
            if masked[y, x] == 255:
                square = masked[y-half_side:y+half_side, x-half_side:x+half_side]
                if np.sum(square) / 255 >= (2 * half_side * 2 * half_side * tolerance):
                    new_mask[y-half_side:y+half_side, x-half_side:x+half_side] = 255

    contours, _ = cv.findContours(new_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(new_mask)
    for contour in contours:
        if cv.contourArea(contour) >= min_area:
            cv.drawContours(final_mask, [contour], -1, (255), thickness=cv.FILLED)

    result = img.copy()
    result[final_mask > 0] = [0, 255, 0]
    return result


def get_hsv_range_from_line(image, start_point, end_point, h_buffer = H_BFR, s_buffer = S_BFR, v_buffer = V_BFR):
    # Create a blank image the same size as the original
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw a line on the mask
    cv.line(mask, start_point, end_point, 255, 2)
    
    # Find the Pixels with the drawn line
    y_coords, x_coords = np.where(mask == 255)
    
    # Store HSV values of pixels detected
    h_values, s_values, v_values = [], [], []
    
    # Convert image from BGR to HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Extract the HSV values
    for x, y in zip(x_coords, y_coords):
        h, s, v = hsv_image[y, x]
        h_values.append(h)
        s_values.append(s)
        v_values.append(v)
        
    # Calculate ranges of values
    h_min = max(0, min(h_values) - h_buffer)
    h_max = min(180, max(h_values) + h_buffer)
    s_min = max(0, min(s_values) - s_buffer)
    s_max = min(255, max(s_values) + s_buffer)
    v_min = max(0, min(v_values) - v_buffer)
    v_max = min(255, max(v_values) + v_buffer)
    
    return h_min, h_max, s_min, s_max, v_min, v_max

def TransformToDecimal(bits):
    decimal_str = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_value = int(''.join(str(bit) for bit in byte), 2)
        decimal_str += str(byte_value) + ", "
    decimal_str += ';'   
    return decimal_str


# Read Image, resize it and copy
PL = cv.imread('./Pics/PLPi1.jpg')
PLResized = cv.resize(PL, (1500, 600), interpolation = cv.INTER_AREA)
PLOG = PLResized.copy()

# Define lines for HSV values extraction
StartPointB = (2150, 620)
EndPointB = (2165, 710)

StartPointP = (2150, 620)
EndPointP = (2165, 710)

StartPointR = (2150, 620)
EndPointR = (2165, 710)

StartPointG = (2150, 620)
EndPointG = (2165, 710)

StartPointO = (2150, 620)
EndPointO = (2165, 710)

StartPointY = (2150, 620)
EndPointY = (2165, 710)

# Get HSV Ranges for each row
B_h_min, B_h_max, B_s_min, B_s_max, B_v_min, B_v_max = get_hsv_range_from_line(PL, StartPointB, EndPointB)
P_h_min, P_h_max, P_s_min, P_s_max, P_v_min, P_v_max = get_hsv_range_from_line(PL, StartPointP, EndPointP)
R_h_min, R_h_max, R_s_min, R_s_max, R_v_min, R_v_max = get_hsv_range_from_line(PL, StartPointR, EndPointR)
G_h_min, G_h_max, G_s_min, G_s_max, G_v_min, G_v_max = get_hsv_range_from_line(PL, StartPointG, EndPointG)
O_h_min, O_h_max, O_s_min, O_s_max, O_v_min, O_v_max = get_hsv_range_from_line(PL, StartPointO, EndPointO)
Y_h_min, Y_h_max, Y_s_min, Y_s_max, Y_v_min, Y_v_max = get_hsv_range_from_line(PL, StartPointY, EndPointY)

# Exclusive Buffers
# Blue
B_h_max = B_h_max + 30
B_s_max = B_s_max + 0
B_v_max = B_v_max + 0

# Purple
P_h_max = P_h_max + 30
P_s_max = P_s_max + 0
P_v_max = P_v_max + 0

# Red
R_h_max = R_h_max + 30
R_s_max = R_s_max + 0
R_v_max = R_v_max + 0

# Green
G_h_max = G_h_max + 30
G_s_max = G_s_max + 0
G_v_max = G_v_max + 0

# Orange
O_h_max = O_h_max + 30
O_s_max = O_s_max + 0
O_v_max = O_v_max + 0

# Yellow
Y_h_max = Y_h_max + 30
Y_s_max = Y_s_max + 0
Y_v_max = Y_v_max + 0


# Define regions of interest and adjust parameters
regions = [

    # Blue Row 1
    {
        'vertices': [(80, 380), (780, 438), (780, 550), (0, 475)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 12,
        'tolerance': 0.9,
        'min_area': 2500
    },

    # Blue Row 2
    {
        'vertices': [(780, 438), (1200, 438), (1250, 570), (780, 570)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 12,
        'tolerance': 0.9,
        'min_area': 2500
    },

    # Blue Row 3
    {
        'vertices': [(1200, 438), (1500, 420), (1500, 530), (1250, 570)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 12,
        'tolerance': 0.9,
        'min_area': 2500
    },

    # Purple Row 1
    {
        'vertices': [(230, 225), (760, 260), (770, 335), (200, 290)],
        'lower_gray': np.array([0, 0, 50]), 
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]), 
        'half_side': 10,
        'tolerance': 1.1,
        'min_area': 1050
    },

    # Purple Row 2
    {
        'vertices': [(760, 260), (1360, 260), (1410, 335), (770, 335)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 10,
        'tolerance': 1.1,
        'min_area': 1050
    },
    
    # Red Row 1
    {
        'vertices': [(350, 175), (760, 200), (765, 240), (330, 215)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([R_h_max, R_s_max, R_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 800
    },
    
    # Red Row 2
    {
        'vertices': [(765, 200), (1230, 185), (1270, 235), (770, 240)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([R_h_max, R_s_max, R_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 800
    },

    # Green Row 1
    {
        'vertices': [(380, 100), (750, 115), (755, 155), (365, 135)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([G_h_max, G_s_max, G_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 800
    },
    
    # Green Row 2
    {
        'vertices': [(750, 115), (1140, 108), (1160, 143), (755, 155)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([G_h_max, G_s_max, G_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 800
    },

    # Orange Row 1
    {
        'vertices': [(430, 75), (750, 87), (750, 105), (415, 90)],
        'lower_gray': np.array([0, 0, 50]), 
        'upper_gray': np.array([O_h_max, O_s_max, O_v_max]),
        'half_side': 5,
        'tolerance': 0.9,
        'min_area': 500
    },

    # Orange Row 2
    {
        'vertices': [(750, 89), (1090, 80), (1110, 95), (755, 105)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([O_h_max, O_s_max, O_v_max]),
        'half_side': 5,
        'tolerance': 0.9,
        'min_area': 500
    },

    # Yellow Row 1
    {
        'vertices': [(365, 25), (540, 25), (530, 55), (350, 50)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 6,
        'tolerance': 1.1,
        'min_area': 300
    },

    # Yellow Row 2
    {
        'vertices': [(540, 35), (760, 38), (760, 65), (530, 55)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 6,
        'tolerance': 1.1,
        'min_area': 300
    },

    # Yellow Row 3
    {
        'vertices': [(760, 38), (1040, 32), (1050, 55), (760, 65)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 6,
        'tolerance': 1,
        'min_area': 300
    },
]


    # Define coordenates for parking stalls
parking_spots = [
    
    # Blues
    {"Label": "B01", "vertices": np.array([[80, 395], [114, 397], [33, 482], [3, 478]]), "Umbral": 0.65, "ArrayVal": 30},
    {"Label": "B02", "vertices": np.array([[120, 394], [162, 400], [75, 497], [35, 490]]), "Umbral": 0.65, "ArrayVal": 29},
    {"Label": "B03", "vertices": np.array([[164, 400], [202, 402], [125, 499], [80, 497]]), "Umbral": 0.65, "ArrayVal": 28},
    {"Label": "B04", "vertices": np.array([[205, 402], [245, 404], [175, 502], [130, 499]]), "Umbral": 0.65, "ArrayVal": 27},
    {"Label": "B05", "vertices": np.array([[245, 404], [283, 406], [223, 505], [180, 502]]), "Umbral": 0.65, "ArrayVal": 26},
    {"Label": "B06", "vertices": np.array([[287, 406], [325, 408], [270, 510], [228, 507]]), "Umbral": 0.65, "ArrayVal": 25},
    {"Label": "B07", "vertices": np.array([[330, 408], [363, 412], [320, 514], [278, 510]]), "Umbral": 0.65, "ArrayVal": 24},
    {"Label": "B08", "vertices": np.array([[365, 412], [405, 414], [370, 518], [325, 516]]), "Umbral": 0.65, "ArrayVal": 23},
    {"Label": "B09", "vertices": np.array([[410, 414], [453, 418], [420, 523], [375, 520]]), "Umbral": 0.65, "ArrayVal": 22},
    {"Label": "B10", "vertices": np.array([[454, 418], [500, 421], [472, 526], [425, 523]]), "Umbral": 0.65, "ArrayVal": 21},
    {"Label": "B11", "vertices": np.array([[505, 421], [547, 424], [526, 530], [477, 526]]), "Umbral": 0.65, "ArrayVal": 20},
    {"Label": "B12", "vertices": np.array([[550, 425], [591, 428], [575, 535], [530, 530]]), "Umbral": 0.65, "ArrayVal": 19},
    {"Label": "B13", "vertices": np.array([[595, 428], [638, 432], [625, 542], [580, 536]]), "Umbral": 0.65, "ArrayVal": 18},
    {"Label": "B14", "vertices": np.array([[643, 432], [685, 435], [680, 544], [632, 542]]), "Umbral": 0.65, "ArrayVal": 17},
    {"Label": "B15", "vertices": np.array([[687, 435], [735, 437], [730, 548], [683, 545]]), "Umbral": 0.65, "ArrayVal": 16},
    {"Label": "B16", "vertices": np.array([[740, 439], [783, 442], [785, 553], [735, 550]]), "Umbral": 0.65, "ArrayVal": 15},
    {"Label": "B17", "vertices": np.array([[785, 442], [830, 442], [840, 553], [790, 550]]), "Umbral": 0.65, "ArrayVal": 14},
    {"Label": "B18", "vertices": np.array([[835, 442], [877, 444], [897, 555], [850, 555]]), "Umbral": 0.65, "ArrayVal": 13},
    {"Label": "B19", "vertices": np.array([[880, 444], [925, 446], [950, 555], [900, 556]]), "Umbral": 0.65, "ArrayVal": 12},
    {"Label": "B20", "vertices": np.array([[930, 446], [975, 446], [1010, 557], [960, 557]]), "Umbral": 0.65, "ArrayVal": 11},
    {"Label": "B21", "vertices": np.array([[980, 446], [1025, 446], [1070, 562], [1020, 562]]), "Umbral": 0.65, "ArrayVal": 10},
    {"Label": "B22", "vertices": np.array([[1030, 446], [1080, 446], [1125, 564], [1075, 562]]), "Umbral": 0.65, "ArrayVal": 9},
    {"Label": "B23", "vertices": np.array([[1085, 446], [1135, 448], [1190, 568], [1130, 566]]), "Umbral": 0.65, "ArrayVal": 8},
    {"Label": "B24", "vertices": np.array([[1140, 448], [1180, 448], [1240, 572], [1195, 570]]), "Umbral": 0.65, "ArrayVal": 7},
    {"Label": "B25", "vertices": np.array([[1200, 450], [1250, 448], [1305, 570], [1250, 572]]), "Umbral": 0.65, "ArrayVal": 6},
    {"Label": "B26", "vertices": np.array([[1250, 448], [1300, 446], [1360, 568], [1310, 572]]), "Umbral": 0.65, "ArrayVal": 5},
    {"Label": "B27", "vertices": np.array([[1300, 446], [1350, 444], [1405, 558], [1360, 566]]), "Umbral": 0.65, "ArrayVal": 4},
    {"Label": "B28", "vertices": np.array([[1355, 442], [1390, 436], [1445, 544], [1410, 556]]), "Umbral": 0.65, "ArrayVal": 3},
    {"Label": "B29", "vertices": np.array([[1390, 442], [1430, 436], [1490, 530], [1450, 542]]), "Umbral": 0.65, "ArrayVal": 2},
    {"Label": "B30", "vertices": np.array([[1440, 440], [1475, 430], [1500, 520], [1483, 530]]), "Umbral": 0.65, "ArrayVal": 1},
    {"Label": "B31", "vertices": np.array([[1480, 410], [1500, 397], [1500, 450], [1495, 450]]), "Umbral": 0.65, "ArrayVal": 0},
    
    # Purples
    {"Label": "P01", "vertices": np.array([[245, 235], [275, 237], [240, 290], [205, 286]]), "Umbral": 0.65, "ArrayVal": 60},
    {"Label": "P02", "vertices": np.array([[275, 234], [305, 236], [276, 292], [243, 290]]), "Umbral": 0.65, "ArrayVal": 59},
    {"Label": "P03", "vertices": np.array([[312, 236], [342, 238], [312, 296], [278, 292]]), "Umbral": 0.65, "ArrayVal": 58},
    {"Label": "P04", "vertices": np.array([[345, 238], [375, 239], [349, 298], [315, 296]]), "Umbral": 0.65, "ArrayVal": 57},
    {"Label": "P05", "vertices": np.array([[375, 239], [410, 241], [385, 302], [352, 300]]), "Umbral": 0.65, "ArrayVal": 56},
    {"Label": "P06", "vertices": np.array([[410, 242], [440, 243], [422, 305], [388, 303]]), "Umbral": 0.65, "ArrayVal": 55},
    {"Label": "P07", "vertices": np.array([[445, 247], [475, 278], [460, 307], [425, 305]]), "Umbral": 0.65, "ArrayVal": 54},
    {"Label": "P08", "vertices": np.array([[480, 248], [513, 250], [496, 311], [460, 309]]), "Umbral": 0.65, "ArrayVal": 53},
    {"Label": "P09", "vertices": np.array([[515, 252], [550, 254], [535, 315], [499, 312]]), "Umbral": 0.65, "ArrayVal": 52},
    {"Label": "P10", "vertices": np.array([[552, 254], [583, 256], [573, 317], [538, 315]]), "Umbral": 0.65, "ArrayVal": 51},
    {"Label": "P11", "vertices": np.array([[585, 258], [618, 260], [612, 319], [575, 317]]), "Umbral": 0.65, "ArrayVal": 50},
    {"Label": "P12", "vertices": np.array([[621, 260], [655, 262], [650, 322], [615, 319]]), "Umbral": 0.65, "ArrayVal": 49},
    {"Label": "P13", "vertices": np.array([[657, 262], [690, 265], [690, 326], [650, 324]]), "Umbral": 0.65, "ArrayVal": 48},
    {"Label": "P14", "vertices": np.array([[690, 265], [725, 266], [730, 329], [690, 327]]), "Umbral": 0.65, "ArrayVal": 47},
    {"Label": "P15", "vertices": np.array([[730, 266], [765, 266], [765, 329], [730, 329]]), "Umbral": 0.65, "ArrayVal": 46},
    {"Label": "P16", "vertices": np.array([[765, 266], [803, 266], [810, 330], [765, 329]]), "Umbral": 0.65, "ArrayVal": 45},
    {"Label": "P17", "vertices": np.array([[803, 266], [838, 266], [850, 331], [810, 330]]), "Umbral": 0.65, "ArrayVal": 44},
    {"Label": "P18", "vertices": np.array([[838, 266], [880, 265], [890, 330], [850, 331]]), "Umbral": 0.65, "ArrayVal": 43},
    {"Label": "P19", "vertices": np.array([[880, 265], [920, 266], [930, 331], [892, 330]]), "Umbral": 0.65, "ArrayVal": 42},
    {"Label": "P20", "vertices": np.array([[920, 266], [960, 266], [975, 332], [935, 331]]), "Umbral": 0.65, "ArrayVal": 41},
    {"Label": "P21", "vertices": np.array([[960, 266], [1000, 265], [1015, 333], [975, 332]]), "Umbral": 0.65, "ArrayVal": 40},
    {"Label": "P22", "vertices": np.array([[1000, 265], [1037, 265], [1060, 333], [1020, 333]]), "Umbral": 0.65, "ArrayVal": 39},
    {"Label": "P23", "vertices": np.array([[1037, 265], [1075, 265], [1100, 333], [1060, 333]]), "Umbral": 0.65, "ArrayVal": 38},
    {"Label": "P24", "vertices": np.array([[1080, 266], [1118, 265], [1143, 334], [1102, 334]]), "Umbral": 0.65, "ArrayVal": 37},
    {"Label": "P25", "vertices": np.array([[1118, 266], [1158, 265], [1187, 334], [1145, 334]]), "Umbral": 0.65, "ArrayVal": 36},
    {"Label": "P26", "vertices": np.array([[1155, 266], [1195, 265], [1231, 334], [1190, 334]]), "Umbral": 0.65, "ArrayVal": 35},
    {"Label": "P27", "vertices": np.array([[1195, 266], [1230, 266], [1275, 334], [1235, 334]]), "Umbral": 0.65, "ArrayVal": 34},
    {"Label": "P28", "vertices": np.array([[1230, 266], [1270, 266], [1320, 333], [1280, 334]]), "Umbral": 0.65, "ArrayVal": 33},
    {"Label": "P29", "vertices": np.array([[1275, 268], [1310, 266], [1360, 331], [1325, 334]]), "Umbral": 0.65, "ArrayVal": 32},
    {"Label": "P30", "vertices": np.array([[1315, 266], [1352, 264], [1400, 327], [1365, 331]]), "Umbral": 0.65, "ArrayVal": 31},
    
    # Reds
##    {"Label": "R01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 89},
##    {"Label": "R02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 88},
##    {"Label": "R03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 87},
##    {"Label": "R04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 86},
##    {"Label": "R05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 85},
##    {"Label": "R06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 84},
##    {"Label": "R07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 83},
##    {"Label": "R08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 82},
##    {"Label": "R09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 81},
##    {"Label": "R10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 80},
##    {"Label": "R11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 79},
##    {"Label": "R12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 78},
##    {"Label": "R13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 77},
##    {"Label": "R14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 76},
##    {"Label": "R15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 75},
##    {"Label": "R16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 74},
##    {"Label": "R17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 73},
##    {"Label": "R18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 72},
##    {"Label": "R19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 71},
##    {"Label": "R20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 70},
##    {"Label": "R21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 69},
##    {"Label": "R22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 68},
##    {"Label": "R23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 67},
##    {"Label": "R24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 66},
##    {"Label": "R25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 65},
##    {"Label": "R26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 64},
##    {"Label": "R27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 63},
##    {"Label": "R28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 62},
##    {"Label": "R29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 61},

    # Greens
##    {"Label": "G01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 119},
##    {"Label": "G02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 118},
##    {"Label": "G03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 117},
##    {"Label": "G04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 116},
##    {"Label": "G05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 115},
##    {"Label": "G06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 114},
##    {"Label": "G07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 113},
##    {"Label": "G08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 112},
##    {"Label": "G09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 111},
##    {"Label": "G10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 110},
##    {"Label": "G11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 109},
##    {"Label": "G12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 108},
##    {"Label": "G13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 107},
##    {"Label": "G14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 106},
##    {"Label": "G15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 105},
##    {"Label": "G16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 104},
##    {"Label": "G17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 103},
##    {"Label": "G18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 102},
##    {"Label": "G19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 101},
##    {"Label": "G20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 100},
##    {"Label": "G21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 99},
##    {"Label": "G22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 98},
##    {"Label": "G23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 97},
##    {"Label": "G24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 96},
##    {"Label": "G25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 95},
##    {"Label": "G26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 94},
##    {"Label": "G27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 93},
##    {"Label": "G28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 92},
##    {"Label": "G29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 91},
##    {"Label": "G30", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 90},
    
    # Oranges
##    {"Label": "O01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 149},
##    {"Label": "O02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 148},
##    {"Label": "O03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 147},
##    {"Label": "O04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 146},
##    {"Label": "O05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 145},
##    {"Label": "O06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 144},
##    {"Label": "O07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 143},
##    {"Label": "O08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 142},
##    {"Label": "O09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 141},
##    {"Label": "O10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 140},
##    {"Label": "O11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 139},
##    {"Label": "O12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 138},
##    {"Label": "O13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 137},
##    {"Label": "O14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 136},
##    {"Label": "O15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 135},
##    {"Label": "O16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 134},
##    {"Label": "O17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 133},
##    {"Label": "O18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 132},
##    {"Label": "O19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 131},
##    {"Label": "O20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 130},
##    {"Label": "O21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 129},
##    {"Label": "O22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 128},
##    {"Label": "O23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 127},
##    {"Label": "O24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 126},
##    {"Label": "O25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 125},
##    {"Label": "O26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 124},
##    {"Label": "O27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 123},
##    {"Label": "O28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 122},
##    {"Label": "O29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 121},
##    {"Label": "O30", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 120},

    # Yellows
##    {"Label": "Y01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 179},
##    {"Label": "Y02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 178},
##    {"Label": "Y03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 177},
##    {"Label": "Y04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 176},
##    {"Label": "Y05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 175},
##    {"Label": "Y06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 174},
##    {"Label": "Y07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 173},
##    {"Label": "Y08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 172},
##    {"Label": "Y09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 171},
##    {"Label": "Y10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 170},
##    {"Label": "Y11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 169},
##    {"Label": "Y12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 168},
##    {"Label": "Y13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 167},
##    {"Label": "Y14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 166},
##    {"Label": "Y15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 165},
##    {"Label": "Y16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 164},
##    {"Label": "Y17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 163},
##    {"Label": "Y18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 162},
##    {"Label": "Y19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 161},
##    {"Label": "Y20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 160},
##    {"Label": "Y21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 159},
##    {"Label": "Y22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 158},
##    {"Label": "Y23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 157},
##    {"Label": "Y24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 156},
##    {"Label": "Y25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 155},
##    {"Label": "Y26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 154},
##    {"Label": "Y27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 153},
##    {"Label": "Y28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 152},
##    {"Label": "Y29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 151},
##    {"Label": "Y30", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 150},
]


# Process each region based on the parameters
for region in regions:
    vertices = region['vertices']
    poly_mask = np.zeros(PLResized.shape[:2], dtype=np.uint8)
    vertices_array = np.array([vertices], dtype=np.int32)
    cv.fillPoly(poly_mask, vertices_array, 255)

    processed_roi = process_region(
        PLResized,
        vertices,
        region['lower_gray'],
        region['upper_gray'],
        region['half_side'],
        region['tolerance'],
        region['min_area']
    )

    PLResized[poly_mask > 0] = processed_roi[poly_mask > 0]

imgz = PLResized.copy()
    
# Process each parking stall, measure the percentage of green pixels
for spot in parking_spots:
    
    # Create a mask for each polygonal region
    mask = np.zeros_like(imgz[:,:,0])
    cv.fillConvexPoly(mask, spot["vertices"], 255)
    
    # Test the ratio of green pixels and compare to the minimum requirement
    green_pixels = cv.countNonZero(cv.bitwise_and(mask, cv.inRange(imgz, (0, 255, 0), (0, 255, 0))))
    total_pixels = cv.countNonZero(mask)
    
    
    if green_pixels / total_pixels > (spot['Umbral']):
        OpenSpotsCtr = OpenSpotsCtr + 1
        ParkingSpotsArray[spot['ArrayVal']] = 0
        print(f"Parking spot {spot['Label']} is open")
        


# Conversion Leonardo
TX_list = TransformToDecimal(ParkingSpotsArray)

# TX Snipet
ser.write(TX_list.encode('utf-8'))

# Print
print(f"There are {OpenSpotsCtr} open spots.")
print('\n Parking Lot Occupancy Array: \n', ParkingSpotsArray)
print('\n List', TX_list)

# Show Images
cv.imshow("Funciona?", PLResized)
cv.imshow("OG", PLOG)
cv.waitKey(0)
cv.destroyAllWindows()
