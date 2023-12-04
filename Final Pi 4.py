"""
Program Name: Spot Detection from Pi Image, V-4
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
PL = cv.imread('./Pics/PLPi4.jpg')
PLResized = cv.resize(PL, (1500, 600), interpolation = cv.INTER_AREA)
PLOG = PLResized.copy()


# Define lines for HSV values extraction
StartPointB = (2122, 590)
EndPointB = (2109, 710)

StartPointP = (2155, 165)
EndPointP = (2155, 198)

StartPointR = (2155, 165)
EndPointR = (2155, 198)

StartPointG = (2155, 165)
EndPointG = (2155, 198)

StartPointO = (2155, 165)
EndPointO = (2155, 198)

StartPointY = (2155, 165)
EndPointY = (2155, 198)

# Get HSV Ranges for each row
B_h_min, B_h_max, B_s_min, B_s_max, B_v_min, B_v_max = get_hsv_range_from_line(PL, StartPointB, EndPointB)
P_h_min, P_h_max, P_s_min, P_s_max, P_v_min, P_v_max = get_hsv_range_from_line(PL, StartPointP, EndPointP)
R_h_min, R_h_max, R_s_min, R_s_max, R_v_min, R_v_max = get_hsv_range_from_line(PL, StartPointR, EndPointR)
G_h_min, G_h_max, G_s_min, G_s_max, G_v_min, G_v_max = get_hsv_range_from_line(PL, StartPointG, EndPointG)
O_h_min, O_h_max, O_s_min, O_s_max, O_v_min, O_v_max = get_hsv_range_from_line(PL, StartPointO, EndPointO)
Y_h_min, Y_h_max, Y_s_min, Y_s_max, Y_v_min, Y_v_max = get_hsv_range_from_line(PL, StartPointY, EndPointY)

# Exclusive Buffers
# Blue
B_h_max = B_h_max + 0
B_s_max = B_s_max + 0
B_v_max = B_v_max + 0

# Purple
P_h_max = P_h_max + 0
P_s_max = P_s_max + 0
P_v_max = P_v_max + 0

# Red
R_h_max = R_h_max + 0
R_s_max = R_s_max + 0
R_v_max = R_v_max + 0

# Green
G_h_max = G_h_max + 5
G_s_max = G_s_max + 0
G_v_max = G_v_max + 0

# Orange
O_h_max = O_h_max + 0
O_s_max = O_s_max + 0
O_v_max = O_v_max + 0

# Yellow
Y_h_max = Y_h_max + 0
Y_s_max = Y_s_max + 0
Y_v_max = Y_v_max + 0


# Define regions of interest and adjust parameters
regions = [

    # Blue Row 1
    {
        'vertices': [(60, 435), (295, 448), (250, 560), (0, 530)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 10,
        'tolerance': 0.9,
        'min_area': 1200
    },

    # Blue Row 2
    {
        'vertices': [(295, 448), (757, 465), (750, 575), (250, 560)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 10,
        'tolerance': 0.9,
        'min_area': 2000
    },

    # Blue Row 3
    {
        'vertices': [(757, 465), (1445, 455), (1500, 555), (750, 575)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 10,
        'tolerance': 0.9,
        'min_area': 1200
    },

    # Purple Row 1
    {
        'vertices': [(240, 266), (751, 283), (752, 345), (192, 323)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([P_h_max, P_s_max, P_v_max]), 
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 1500
    },

    # Purple Row 2
    {
        'vertices': [(755, 283), (1280, 268), (1325, 328), (752, 345)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([P_h_max, P_s_max, P_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 1500
    },
    
    # Red Row 1
    {
        'vertices': [(360, 205), (760, 217), (762, 255), (345, 235)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([R_h_max, R_s_max, R_v_max]),
        'half_side': 6,
        'tolerance': 0.9,
        'min_area': 1000
    },
    
    # Red Row 2
    {
        'vertices': [(763, 217), (1200, 200), (1225, 235), (762, 255)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([R_h_max, R_s_max, R_v_max]),
        'half_side': 6,
        'tolerance': 0.9,
        'min_area': 1000
    },

    # Green Row 1
    {
        'vertices': [(405, 115), (760, 130), (760, 162), (380, 148)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([G_h_max, G_s_max, G_v_max]),
        'half_side': 8,
        'tolerance': 0.95,
        'min_area': 800
    },
    
    # Green Row 2
    {
        'vertices': [(765, 130), (1115, 110), (1140, 143), (765, 162)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([G_h_max, G_s_max, G_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 800
    },

    # Orange Row 1
    {
        'vertices': [(450, 87), (765, 100), (765, 118), (443, 108)],
        'lower_gray': np.array([0, 0, 50]), 
        'upper_gray': np.array([O_h_max, O_s_max, O_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 100
    },

    # Orange Row 2
    {
        'vertices': [(765, 100), (1076, 85), (1088, 102), (765, 118)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([O_h_max, O_s_max, O_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 150
    },

    # Yellow Row 1
    {
        'vertices': [(491, 37), (564, 40), (555, 59), (482, 56)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 150
    },

    # Yellow Row 2
    {
        'vertices': [(564, 40), (780, 45), (779, 67), (555, 59)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 150
    },

    # Yellow Row 3
    {
        'vertices': [(780, 45), (1030, 35), (1050, 55), (779, 67)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 150
    },
]


    # Define coordenates for parking stalls
parking_spots = [
    
    #Blues
    {"Label": "B01", "vertices": np.array([[65, 435], [102, 440], [22, 532], [0, 525]]), "Umbral": 0.5, "ArrayVal": 30},
    {"Label": "B02", "vertices": np.array([[102, 440], [140, 444], [58, 543], [22, 532]]), "Umbral": 0.2, "ArrayVal": 29},
##    {"Label": "B03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 28},
##    {"Label": "B04", "vertices": np.array([[], [], [], []]), "Umbral": 0.33, "ArrayVal": 27},
    {"Label": "B05", "vertices": np.array([[230, 448], [265, 450], [195, 555], [152, 553]]), "Umbral": 0.45, "ArrayVal": 26},
##    {"Label": "B06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 25},
##    {"Label": "B07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 24},
##    {"Label": "B08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 23},
##    {"Label": "B09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 22},
##    {"Label": "B10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 21},
##    {"Label": "B11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 20},
##    {"Label": "B12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 19},
##    {"Label": "B13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 18},
##    {"Label": "B14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 17},
##    {"Label": "B15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 16},
##    {"Label": "B16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 15},
##    {"Label": "B17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 14},
##    {"Label": "B18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 13},
##    {"Label": "B19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 12},
##    {"Label": "B20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 11},
##    {"Label": "B21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 10},
##    {"Label": "B22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 9},
##    {"Label": "B23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 8},
##    {"Label": "B24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 7},
##    {"Label": "B25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 6},
##    {"Label": "B26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 5},
##    {"Label": "B27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 4},
##    {"Label": "B28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 3},
##    {"Label": "B29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 2},
##    {"Label": "B30", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 1},
##    {"Label": "B31", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 0},
    
    #Purples
##    {"Label": "P01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 60},
##    {"Label": "P02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 59},
##    {"Label": "P03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 58},
##    {"Label": "P04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 57},
##    {"Label": "P05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 56},
##    {"Label": "P06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 55},
##    {"Label": "P07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 54},
    {"Label": "P08", "vertices": np.array([[484, 270], [513, 271], [492, 336], [458, 334]]), "Umbral": 0.65, "ArrayVal": 53},
    {"Label": "P09", "vertices": np.array([[516, 272], [547, 274], [530, 338], [495, 336]]), "Umbral": 0.65, "ArrayVal": 52},
    {"Label": "P10", "vertices": np.array([[548, 275], [578, 276], [567, 339], [532, 338]]), "Umbral": 0.65, "ArrayVal": 51},
    {"Label": "P11", "vertices": np.array([[580, 278], [615, 279], [603, 341], [570, 340]]), "Umbral": 0.65, "ArrayVal": 50},
    {"Label": "P12", "vertices": np.array([[617, 279], [648, 280], [642, 342], [607, 341]]), "Umbral": 0.65, "ArrayVal": 49},
    {"Label": "P13", "vertices": np.array([[650, 281], [685, 282], [679, 344], [643, 343]]), "Umbral": 0.65, "ArrayVal": 48},
    {"Label": "P14", "vertices": np.array([[686, 282], [720, 283], [716, 345], [681, 344]]), "Umbral": 0.65, "ArrayVal": 47},
    {"Label": "P15", "vertices": np.array([[722, 283], [750, 284], [748, 346], [718, 345]]), "Umbral": 0.65, "ArrayVal": 46},
    {"Label": "P16", "vertices": np.array([[759, 280], [791, 280], [791, 346], [757, 346]]), "Umbral": 0.65, "ArrayVal": 45},
##    {"Label": "P17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 44},
##    {"Label": "P18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 43},
##    {"Label": "P19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 42},
##    {"Label": "P20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 41},
##    {"Label": "P21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 40},
##    {"Label": "P22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 39},
##    {"Label": "P23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 38},
##    {"Label": "P24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 37},
##    {"Label": "P25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 36},
    {"Label": "P26", "vertices": np.array([[1105, 270], [1135, 269], [1170, 334], [1134, 335]]), "Umbral": 0.65, "ArrayVal": 35},
##    {"Label": "P27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 34},
##    {"Label": "P28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 33},
##    {"Label": "P29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 32},
    {"Label": "P30", "vertices": np.array([[1253, 269], [1282, 267], [1330, 325], [1290, 328]]), "Umbral": 0.65, "ArrayVal": 31},
    
    #Reds
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
    {"Label": "R14", "vertices": np.array([[737, 215], [760, 215], [761, 252], [735, 252]]), "Umbral": 0.65, "ArrayVal": 76},
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

    #Greens
##    {"Label": "G01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 119},
##    {"Label": "G02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 118},
    {"Label": "G03", "vertices": np.array([[478, 118], [498, 119], [483, 152], [462, 151]]), "Umbral": 0.65, "ArrayVal": 117},
    {"Label": "G04", "vertices": np.array([[451, 117], [473, 118], [459, 151], [436, 150]]), "Umbral": 0.65, "ArrayVal": 116},
##    {"Label": "G05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 115},
##    {"Label": "G06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 114},
##    {"Label": "G07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 113},
##    {"Label": "G08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 112},
##    {"Label": "G09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 111},
    {"Label": "G10", "vertices": np.array([[618, 122], [638, 123], [632, 158], [611, 157]]), "Umbral": 0.65, "ArrayVal": 110},
##    {"Label": "G11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 109},
##    {"Label": "G12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 108},
##    {"Label": "G13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 107},
##    {"Label": "G14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 106},
##    {"Label": "G15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 105},
    {"Label": "G16", "vertices": np.array([[765, 126], [784, 126], [785, 160], [765, 160]]), "Umbral": 0.65, "ArrayVal": 104},
##    {"Label": "G17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 103},
    {"Label": "G18", "vertices": np.array([[807, 126], [828, 125], [834, 159], [810, 160]]), "Umbral": 0.65, "ArrayVal": 102},
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
    
    #Oranges
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
    {"Label": "O12", "vertices": np.array([[684, 98], [701, 98], [699, 113], [682, 112]]), "Umbral": 0.65, "ArrayVal": 138},
##    {"Label": "O13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 137},
##    {"Label": "O14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 136},
##    {"Label": "O15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 135},
##    {"Label": "O16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 134},
##    {"Label": "O17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 133},
    {"Label": "O18", "vertices": np.array([[805, 98], [823, 98], [824, 117], [806, 117]]), "Umbral": 0.65, "ArrayVal": 132},
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

    #Yellows
    {"Label": "Y01", "vertices": np.array([[493, 37], [508, 38], [500, 57], [483, 56]]), "Umbral": 0.65, "ArrayVal": 179},
##    {"Label": "Y02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 178},
##    {"Label": "Y03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 177},
##    {"Label": "Y04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 176},
    {"Label": "Y05", "vertices": np.array([[563, 40], [580, 41], [572, 60], [557, 59]]), "Umbral": 0.65, "ArrayVal": 175},
##    {"Label": "Y06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 174},
##    {"Label": "Y07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 173},
    {"Label": "Y08", "vertices": np.array([[617, 41], [633, 42], [630, 62], [613, 61]]), "Umbral": 0.65, "ArrayVal": 172},
##    {"Label": "Y09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 171},
##    {"Label": "Y10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 170},
##    {"Label": "Y11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 169},
    {"Label": "Y12", "vertices": np.array([[690, 44], [705, 45], [703, 66], [687, 65]]), "Umbral": 0.65, "ArrayVal": 168},
##    {"Label": "Y13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 167},
##    {"Label": "Y14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 166},
##    {"Label": "Y15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 165},
    {"Label": "Y16", "vertices": np.array([[762, 45], [780, 45], [780, 67], [762, 67]]), "Umbral": 0.3, "ArrayVal": 164},
##    {"Label": "Y17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 163},
##    {"Label": "Y18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 162},
##    {"Label": "Y19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 161},
    {"Label": "Y20", "vertices": np.array([[835, 45], [850, 45], [855, 65], [838, 66]]), "Umbral": 0.65, "ArrayVal": 160},
##    {"Label": "Y21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 159},
##    {"Label": "Y22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 158},
##    {"Label": "Y23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 157},
##    {"Label": "Y24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 156},
##    {"Label": "Y25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 155},
##    {"Label": "Y26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 154},
##    {"Label": "Y27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 153},
    {"Label": "Y28", "vertices": np.array([[978, 37], [995, 36], [1006, 55], [988, 56]]), "Umbral": 0.65, "ArrayVal": 152},
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
