"""
Program Name: Spot Detection from Pi Image, V-3
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
    decimal_str = decimal_str.strip(", ")  # Remueve la última coma y espacio
    return decimal_str


# Read Image, resize it and copy
PL = cv.imread('./Pics/PLPi3.jpg')
PLResized = cv.resize(PL, (1500, 600), interpolation = cv.INTER_AREA)
PLOG = PLResized.copy()


# Define lines for HSV values extraction
StartPointB = (2237, 582)
EndPointB = (2250, 685)

StartPointP = (2170, 355)
EndPointP = (2175, 410)

StartPointR = (2173, 48)
EndPointR = (2175, 65)

StartPointG = (2173, 48)
EndPointG = (2175, 65)

StartPointO = (2173, 48)
EndPointO = (2175, 65)

StartPointY = (2173, 48)
EndPointY = (2175, 65)

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
B_v_max = B_v_max + 15

# Purple
P_h_max = P_h_max + 20
P_s_max = P_s_max + 0
P_v_max = P_v_max + 20

# Red
R_h_max = R_h_max + 0
R_s_max = R_s_max + 5
R_v_max = R_v_max + 15

# Green
G_h_max = G_h_max + 0
G_s_max = G_s_max + 10
G_v_max = G_v_max + 10

# Orange
O_h_max = O_h_max + 0
O_s_max = O_s_max + 0
O_v_max = O_v_max + 15

# Yellow
Y_h_max = Y_h_max + 0
Y_s_max = Y_s_max + 0
Y_v_max = Y_v_max + 0


# Define regions of interest and adjust parameters
regions = [

    # Blue Row 1
    {
        'vertices': [(113, 433), (777, 455), (780, 570), (25, 550)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 10,
        'tolerance': 0.99,
        'min_area': 1000
    },

    # Blue Row 2
    {
        'vertices': [(777, 455), (1440, 440), (1500, 530), (780, 570)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([B_h_max, B_s_max, B_v_max]),
        'half_side': 10,
        'tolerance': 0.99,
        'min_area': 1000
    },

    # Purple Row 1
    {
        'vertices': [(260, 250), (760, 270), (760, 332), (200, 310)],
        'lower_gray': np.array([0, 0, 50]), 
        'upper_gray': np.array([P_h_max, P_s_max, P_v_max]), 
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 1000
    },

    # Purple Row 2
    {
        'vertices': [(760, 270), (1270, 260), (1315, 315), (760, 335)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([P_h_max, P_s_max, P_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 1000
    },
    
    # Red Row 1
    {
        'vertices': [(370, 185), (760, 205), (760, 245), (345, 230)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([R_h_max, R_s_max, R_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 700
    },
    
    # Red Row 2
    {
        'vertices': [(760, 205), (1200, 190), (1230, 225), (765, 245)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([R_h_max, R_s_max, R_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 700
    },

    # Green Row 1
    {
        'vertices': [(395, 98), (750, 118), (750, 150), (370, 130)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([G_h_max, G_s_max, G_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 500
    },
    
    # Green Row 2
    {
        'vertices': [(750, 118), (1090, 105), (1125, 136), (750, 150)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([G_h_max, G_s_max, G_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 500
    },

    # Orange Row 1
    {
        'vertices': [(443, 70), (755, 87), (752, 105), (430, 87)],
        'lower_gray': np.array([0, 0, 50]), 
        'upper_gray': np.array([O_h_max, O_s_max, O_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 300
    },

    # Orange Row 2
    {
        'vertices': [(755, 87), (1055, 77), (1065, 95), (758, 105)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([O_h_max, O_s_max, O_v_max]),
        'half_side': 8,
        'tolerance': 0.9,
        'min_area': 300
    },

    # Yellow Row 1
    {
        'vertices': [(360, 20), (530, 20), (522, 43), (354, 38)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 2,
        'tolerance': 0.9,
        'min_area': 150
    },

    # Yellow Row 2
    {
        'vertices': [(530, 20), (760, 30), (760, 57), (522, 43)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 300
    },

    # Yellow Row 3
    {
        'vertices': [(760, 32), (1010, 28), (1020, 45), (760, 55)],
        'lower_gray': np.array([0, 0, 50]),
        'upper_gray': np.array([Y_h_max, Y_s_max, Y_v_max]),
        'half_side': 4,
        'tolerance': 0.9,
        'min_area': 200
    },
]


    # Define coordenates for parking stalls
parking_spots = [
    
    #Blues
    {"Label": "B01", "vertices": np.array([[80, 430], [110, 433], [32, 532], [0, 525]]), "Umbral": 0.65, "ArrayVal": 30},
    {"Label": "B02", "vertices": np.array([[110, 435], [153, 438], [75, 538], [35, 530]]), "Umbral": 0.65, "ArrayVal": 29},
    {"Label": "B03", "vertices": np.array([[155, 438], [198, 440], [120, 543], [80, 541]]), "Umbral": 0.65, "ArrayVal": 28},
    {"Label": "B04", "vertices": np.array([[200, 440], [240, 440], [170, 545], [125, 543]]), "Umbral": 0.65, "ArrayVal": 27},
##    {"Label": "B05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 26},
##    {"Label": "B06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 25},
##    {"Label": "B07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 24},
##    {"Label": "B08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 23},
##    {"Label": "B09", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 22},
##    {"Label": "B10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 21},
    {"Label": "B11", "vertices": np.array([[518, 452], [555, 454], [530, 560], [485, 558]]), "Umbral": 0.65, "ArrayVal": 20},
##    {"Label": "B12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 19},
##    {"Label": "B13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 18},
    {"Label": "B14", "vertices": np.array([[650, 458], [690, 458], [682, 570], [635, 570]]), "Umbral": 0.65, "ArrayVal": 17},
    {"Label": "B15", "vertices": np.array([[695, 458], [735, 458], [730, 570], [687, 570]]), "Umbral": 0.65, "ArrayVal": 16},
##    {"Label": "B16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 15},
    {"Label": "B17", "vertices": np.array([[780, 458], [820, 458], [830, 570], [785, 570]]), "Umbral": 0.65, "ArrayVal": 14},
    {"Label": "B18", "vertices": np.array([[825, 458], [865, 456], [878, 567], [835, 568]]), "Umbral": 0.65, "ArrayVal": 13},
    {"Label": "B19", "vertices": np.array([[868, 456], [908, 456], [930, 565], [885, 566]]), "Umbral": 0.65, "ArrayVal": 12},
##    {"Label": "B20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 11},
##    {"Label": "B21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 10},
##    {"Label": "B22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 9},
##    {"Label": "B23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 8},
##    {"Label": "B24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 7},
##    {"Label": "B25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 6},
    {"Label": "B26", "vertices": np.array([[1175, 451], [1210, 450], [1270, 552], [1230, 555]]), "Umbral": 0.50, "ArrayVal": 5},
##    {"Label": "B27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 4},
##    {"Label": "B28", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 3},
##    {"Label": "B29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 2},
    {"Label": "B30", "vertices": np.array([[1345, 444], [1375, 440], [1440, 525], [1403, 530]]), "Umbral": 0.25, "ArrayVal": 1},
##    {"Label": "B31", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 0},

    
    #Purples
##    {"Label": "P01", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 60},
##    {"Label": "P02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 59},
##    {"Label": "P03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 58},
##    {"Label": "P04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 57},
##    {"Label": "P05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 56},
    {"Label": "P06", "vertices": np.array([[422, 259], [455, 260], [430, 317], [395, 315]]), "Umbral": 0.65, "ArrayVal": 55},
    {"Label": "P07", "vertices": np.array([[458, 260], [488, 260], [467, 319], [432, 317]]), "Umbral": 0.65, "ArrayVal": 54},
    {"Label": "P08", "vertices": np.array([[492, 263], [523, 265], [505, 321], [470, 319]]), "Umbral": 0.65, "ArrayVal": 53},
    {"Label": "P09", "vertices": np.array([[525, 263], [556, 265], [542, 325], [507, 322]]), "Umbral": 0.65, "ArrayVal": 52},
    {"Label": "P10", "vertices": np.array([[560, 266], [590, 268], [580, 328], [545, 325]]), "Umbral": 0.65, "ArrayVal": 51},
    {"Label": "P11", "vertices": np.array([[595, 267], [625, 268], [617, 330], [583, 328]]), "Umbral": 0.65, "ArrayVal": 50},
    {"Label": "P12", "vertices": np.array([[630, 268], [655, 269], [652, 332], [620, 330]]), "Umbral": 0.65, "ArrayVal": 49},
##    {"Label": "P13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 48},
##    {"Label": "P14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 47},
    {"Label": "P15", "vertices": np.array([[730, 270], [758, 271], [760, 332], [732, 333]]), "Umbral": 0.65, "ArrayVal": 46},
##    {"Label": "P16", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 45},
##    {"Label": "P17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 44},
##    {"Label": "P18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 43},
##    {"Label": "P19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 42},
    {"Label": "P20", "vertices": np.array([[900, 271], [930, 271], [950, 332], [915, 333]]), "Umbral": 0.65, "ArrayVal": 41},
##    {"Label": "P21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 40},
##    {"Label": "P22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 39},
##    {"Label": "P23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 38},
##    {"Label": "P24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 37},
##    {"Label": "P25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 36},
##    {"Label": "P26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 35},
##    {"Label": "P27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 34},
    {"Label": "P28", "vertices": np.array([[1170, 269], [1195, 269], [1242, 325], [1210, 325]]), "Umbral": 0.5, "ArrayVal": 33},
##    {"Label": "P29", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 32},
##    {"Label": "P30", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 31},

    #Reds
    {"Label": "R01", "vertices": np.array([[370, 187], [396, 188], [380, 222], [350, 220]]), "Umbral": 0.65, "ArrayVal": 89},
    {"Label": "R02", "vertices": np.array([[397, 191], [425, 191], [403, 228], [376, 228]]), "Umbral": 0.65, "ArrayVal": 88},
    {"Label": "R03", "vertices": np.array([[428, 191], [450, 191], [432, 228], [407, 228]]), "Umbral": 0.55, "ArrayVal": 87},
##    {"Label": "R04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 86},
##    {"Label": "R05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 85},
##    {"Label": "R06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 84},
##    {"Label": "R07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 83},
##    {"Label": "R08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 82},
    {"Label": "R09", "vertices": np.array([[595, 198], [622, 198], [618, 234], [590, 234]]), "Umbral": 0.65, "ArrayVal": 81},
##    {"Label": "R10", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 80},
    {"Label": "R11", "vertices": np.array([[652, 200], [678, 200], [675, 238], [647, 238]]), "Umbral": 0.65, "ArrayVal": 79},
##    {"Label": "R12", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 78},
    {"Label": "R13", "vertices": np.array([[705, 203], [730, 203], [732, 240], [708, 240]]), "Umbral": 0.65, "ArrayVal": 77},
    {"Label": "R14", "vertices": np.array([[735, 203], [755, 203], [762, 242], [738, 242]]), "Umbral": 0.65, "ArrayVal": 76},
    {"Label": "R15", "vertices": np.array([[760, 205], [785, 205], [793, 242], [768, 242]]), "Umbral": 0.65, "ArrayVal": 75},  
    {"Label": "R16", "vertices": np.array([[790, 205], [813, 205], [821, 242], [795, 242]]), "Umbral": 0.65, "ArrayVal": 74},
    {"Label": "R17", "vertices": np.array([[816, 205], [840, 205], [848, 242], [825, 242]]), "Umbral": 0.65, "ArrayVal": 73},
    {"Label": "R18", "vertices": np.array([[845, 205], [868, 205], [878, 242], [853, 242]]), "Umbral": 0.65, "ArrayVal": 72},
    {"Label": "R19", "vertices": np.array([[870, 204], [895, 203], [908, 242], [885, 243]]), "Umbral": 0.65, "ArrayVal": 71},
    {"Label": "R20", "vertices": np.array([[898, 203], [922, 202], [938, 241], [914, 242]]), "Umbral": 0.65, "ArrayVal": 70},
    {"Label": "R21", "vertices": np.array([[925, 202], [950, 201], [967, 240], [940, 241]]), "Umbral": 0.65, "ArrayVal": 69},
    {"Label": "R22", "vertices": np.array([[952, 201], [975, 199], [995, 239], [970, 240]]), "Umbral": 0.65, "ArrayVal": 68},
    {"Label": "R23", "vertices": np.array([[980, 199], [1005, 198], [1025, 238], [1000, 239]]), "Umbral": 0.65, "ArrayVal": 67},
    {"Label": "R24", "vertices": np.array([[1006, 199], [1028, 198], [1055, 237], [1028, 238]]), "Umbral": 0.65, "ArrayVal": 66},
    {"Label": "R25", "vertices": np.array([[1040, 198], [1063, 197], [1085, 235], [1058, 236]]), "Umbral": 0.65, "ArrayVal": 65},
    {"Label": "R26", "vertices": np.array([[1062, 197], [1084, 196], [1110, 234], [1088, 235]]), "Umbral": 0.15, "ArrayVal": 64},
    {"Label": "R27", "vertices": np.array([[1085, 195], [1110, 194], [1145, 232], [1120, 233]]), "Umbral": 0.65, "ArrayVal": 63},
    {"Label": "R28", "vertices": np.array([[1115, 193], [1140, 192], [1180, 231], [1148, 232]]), "Umbral": 0.65, "ArrayVal": 62},
    {"Label": "R29", "vertices": np.array([[1147, 193], [1170, 192], [1205, 229], [1180, 230]]), "Umbral": 0.65, "ArrayVal": 61},

    #Greens
    {"Label": "G01", "vertices": np.array([[396, 100], [417, 100], [403, 130], [376, 128]]), "Umbral": 0.65, "ArrayVal": 119},
    {"Label": "G02", "vertices": np.array([[417, 100], [440, 101], [430, 132], [403, 130]]), "Umbral": 0.65, "ArrayVal": 118},
    {"Label": "G03", "vertices": np.array([[443, 102], [463, 103], [452, 133], [430, 132]]), "Umbral": 0.65, "ArrayVal": 117},
    {"Label": "G04", "vertices": np.array([[465, 104], [488, 105], [478, 135], [455, 134]]), "Umbral": 0.65, "ArrayVal": 116},
    {"Label": "G05", "vertices": np.array([[490, 104], [515, 105], [500, 136], [478, 135]]), "Umbral": 0.65, "ArrayVal": 115},
    {"Label": "G06", "vertices": np.array([[514, 105], [537, 106], [527, 138], [503, 137]]), "Umbral": 0.65, "ArrayVal": 114},
    {"Label": "G07", "vertices": np.array([[537, 106], [558, 107], [552, 138], [530, 137]]), "Umbral": 0.65, "ArrayVal": 113},
    {"Label": "G08", "vertices": np.array([[562, 108], [583, 109], [580, 139], [557, 138]]), "Umbral": 0.65, "ArrayVal": 112},
    {"Label": "G09", "vertices": np.array([[587, 110], [607, 111], [602, 142], [582, 140]]), "Umbral": 0.65, "ArrayVal": 111},
    {"Label": "G10", "vertices": np.array([[609, 111], [629, 113], [625, 144], [605, 142]]), "Umbral": 0.65, "ArrayVal": 110},
    {"Label": "G11", "vertices": np.array([[631, 113], [653, 114], [651, 145], [628, 144]]), "Umbral": 0.65, "ArrayVal": 109},
    {"Label": "G12", "vertices": np.array([[655, 115], [675, 116], [675, 148], [654, 147]]), "Umbral": 0.65, "ArrayVal": 108},
    {"Label": "G13", "vertices": np.array([[678, 115], [700, 116], [700, 148], [678, 147]]), "Umbral": 0.65, "ArrayVal": 107},
    {"Label": "G14", "vertices": np.array([[703, 116], [723, 117], [725, 150], [703, 149]]), "Umbral": 0.65, "ArrayVal": 106},
    {"Label": "G15", "vertices": np.array([[725, 118], [748, 117], [750, 149], [728, 150]]), "Umbral": 0.65, "ArrayVal": 105},
    {"Label": "G16", "vertices": np.array([[752, 118], [770, 118], [775, 150], [755, 150]]), "Umbral": 0.65, "ArrayVal": 104},
    {"Label": "G17", "vertices": np.array([[772, 118], [793, 118], [800, 150], [778, 150]]), "Umbral": 0.65, "ArrayVal": 103},
    {"Label": "G18", "vertices": np.array([[795, 118], [817, 117], [825, 149], [802, 150]]), "Umbral": 0.65, "ArrayVal": 102},
    {"Label": "G19", "vertices": np.array([[819, 118], [840, 117], [850, 148], [827, 149]]), "Umbral": 0.65, "ArrayVal": 101},
    {"Label": "G20", "vertices": np.array([[843, 117], [865, 116], [874, 147], [852, 148]]), "Umbral": 0.65, "ArrayVal": 100},
    {"Label": "G21", "vertices": np.array([[865, 117], [884, 116], [898, 146], [875, 147]]), "Umbral": 0.65, "ArrayVal": 99},
    {"Label": "G22", "vertices": np.array([[887, 116], [908, 115], [923, 145], [900, 146]]), "Umbral": 0.65, "ArrayVal": 98},
    {"Label": "G23", "vertices": np.array([[911, 115], [930, 114], [947, 143], [925, 144]]), "Umbral": 0.65, "ArrayVal": 97},
    {"Label": "G24", "vertices": np.array([[935, 113], [955, 112], [970, 142], [950, 143]]), "Umbral": 0.65, "ArrayVal": 96},
    {"Label": "G25", "vertices": np.array([[955, 113], [975, 112], [995, 141], [973, 142]]), "Umbral": 0.65, "ArrayVal": 95},
    {"Label": "G26", "vertices": np.array([[980, 112], [1000, 111], [1022, 140], [997, 141]]), "Umbral": 0.65, "ArrayVal": 94},
    {"Label": "G27", "vertices": np.array([[1003, 112], [1023, 111], [1043, 138], [1022, 139]]), "Umbral": 0.65, "ArrayVal": 93},
    {"Label": "G28", "vertices": np.array([[1026, 109], [1045, 108], [1068, 138], [1047, 139]]), "Umbral": 0.65, "ArrayVal": 92},
    {"Label": "G29", "vertices": np.array([[1048, 109], [1068, 108], [1092, 138], [1070, 139]]), "Umbral": 0.65, "ArrayVal": 91},
    {"Label": "G30", "vertices": np.array([[1070, 108], [1090, 108], [1118, 136], [1095, 136]]), "Umbral": 0.65, "ArrayVal": 90},
    
    #Oranges
    {"Label": "O01", "vertices": np.array([[443, 72], [460, 73], [452, 84], [435, 83]]), "Umbral": 0.65, "ArrayVal": 149},
##    {"Label": "O02", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 148},
##    {"Label": "O03", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 147},
##    {"Label": "O04", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 146},
##    {"Label": "O05", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 145},
##    {"Label": "O06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 144},
##    {"Label": "O07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 143},
##    {"Label": "O08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 142},
    {"Label": "O09", "vertices": np.array([[610, 80], [627, 81], [625, 106], [608, 105]]), "Umbral": 0.65, "ArrayVal": 141},
    {"Label": "O10", "vertices": np.array([[628, 82], [647, 83], [645, 103], [626, 102]]), "Umbral": 0.65, "ArrayVal": 140},
    {"Label": "O11", "vertices": np.array([[648, 82], [668, 83], [667, 104], [647, 103]]), "Umbral": 0.4, "ArrayVal": 139},
    {"Label": "O12", "vertices": np.array([[668, 85], [687, 87], [685, 106], [667, 105]]), "Umbral": 0.55, "ArrayVal": 138},
    {"Label": "O13", "vertices": np.array([[689, 87], [706, 88], [705, 106], [688, 105]]), "Umbral": 0.65, "ArrayVal": 137},
##    {"Label": "O14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 136},
##    {"Label": "O15", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 135},
    {"Label": "O16", "vertices": np.array([[743, 87], [767, 88], [767, 106], [744, 106]]), "Umbral": 0.65, "ArrayVal": 134},
##    {"Label": "O17", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 133},
##    {"Label": "O18", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 132},
##    {"Label": "O19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 131},
    {"Label": "O20", "vertices": np.array([[828, 89], [845, 88], [847, 105], [830, 106]]), "Umbral": 0.65, "ArrayVal": 130},
##    {"Label": "O21", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 129},
##    {"Label": "O22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 128},
    {"Label": "O23", "vertices": np.array([[890, 87], [907, 86], [914, 105], [897, 106]]), "Umbral": 0.65, "ArrayVal": 127},
##    {"Label": "O24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 126},
##    {"Label": "O25", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 125},
##    {"Label": "O26", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 124},
##    {"Label": "O27", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 123},
    {"Label": "O28", "vertices": np.array([[992, 81], [1013, 80], [1020, 97], [1003, 98]]), "Umbral": 0.50, "ArrayVal": 122},
    {"Label": "O29", "vertices": np.array([[1013, 81], [1032, 80], [1042, 96], [1024, 97]]), "Umbral": 0.65, "ArrayVal": 121},
    {"Label": "O30", "vertices": np.array([[1034, 81], [1050, 80], [1063, 96], [1046, 97]]), "Umbral": 0.65, "ArrayVal": 120},

    #Yellows
    {"Label": "Y01", "vertices": np.array([[476, 21], [490, 22], [482, 39], [465, 38]]), "Umbral": 0.65, "ArrayVal": 179},
    {"Label": "Y02", "vertices": np.array([[492, 21], [508, 22], [500, 41], [482, 40]]), "Umbral": 0.65, "ArrayVal": 178},
    {"Label": "Y03", "vertices": np.array([[508, 21], [525, 22], [519, 41], [501, 40]]), "Umbral": 0.65, "ArrayVal": 177},
    {"Label": "Y04", "vertices": np.array([[527, 22], [545, 23], [538, 42], [520, 41]]), "Umbral": 0.65, "ArrayVal": 176},
    {"Label": "Y05", "vertices": np.array([[545, 23], [560, 24], [556, 43], [540, 42]]), "Umbral": 0.65, "ArrayVal": 175},
##    {"Label": "Y06", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 174},
##    {"Label": "Y07", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 173},
##    {"Label": "Y08", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 172},
    {"Label": "Y09", "vertices": np.array([[620, 27], [634, 28], [631, 47], [618, 46]]), "Umbral": 0.65, "ArrayVal": 171},
    {"Label": "Y10", "vertices": np.array([[635, 27], [653, 28], [650, 49], [633, 48]]), "Umbral": 0.65, "ArrayVal": 170},
##    {"Label": "Y11", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 169},
    {"Label": "Y12", "vertices": np.array([[670, 29], [687, 30], [687, 51], [670, 50]]), "Umbral": 0.65, "ArrayVal": 168},
##    {"Label": "Y13", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 167},
##    {"Label": "Y14", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 166},
    {"Label": "Y15", "vertices": np.array([[722, 30], [740, 31], [741, 53], [725, 52]]), "Umbral": 0.65, "ArrayVal": 165},
    {"Label": "Y16", "vertices": np.array([[743, 31], [759, 32], [762, 53], [743, 53]]), "Umbral": 0.15, "ArrayVal": 164},
    {"Label": "Y17", "vertices": np.array([[762, 33], [777, 33], [780, 53], [763, 53]]), "Umbral": 0.65, "ArrayVal": 163},
    {"Label": "Y18", "vertices": np.array([[780, 33], [796, 33], [798, 55], [782, 55]]), "Umbral": 0.65, "ArrayVal": 162},
##    {"Label": "Y19", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 161},
##    {"Label": "Y20", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 160},
    {"Label": "Y21", "vertices": np.array([[831, 34], [848, 33], [853, 53], [838, 54]]), "Umbral": 0.65, "ArrayVal": 159},
##    {"Label": "Y22", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 158},
##    {"Label": "Y23", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 157},
##    {"Label": "Y24", "vertices": np.array([[], [], [], []]), "Umbral": 0.65, "ArrayVal": 156},
    {"Label": "Y25", "vertices": np.array([[903, 31], [917, 30], [927, 50], [912, 51]]), "Umbral": 0.65, "ArrayVal": 155},
    {"Label": "Y26", "vertices": np.array([[919, 31], [935, 30], [945, 49], [929, 50]]), "Umbral": 0.65, "ArrayVal": 154},
    {"Label": "Y27", "vertices": np.array([[937, 30], [954, 29], [965, 48], [948, 50]]), "Umbral": 0.65, "ArrayVal": 153},
    {"Label": "Y28", "vertices": np.array([[954, 29], [970, 28], [982, 47], [966, 48]]), "Umbral": 0.65, "ArrayVal": 152},
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
