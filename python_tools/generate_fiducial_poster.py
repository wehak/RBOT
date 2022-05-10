from math import sqrt
import numpy as np
from pathlib import Path
import json
import subprocess

# from PIL import Image
# from cv2 import line
import cv2
 


""" parameters """
N_TAGS = 4
TAG_SIZE = 140 # mm
TAG_KEY = "APRILTAG_16h5" 
MARGIN_SIZE = 30 # mm
HOLE_DISTANCE = 48 # mm
HOLE_DIAMETER = 1 # mm
MODEL_BASE_DIAMETER = 70 # mm
MODEL_BASE_DISPERSION = 300 # mm
FIVE_SOCKETS = True
LINE_THICKNESS = 1 # mm
BORDERSIZE = 0.2 # mm
font_scale = 30
DPI = 1440


# functions that translate between mm and pixels based on DPI
def mm2pxl(millimeters):
    return round( millimeters / 25.4 * DPI)

def pxl2mm(pixels):
    return round( 25.4 * pixels / DPI )


# convert parameters in mm to pixels
tag_size = mm2pxl(TAG_SIZE)
model_base_diameter = mm2pxl(MODEL_BASE_DIAMETER)
model_base_dispersion = mm2pxl(MODEL_BASE_DISPERSION)
hole_distance = mm2pxl(HOLE_DISTANCE)
hole_diameter = mm2pxl(HOLE_DIAMETER)
margin_size = mm2pxl(MARGIN_SIZE)
bordersize = mm2pxl(BORDERSIZE)


#  map necessary to convert parameters to dictionary objects 
ARUCO_DICT = {
    "ARUCO_4X4_50": cv2.aruco.DICT_4X4_50,
	"ARUCO_4X4_100": cv2.aruco.DICT_4X4_100,
	"ARUCO_4X4_250": cv2.aruco.DICT_4X4_250,
	"ARUCO_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"ARUCO_5X5_50": cv2.aruco.DICT_5X5_50,
	"ARUCO_5X5_100": cv2.aruco.DICT_5X5_100,
	"ARUCO_5X5_250": cv2.aruco.DICT_5X5_250,
	"ARUCO_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"ARUCO_6X6_50": cv2.aruco.DICT_6X6_50,
	"ARUCO_6X6_100": cv2.aruco.DICT_6X6_100,
	"ARUCO_6X6_250": cv2.aruco.DICT_6X6_250,
	"ARUCO_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"ARUCO_7X7_50": cv2.aruco.DICT_7X7_50,
	"ARUCO_7X7_100": cv2.aruco.DICT_7X7_100,
	"ARUCO_7X7_250": cv2.aruco.DICT_7X7_250,
	"ARUCO_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# function to draw model sockets around an origin
def draw_circles_around_center(circle_center, image):
    holes = []
    for x in [1, -1]:
        for y in [1, -1]:
            coord = (
                circle_center[0] + round((hole_distance / 2) / sqrt(2)) * x, # x coord
                circle_center[1] + round((hole_distance / 2) / sqrt(2)) * y # y
            )
            holes.append(coord)

    # draw big circle
    image = cv2.circle(
            img=image,
            center=circle_center,
            radius=int(model_base_diameter/2),
            color=0,
            thickness=mm2pxl(1),
        )

    # draw screw holes for the model sockets
    for hole in holes:
        image = cv2.circle(
            img=image,
            center=hole,
            radius=hole_diameter,
            color=0,
            thickness=-1
        )
    
    return image

def make_board(tag_key, n):

    # set variables
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[tag_key])
    tags = []

    # make list of fiducials
    for i in range(4*n-4):
        tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(
            aruco_dict,
            i,
            tag_size,
            tag,
            1,
            )
        tags.append(tag)

    # make the board
    board_size = tag_size * n + margin_size * (n + 1)
    board = np.full((board_size, board_size, 1), 255, dtype="uint8")
    tag_positions = [None for i in range(4*n-4)]
    center = round(board_size/2)
    print("Image resolution:", board.shape[:2])
    
    # write fiducials to board
    for i in range(n):
        # write high and low row
        pos_y = margin_size + (margin_size + tag_size)*i
        board[pos_y: pos_y + tag_size, margin_size: margin_size+tag_size] = tags[i]
        board[pos_y: pos_y + tag_size, board_size - (margin_size+tag_size): board_size - margin_size] = tags[i+n]

        # record positions
        tag_positions[i] = (margin_size + tag_size//2, pos_y + tag_size//2,)
        tag_positions[i+n] = (board_size - (margin_size+tag_size//2), pos_y + tag_size//2)

        # write left and right column
        if (i != 0) and (i != n-1):
            board[margin_size :margin_size + tag_size, margin_size + (margin_size + tag_size)*i:margin_size + (margin_size + tag_size)*i + tag_size] = tags[i + n*2 - 1]
            board[board_size - (margin_size+tag_size): board_size - margin_size, margin_size + (margin_size + tag_size)*i : margin_size + (margin_size + tag_size)*i + tag_size] = tags[i + n*2 + n-2 - 1]

            # record positions
            tag_positions[i + n*2 - 1] = (margin_size + (margin_size + tag_size)*i + tag_size//2, margin_size + tag_size//2)
            tag_positions[i + n*2 + n-2 - 1] = (margin_size + (margin_size + tag_size)*i + tag_size//2, board_size - (margin_size+tag_size//2))
        
    # record data to the output json file
    tag_position_dict = {}
    for i, element in enumerate(tag_positions):
        tag_position_dict[i] = [
            pxl2mm(center) - pxl2mm(element[0]),
            pxl2mm(element[1]) - pxl2mm(center),
        ]


    # draw cross in the center
    line_t = mm2pxl(LINE_THICKNESS)
    cv2.line(
        board,
        (center - mm2pxl(5), center),
        (center + mm2pxl(5), center),
        0,
        thickness=line_t
    )

    cv2.line(
        board,
        (center, center - mm2pxl(5)),
        (center, center + mm2pxl(5)),
        0,
        thickness=line_t
    )

    # draw xy at a origin
    origin = (
        margin_size * 2 + tag_size, # x 
        board_size - (margin_size * 2 + tag_size) # y
        )

    cv2.line(
        board, 
        origin,
        (origin[0] + tag_size//2, origin[1]),
        0,
        thickness=line_t
    )

    cv2.putText(
        board,
        "x",
        (origin[0] + tag_size//2 + margin_size // 2, origin[1] + 7),
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale,
        0,
        mm2pxl(0.5),
    )

    cv2.line(
        board, 
        origin,
        (origin[0] , origin[1] - tag_size // 2),
        0,
        thickness=line_t
    )

    cv2.putText(
        board,
        "y",
        (origin[0] - 8, origin[1] - (tag_size//2 + margin_size//2)),
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale,
        0,
        mm2pxl(0.5),
    )

    # write dict name
    cv2.putText(
        board,
        tag_key,
        ((tag_size + margin_size * 2) + 0, (tag_size + margin_size * 2) + font_scale),
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale,
        0,
        mm2pxl(0.5),
    )

    # draw model base
    draw_circles_around_center((center, center), board)
    if FIVE_SOCKETS:
        model_bases = []
        for x in [1, -1]:
            for y in [1, -1]:
                coord = (
                    center + round((model_base_dispersion / 2) / sqrt(2)) * x, # x coord
                    center + round((model_base_dispersion / 2) / sqrt(2)) * y # y
                )
                model_bases.append(coord)
        print(f"Secondary models {pxl2mm(round((model_base_dispersion / 2) / sqrt(2)))} mm from center in X and Y")
        for xy_coord in model_bases:
            draw_circles_around_center(xy_coord, board)      

    # draw outside border to avoid printer from cropping too much
    board = cv2.copyMakeBorder(
        board,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    
    return board, tag_position_dict


# create output folder
output = Path("data/output/other/aruco_boards")
output.mkdir(parents=True, exist_ok=True)

# generate the poster
board, positions = make_board(TAG_KEY, N_TAGS)

# save as raster image
poster_name = f"poster_{TAG_KEY}_{TAG_SIZE}mm"
cv2.imwrite(str(Path(f"{output}/{poster_name}.pbm")), board)

# write distance between tags and poster center
with open(Path(f"{output}/poster_tag_positions.json"), "w") as f:
    json.dump(positions, f)

""" vectorize raster image """
# create svg file
bash_command = f"potrace {poster_name}.pbm -o {poster_name}.svg -b svg -r {DPI}"
process = subprocess.Popen(
    bash_command.split(),
    stdout=subprocess.PIPE,
    cwd=output.absolute()
    )
subprocess_out, subprocess_error = process.communicate()

if subprocess_error:
    print(subprocess_out)

# create pdf file
bash_command = f"potrace {poster_name}.pbm -o {poster_name}.pdf -b pdf -r {DPI}"
process = subprocess.Popen(
    bash_command.split(),
    stdout=subprocess.PIPE,
    cwd=output.absolute()
    )
subprocess_out, subprocess_error = process.communicate()

if subprocess_error:
    print(subprocess_out)

print(f"Image saved to \"{output}\"")
