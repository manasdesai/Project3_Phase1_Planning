#!/usr/bin/env python3


# Import Libraries
import cv2
import copy
import time
import heapq
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
from typing import List, Dict
import pdb

#############
## CONSTANTS
# Scale Factor
sf = 2  # for Screen

SHARP_LEFT = 60
SLOW_LEFT = 30
STRAIGHT = 0
SLOW_RIGHT = -30
SHARP_RIGHT = -60

ANGLE_MOVES = (60, 30, 0, -30, -60)

# Screen width, height
WIDTH = 600 * sf
HEIGHT = 250 * sf

INITIAL_COORD = (30 * sf, 10 * sf, 0)  # already factored with SF
# SOLUTION_COORD = (10 * SF, 7 * SF)  # already factored with SF
SOLUTION_COORD = (460, 450, 0)  # already factored with SF


class Node:
    """
    class to hold the Node objects.
    """

    def __init__(
        self,
        c_index: int,
        p_index: int,
        cost: int,
        cost_estimate: int,
        x: float,
        y: float,
        theta: int,
    ) -> None:
        self.current_index = c_index
        self.parent_index = p_index
        self.cost = cost
        self.cost_estimate = cost_estimate
        self.x = x
        self.y = y
        self._theta = theta


def EstimateCost(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))  # Euclidean distance
    # return abs((x2 - x1) + (y2 - y1))  # Manhatten distance


def MoveSharpLeft(
    current_node: Node, step_size: int, end_coord: tuple, SF: int
) -> Node:
    end_x, end_y = end_coord
    newNode = copy.deepcopy(current_node)

    newNode.current_index = None
    newNode.parent_index = current_node.current_index
    newNode.cost += step_size * SF

    newNode.x = (
        round(
            current_node.x
            + step_size * np.cos(np.deg2rad(current_node._theta + SHARP_LEFT) * SF) * 2
        )
        / 2
    )
    newNode.y = (
        round(
            current_node.y
            + step_size * np.sin(np.deg2rad(current_node._theta + SHARP_LEFT) * SF) * 2
        )
        / 2
    )
    newNode._theta = current_node._theta + SHARP_LEFT
    newNode.cost_estimate = EstimateCost(newNode.x, newNode.y, end_x, end_y)

    return newNode


def MoveSlowLeft(current_node: Node, step_size: int, end_coord: tuple, SF: int) -> Node:
    end_x, end_y = end_coord
    newNode = copy.deepcopy(current_node)

    newNode.current_index = None
    newNode.parent_index = current_node.current_index
    newNode.cost += step_size * SF

    newNode.x = (
        round(
            current_node.x
            + step_size * np.cos(np.deg2rad(current_node._theta + SLOW_LEFT) * SF) * 2
        )
        / 2
    )
    newNode.y = (
        round(
            current_node.y
            + step_size * np.sin(np.deg2rad(current_node._theta + SLOW_LEFT) * SF) * 2
        )
        / 2
    )
    newNode._theta = current_node._theta + SLOW_LEFT
    newNode.cost_estimate = EstimateCost(newNode.x, newNode.y, end_x, end_y)

    return newNode


def MoveStraight(current_node: Node, step_size: int, end_coord: tuple, SF: int) -> Node:
    end_x, end_y = end_coord
    newNode = copy.deepcopy(current_node)

    newNode.current_index = None
    newNode.parent_index = current_node.current_index
    newNode.cost += step_size * SF

    newNode.x = (
        round(
            current_node.x
            + step_size * np.cos(np.deg2rad(current_node._theta + STRAIGHT) * SF) * 2
        )
        / 2
    )
    newNode.y = (
        round(
            current_node.y
            + step_size * np.sin(np.deg2rad(current_node._theta + STRAIGHT) * SF) * 2
        )
        / 2
    )
    newNode._theta = current_node._theta + STRAIGHT
    newNode.cost_estimate = EstimateCost(newNode.x, newNode.y, end_x, end_y)

    return newNode


def MoveSlowRight(
    current_node: Node, step_size: int, end_coord: tuple, SF: int
) -> Node:
    end_x, end_y = end_coord
    newNode = copy.deepcopy(current_node)

    newNode.current_index = None
    newNode.parent_index = current_node.current_index
    newNode.cost += step_size * SF

    newNode.x = (
        round(
            current_node.x
            + step_size * np.cos(np.deg2rad(current_node._theta + SLOW_RIGHT) * SF) * 2
        )
        / 2
    )
    newNode.y = (
        round(
            current_node.y
            + step_size * np.sin(np.deg2rad(current_node._theta + SLOW_RIGHT) * SF) * 2
        )
        / 2
    )
    newNode._theta = current_node._theta + SLOW_RIGHT
    newNode.cost_estimate = EstimateCost(newNode.x, newNode.y, end_x, end_y)

    return newNode


def MoveSharpRight(
    current_node: Node, step_size: int, end_coord: tuple, SF: int
) -> Node:
    end_x, end_y = end_coord
    newNode = copy.deepcopy(current_node)

    newNode.current_index = None
    newNode.parent_index = current_node.current_index
    newNode.cost += step_size * SF

    newNode.x = (
        round(
            current_node.x
            + step_size * np.cos(np.deg2rad(current_node._theta + SHARP_RIGHT) * SF) * 2
        )
        / 2
    )
    newNode.y = (
        round(
            current_node.y
            + step_size * np.sin(np.deg2rad(current_node._theta + SHARP_RIGHT) * SF) * 2
        )
        / 2
    )
    newNode._theta = current_node._theta + SHARP_RIGHT
    newNode.cost_estimate = EstimateCost(newNode.x, newNode.y, end_x, end_y)

    return newNode



def generate_map(SF=2):
    # EMPTY map
    map = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    def Wall_obstacles(x, y):
        return x == 0 or x == WIDTH - 1 or y == 0 or y == HEIGHT - 1
    
    ########
    ## E obstacle
    def E1_obstacle(x, y):
        return (SF * 7 <= x <= SF * 12) and (SF * 15 <= y <= SF * 40)

    def E2_obstacle(x, y):
        return (SF * 12 <= x <= SF * 20) and (SF * 15 <= y <= SF * 20)

    def E3_obstacle(x, y):
        return (SF * 12 <= x <= SF * 20) and (SF * 25 <= y <= SF * 30)

    def E4_obstacle(x, y):
        return (SF * 12 <= x <= SF * 20) and (SF * 35 <= y <= SF * 40)

    #######################################################################
    ## N obstacle
    def N1_obstacle(x, y):
        return (SF * 27 <= x <= SF * 32) and (SF * 15 <= y <= SF * 40)

    def N2_obstacle(x, y):
        return (SF * 32 <= x <= SF * 40.50) and (
            -1.82 * x + 88.74 * SF <= y <= -1.82 * x + 98.24 * SF
        )

    def N3_obstacle(x, y):
        return (SF * 40.50 <= x <= SF * 45.50) and (SF * 15 <= y <= SF * 40)

    #############################################################################
    ## P obstacle
    def P1_obstacle(x, y):
        return (SF * 52.50 <= x <= SF * 57.50) and (SF * 15 <= y <= SF * 40)

    def P2_obstacle(x, y):
        return (SF * 57.50 <= x <= SF * 63.50) and (
            (x - 57.50 * SF) ** 2 + (y - 34 * SF) ** 2 <= (6 * SF) ** 2
        )

    #############################################################################
    ## M obstacle
    def M1_obstacle(x, y):
        return (SF * 70.50 <= x <= SF * 75.50) and (SF * 15 <= y <= SF * 40)

    def M2_obstacle(x, y):
        return (
            (SF * 75.5 <= x <= SF * 86.13)
            and (-2.35 * x + 208.9 * SF <= y <= -2.35 * x + 217.43 * SF)
            and (15 * SF <= y <= 40 * SF)
        )

    def M3_obstacle(x, y):
        return (SF * 84 <= x <= SF * 88) and (SF * 15 <= y <= SF * 20)

    def M4_obstacle(x, y):
        #  2.29*x - 180.96*SF
        #  2.35*x - 195.32*SF
        return (
            (SF * 85.87 <= x <= SF * 96.50)
            and (2.35 * x - 195.32 * SF <= y <= 2.29 * x - 180.96 * SF)
            and (15 * SF <= y <= 40 * SF)
        )

    def M5_obstacle(x, y):
        return (SF * 96.50 <= x <= SF * 101.50) and (SF * 15 <= y <= SF * 40)

    #############################################################################
    # Six obstacle 1
    def Six1_obstacle1(x, y):
        return (x - 117.50 * SF) ** 2 + (y - 24 * SF) ** 2 <= (4 * SF) ** 2
        pass

    def Six1_obstacle2(x, y):
        return (x - 117.50 * SF) ** 2 + (y - 24 * SF) ** 2 <= (9 * SF) ** 2
        pass

    def Six1_obstacle3(x, y):
        return (
            ((x - 130 * SF) ** 2 + (y - 24 * SF) ** 2 >= (16.50 * SF) ** 2)
            and ((x - 130 * SF) ** 2 + (y - 24 * SF) ** 2 <= (21.50 * SF) ** 2)
            and (24 * SF <= y <= -1.462 * x + 217.38 * SF)
        )

    def Six1_obstacle4(x, y):
        return (x - 119.75 * SF) ** 2 + (y - 40 * SF) ** 2 <= (2.5 * SF) ** 2

    #############################################################################
    # Six obstacle 2
    def Six2_obstacle1(x, y):
        return (x - 142.50 * SF) ** 2 + (y - 24 * SF) ** 2 <= (4 * SF) ** 2
        pass

    def Six2_obstacle2(x, y):
        return (x - 142.50 * SF) ** 2 + (y - 24 * SF) ** 2 <= (9 * SF) ** 2

    def Six2_obstacle3(x, y):
        return (
            ((x - 155 * SF) ** 2 + (y - 24 * SF) ** 2 >= (16.50 * SF) ** 2)
            and ((x - 155 * SF) ** 2 + (y - 24 * SF) ** 2 <= (21.50 * SF) ** 2)
            and (24 * SF <= y <= -1.462 * x + 250.38 * SF)
        )

    def Six2_obstacle4(x, y):
        return (x - 145 * SF) ** 2 + (y - 40.16 * SF) ** 2 <= (2.5 * SF) ** 2

    #############################################################################
    ## One obstacel
    def One_obstacle(x, y):
        return (SF * 158.5 <= x <= SF * 163.50) and (SF * 15 <= y <= SF * 43)

    #############################################################################
    # Total Obstacle
    def E_obstacle(x, y):
        return (
            E1_obstacle(x, y)
            or E2_obstacle(x, y)
            or E3_obstacle(x, y)
            or E4_obstacle(x, y)
        )

    def N_obstacle(x, y):
        return N1_obstacle(x, y) or N2_obstacle(x, y) or N3_obstacle(x, y)

    def P_obstacle(x, y):
        return P1_obstacle(x, y) or P2_obstacle(x, y)

    def M_obstacle(x, y):
        return (
            M1_obstacle(x, y)
            or M2_obstacle(x, y)
            or M3_obstacle(x, y)
            or M4_obstacle(x, y)
            or M5_obstacle(x, y)
        )

    def Six1_obstacle(x, y):
        return (
            Six1_obstacle1(x, y)
            or Six1_obstacle2(x, y)
            or Six1_obstacle3(x, y)
            or Six1_obstacle4(x, y)
        )
        # return Six1_obstacle3(x, y)

    def Six2_obstacle(x, y):
        return (
            Six2_obstacle1(x, y)
            or Six2_obstacle2(x, y)
            or Six2_obstacle3(x, y)
            or Six2_obstacle4(x, y)
        )

    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (
                E_obstacle(x, y)
                or N_obstacle(x, y)
                or P_obstacle(x, y)
                or M_obstacle(x, y)
                or Six1_obstacle(x, y)
                or Six2_obstacle(x, y)
                or One_obstacle(x, y)
                or Wall_obstacles(x, y)
            ):
                map[y, x] = (0, 0, 0)

    map = np.flipud(map)  # flip the map according to opencv
    return map


def dilate_obstacle(obstacle_map: np.ndarray, sf_radius: int) -> np.ndarray:
    """
    Dilate the obstacle map by 2mm

    Args:
        obstacle_map (np.ndarray): original obstacle map
        SF (int, optional): scale factor. Defaults to 5.

    Returns:
        np.ndarray: dilated obstacle map
    """
    gray = cv2.cvtColor(obstacle_map, cv2.COLOR_BGR2GRAY)  # gray map

    _, object_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # object mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * sf_radius, 2 * sf_radius)
    )  # kernel

    # dilate the mask
    dilated_map = cv2.dilate(object_mask, kernel, iterations=1)
    # get original boundry
    default_map = cv2.bitwise_and(dilated_map, object_mask)

    final_map = np.ones_like(
        obstacle_map
    )  # Create a blank canvas with the same size as the original

    final_map[dilated_map == 255] = (
        0,
        0,
        255,
    )  # Boundry: Red
    final_map[dilated_map == 0] = (
        255,
        255,
        255,
    )  # Background: White
    final_map[default_map == 255] = (
        255,
        0,
        0,
    )  # Obstacle: Blue


    return final_map



def Valid_move(x: int, y: int, map_shape: tuple, obstacle_map: np.ndarray) -> bool:
    """
    check if the move is valid or not.

    Args:
        x (int): x coordinate
        y (int): y coordinate
        map_shape (tuple): map dimension
        obstacle_map (np.ndarray): obstacle map

    Returns:
        bool: True if move is valid
    """
    height, width = map_shape
    return (0 <= x < width) and (0 <= y < height) and (obstacle_map[y, x] == 255)


def Match_solution(start_coord: tuple, end_coord: tuple) -> bool:
    """
    Check if the current coordinates are within a 1.5 units radius of the solution.

    Args:
        start_coord (tuple): current coordinates
        end_coord (tuple): solution coordinates

    Returns:
        bool: True if within 1.5 units radius
    """
    distance = np.sqrt((start_coord[0] - end_coord[0]) ** 2 + (start_coord[1] - end_coord[1]) ** 2)
    return distance <= 1.5

def GeneratePath(node_dict: Dict[int, Node], solution_index: int) -> List[tuple]:
    """
    Generate path

    Args:
        node_dict (Dict[int, Node]): al the visited nodes as value, and index as key.
        solution_index (int): index of solutioon

    Returns:
        List[tuple]: _description_
    """
    path = list()

    node = node_dict[solution_index]
    # print_node(node)

    path.append((node.x, node.y))
    parent_index = node.parent_index

    print("START : Generating path")
    while parent_index != -1:
        node = node_dict.get(parent_index)
        path.append((node.x, node.y))
        parent_index = node.parent_index

    print("END : Generating path")

    return path


def Astar(
    dilate_map: np.ndarray,
    start_val: tuple,
    end_coord: tuple,
    radius: int,
    step_size: int,
) -> List[tuple]:
    """
    A* algorithm implementation.

    Args:
        dilate_map (np.ndarray): Dilated map with obstacles.
        dilate_gray_map (np.ndarray): Grayscale version of the dilated map.
        start_val (tuple): Starting coordinates (x, y, theta).
        end_coord (tuple): Goal coordinates (x, y, theta).
        SF (int): Scale factor.

    Returns:
        List[tuple]: Path from start to goal.
    """
    dilate_gray_map = cv2.cvtColor(dilate_map, cv2.COLOR_BGR2GRAY)  # convert to gray


    map_shape = dilate_map.shape[:2]
    height, width = map_shape

    index = 0
    open_list = []
    closed_list = set()
    node_dict = {}

    # Initialize the first node
    first_node = Node(index, -1, 0, EstimateCost(start_val[0], start_val[1], end_coord[0], end_coord[1]), start_val[0], start_val[1], start_val[2])
    heapq.heappush(open_list, (first_node.cost + first_node.cost_estimate, index))
    node_dict[index] = first_node

    while open_list:
        _, current_index = heapq.heappop(open_list)
        current_node = node_dict[current_index]

        # Check if the goal is reached
        if Match_solution((current_node.x, current_node.y), (end_coord[0], end_coord[1])):
            print("Solution Found!")
            return GeneratePath(node_dict, current_index)

        closed_list.add((current_node.x, current_node.y, current_node._theta))

        # Generate possible moves
        for angle in ANGLE_MOVES:
            new_theta = current_node._theta + angle
            new_x = round(current_node.x + step_size* np.cos(np.deg2rad(new_theta)))
            new_y = round(current_node.y + step_size* np.sin(np.deg2rad(new_theta)))
            #check the data type of new_x,new_y, map_shape, dilate_map
            # print(f"new_x: {type(new_x)}, new_y: {type(new_y)}, map_shape: {type(map_shape)}, dilate_map: {type(dilate_map)}")
            if not Valid_move(new_x, new_y, map_shape, dilate_gray_map):
                continue

            if (new_x, new_y, new_theta) in closed_list:
                continue

            new_cost = current_node.cost + step_size
            new_cost_estimate = EstimateCost(new_x, new_y, end_coord[0], end_coord[1])
            new_node = Node(len(node_dict), current_index, new_cost, new_cost_estimate, new_x, new_y, new_theta)

            heapq.heappush(open_list, (new_cost + new_cost_estimate, new_node.current_index))
            node_dict[new_node.current_index] = new_node

    print("No Solution Found!")
    return []
    
def test() -> np.ndarray:
    #### Test parameters
    sf_obstacle = 7
    radius = 5

    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Test", WIDTH, HEIGHT)
    # cv2.setMouseCallback("Test", draw_circle)  # Optional feature: select points directly on screen.

    obstacle_map = generate_map(sf_obstacle)  # generate map
    # obstacle_map = np.ones((50 * SF, 180 * SF, 3), dtype=np.uint8) * (255) # Blank canvas

    # dilate_map = copy.deepcopy(obstacle_map)

    dilate_map = dilate_obstacle(
        obstacle_map, radius * 3
    )  # randomly set, need to check the value.  #sf_obstacle)  # dilate map with scaled-radius

    cv2.circle(
        dilate_map, (INITIAL_COORD[0], HEIGHT - INITIAL_COORD[1]), 5, (0, 0, 0), -1
    )  # Draw circle at start coordinate
    cv2.circle(
        dilate_map, (SOLUTION_COORD[0], HEIGHT - SOLUTION_COORD[1]), 5, (0, 0, 0), -1
    )  # Draw circle at end coordinate

    # dilate_gray_map = cv2.cvtColor(dilate_map, cv2.COLOR_BGR2GRAY)  # convert to gray

    # cv2.imshow("Screen", obstacle_map)  # show Map
    cv2.imshow("Test", dilate_map)  # show Dilate-Map
    cv2.waitKey(0)  # Press Keyboard key to Exit

    # plt.imshow(dilate_map)
    # plt.show()
    return dilate_map

def visualize_and_save_path(
    explored_nodes: List[tuple],
    optimal_path: List[tuple],
    map_image: np.ndarray,
    output_video_path: str = "astar_visualization.mp4",
    fps: int = 10,
) -> None:
    """
    Visualize the exploration and optimal path generation and save it as a video.

    Args:
        explored_nodes (List[tuple]): List of nodes explored during A*.
        optimal_path (List[tuple]): List of nodes in the optimal path.
        map_image (np.ndarray): The map image to overlay the visualization.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # Create a copy of the map to draw on
    visualization_map = map_image.copy()

    # Define video writer
    height, width, _ = visualization_map.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Visualize explored nodes
    for node in explored_nodes:
        x, y = node
        cv2.circle(visualization_map, (x, height - y), 2, (0, 255, 0), -1)  # Green for exploration
        video_writer.write(visualization_map)

    # Visualize optimal path
    for i in range(len(optimal_path) - 1):
        x1, y1 = optimal_path[i]
        x2, y2 = optimal_path[i + 1]
        cv2.line(
            visualization_map,
            (x1, height - y1),
            (x2, height - y2),
            (0, 0, 255),
            2,
        )  # Red for optimal path
        video_writer.write(visualization_map)

    # Release the video writer
    video_writer.release()
    print(f"Visualization saved as {output_video_path}")

def get_user_input_clearance() -> int:
    """
    Get the clearance from the user

    Returns:
        int
    """
    # Get clearance
    while True:
        try:
            clearance = int(input("Clearance (in units, >= 0) (tip: try 5): "))
            if clearance >= 0:
                break
            else:
                print("Invalid input. Clearance must be >= 0.")
        except:
            print("Invalid input. Please enter an integer.")

    return clearance

def get_user_input_robot_radius() -> int:
    """
    Get the robot radius from the user

    Returns:
        int
    """
    while True:
        try:
            radius = int(input("Robot radius (in units, >= 0) (tip: try 5): "))
            if radius >= 0:
                break
            else:
                print("Invalid input. Radius must be >= 0.")
        except:
            print("Invalid input. Please enter an integer.")

    return radius

def get_user_input_step_size():
    """
    Get the step size from the user 

    Returns:
        int
    """
    # Get step size
    while True:
        try:
            step_size = int(input("Step size (1 <= step size <= 10) (tip: try 5): "))
            if 1 <= step_size <= 10:
                break
            else:
                print("Invalid input. Step size must be between 1 and 10.")
        except:
            print("Invalid input. Please enter an integer.")
    
    return step_size

def get_user_input_start_pose(grayscale_map):
    """
    Get a valid start pose from the user

    Params:
        grascape_map (np.ndarray): Requires a grayscale map of the workspace 

    Returns:
        Tuple[int, int, int]

    """
    map_shape = (grayscale_map.shape[0], grayscale_map.shape[1])

    # Get start point coordinates
    while True:
        try:
            start_x = int(input("Start X (0 <= X <= 600) (tip: try 10): "))
            start_y = int(input("Start Y (0 <= Y <= 250) (tip: try 10): "))
            start_theta = int(input("Start θ (multiple of 30, e.g., -60, -30, 0, 30, 60) (tip: try 0): "))
            if Valid_move(start_x, start_y, map_shape, grayscale_map):
                break
            else:
                print("Invalid input. Please enter valid start coordinates.")
        except:
            print("Invalid input. Please enter integers.")

    return (start_x, start_y, start_theta)

def get_user_input_goal_pose(grayscale_map) -> int:
    """
    Get a valid goal pose from the user

    Params:
        grayscale_map (np.ndarray): Requires a grayscale map of the workspace 

    Returns:
        Tuple[int, int, int]

    """
    map_shape = (grayscale_map.shape[0], grayscale_map.shape[1])

    # Get goal point coordinates
    while True:
        try:
            goal_x = int(input("goal X (0 <= X <= 600) (tip: try 400): "))
            goal_y = int(input("goal Y (0 <= Y <= 250) (tip: try 200): "))
            goal_theta = int(input("goal θ (multiple of 30, e.g., -60, -30, 0, 30, 60) (tip: try 0): "))
            if Valid_move(goal_x, goal_y, map_shape, grayscale_map):
                break
            else:
                print("Invalid input. Please enter valid goal coordinates.")
        except:
            print("Invalid input. Please enter integers.")

    return (goal_x, goal_y, goal_theta)

def Collision_between_points(current_x, current_y, neighbor_x, neighbor_y, grayscale_map, step_size):
    """
    Returns true if there is a collision between current and neighbor, false if not. Requires the grayscale map and the step size for the search.
    """
    for baby_step in range(1, step_size):
        dx = neighbor_x - current_x
        dy = neighbor_y - current_y 
        distance = (dx**2 + dy**2)**0.5
        unit_x = dx / distance
        unit_y = dy / distance
        test_x = current_x + unit_x * baby_step
        test_y = current_y + unit_y * baby_step 
        if not Valid_move(
            test_x, test_y, 
            (grayscale_map.shape[0], grayscale_map.shape[1]),
            grayscale_map):
            return True 
    return False              

def main() -> None:
    """
    main function
    """

    # 1. Prompt user for a clearance
    clearance = get_user_input_clearance()
    
    # 2. Prompt user for a robot radius
    robot_radius = get_user_input_robot_radius()

    # 3. Prompt user for a step size
    step_size = get_user_input_step_size()

    # 4. Construct a grayscale map of the workspace
    # 4.1 Construct a BGR map
    sf_obstacle = 7
    bgr_map = generate_map(sf_obstacle) 
    # 4.2 Dilate the map with a safety margin
    safety_margin = max(robot_radius, clearance)
    safety_margin_map = dilate_obstacle(
        bgr_map, safety_margin
    )
    # 4.3 Convert the safety margin map to a grayscale map
    grayscale_map = cv2.cvtColor(safety_margin_map, cv2.COLOR_BGR2GRAY)

    # 6. Prompt user for a VALID start pose
    start_pose = get_user_input_start_pose(grayscale_map)

    # 7. Prompt user for a VALID goal pose
    goal_pose = get_user_input_goal_pose(grayscale_map)

    # 8. Run the A* algorithm
    path = Astar(safety_margin_map, start_pose, goal_pose, safety_margin, step_size)
    

if __name__ == "__main__":
    main()
