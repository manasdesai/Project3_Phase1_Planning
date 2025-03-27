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


def generate_map(SF=2):
    # EMPTY map
    map = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

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
            ):
                map[y, x] = (0, 0, 0)

    map = np.flipud(map)  # flip the map according to opencv
    return map


def dilate_obstacle(obstacle_map: np.ndarray, sf_radius: int = 5) -> np.ndarray:
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


def test() -> None:
    #### Test parameters
    sf_obstacle = 7
    radius = 5
    clearance = 5
    step_size = 1

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

    dilate_gray_map = cv2.cvtColor(dilate_map, cv2.COLOR_BGR2GRAY)  # convert to gray

    # cv2.imshow("Screen", obstacle_map)  # show Map
    cv2.imshow("Test", dilate_map)  # show Dilate-Map
    cv2.waitKey(0)  # Press Keyboard key to Exit

    # plt.imshow(dilate_map)
    # plt.show()


def main() -> None:
    """
    main function
    """
    sf_obstacle = 6
    # Create a window
    cv2.namedWindow("Screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Screen", WIDTH, HEIGHT)

    obstacle_map = generate_map(sf_obstacle)  # generate map

    dilate_map = dilate_obstacle(obstacle_map, sf_obstacle)  # dilate map
    # dilate_gray_map = cv2.cvtColor(dilate_map, cv2.COLOR_BGR2GRAY)  # convert to gray

    cv2.imshow("Screen", dilate_map)  # show Map
    cv2.waitKey(0)


if __name__ == "__main__":
    # main()
    test()
