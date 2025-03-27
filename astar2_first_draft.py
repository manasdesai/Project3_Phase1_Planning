from enum import Enum 
import numpy as np 
import matplotlib.pyplot as plt
from queue import PriorityQueue 
from matplotlib.animation import FuncAnimation
import time
import pdb

class Action(Enum):
    def __init__(self, r, theta):
        """Represents unit vector version of each action"""
        self.r = r 
        self.theta = theta 
    LEFT60 = (1, -60)
    LEFT30 = (1, -30)
    STRAIGHT = (1, 0)
    RIGHT30 = (1, 30)
    RIGHT60 = (1, 60)
    
def coordinate_transformation(standard_frame, spatial_resolution, angular_resolution):
    return (int(standard_frame[0] * spatial_resolution),
            int(standard_frame[1] * spatial_resolution), 
            int(standard_frame[2] / angular_resolution )
    )

def coordinate_transformation_2dof(standard_frame_2d, spatial_resolution):
    return (int(standard_frame_2d[0] * spatial_resolution),
            int(standard_frame_2d[1] * spatial_resolution)
    )

def coordinate_transformation_inverse(grid_frame, spatial_resolution, angular_resolution):
    return (int(grid_frame[0] / spatial_resolution), 
            int(grid_frame[1] / spatial_resolution), 
            int(grid_frame[2] * angular_resolution)
    )

def coordinate_transformation_inverse_2dof(grid_frame, spatial_resolution):
    return (int(grid_frame[0] / spatial_resolution), 
            int(grid_frame[1] / spatial_resolution)
    )

def angle_to_index(angle_deg, angular_resolution):
    return int(angle_deg / angular_resolution)

def index_to_angle(angle_index, angular_resolution):
    return int(angle_index * angular_resolution)

def collision(x, y, scale, safety):
    regions = []

    # Wall buffer region
    regions.append(
        (x - safety <= 0) or
        (x - 180*scale + safety >= 0) or
        (y - safety <= 0) or
        (y - 50*scale + safety >= 0)
    )

    # E
    ##########################
    # E, vertical rectangle primitive
    regions.append(
        (x - 20*scale + safety >= 0) and
        (x - 25*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and   
        (y - 35*scale - safety <= 0)
    )
    
    
    # E, bottom rectangle primitive
    regions.append(
        (x - 20*scale + safety >= 0) and
        (x - 33*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and 
        (y - 15*scale - safety <= 0)

    )
    
    # E, middle rectangle primitive
    regions.append(
        (x - 20*scale + safety >= 0) and
        (x - 33*scale - safety <= 0) and
        (y - 20*scale + safety >= 0) and 
        (y - 25*scale - safety <= 0)

    )
    
    # E, top rectangle primitive
    regions.append(
        (x - 20*scale + safety >= 0) and
        (x - 33*scale - safety <= 0) and
        (y - 30*scale + safety >= 0) and 
        (y - 35*scale - safety <= 0)

    )
    
    # N 
    ##########################
    # N, left vertical rectangle primitive
    regions.append(
        (x - 43*scale + safety >= 0) and
        (x - 48*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and   
        (y - 35*scale - safety <= 0)      
    )
    
    # N, middle in between two segment region
    regions.append(
        (y + 3*x - 179*scale - safety * np.sqrt(10) <= 0) and
        (y + 3*x - 169*scale + safety * np.sqrt(10) >= 0) and 
        (x - 48*scale + safety >= 0) and
        (x - 53*scale - safety <= 0) and 
        (y - 10*scale + safety >= 0) and 
        (y - 35*scale - safety <= 0)      

    )

    # N, right vertical rectangle primitive
    regions.append(
        (x - 53*scale + safety >= 0) and
        (x - 58*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and   
        (y - 35*scale - safety <= 0)      
    )

    # P
    ###############
    # P, left vertical bar
    regions.append(
        (x - 68*scale + safety >= 0) and
        (x - 73*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and   
        (y - 35*scale - safety <= 0)      
    )

    # P, semi-circular region
    regions.append(
        ( (x - 73*scale)**2 + (y - 28.75*scale)**2 - (6.25*scale + safety)**2 <= 0) and
        (x - 73*scale - safety >= 0)
    )
    
     # M
    ##################
    # M, first vertical bar (left vertical)
    regions.append(
        (x - 85*scale + safety >= 0) and
        (x - 90*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and   
        (y - 35*scale - safety <= 0)
    )
    # M, left diagonal region 
    regions.append(
        (5*x + y - 485*scale - safety * np.sqrt(26) <= 0) and
        (5*x + y - 483*scale + safety * np.sqrt(26) >= 0) and
        (x - 90*scale + safety >= 0) and (x - 95*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and (y - 35*scale - safety <= 0)
    )
    # M, right diagonal region
    regions.append(
        (5*x - y - 465*scale - safety * np.sqrt(26) <= 0) and
        (5*x - y - 463*scale + safety * np.sqrt(26) >= 0) and
        (x - 95*scale + safety >= 0) and (x - 100*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and (y - 35*scale - safety <= 0)
    )
    # M, second vertical bar (right vertical)
    regions.append(
        (x - 100*scale + safety >= 0) and
        (x - 105*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and
        (y - 35*scale - safety <= 0)
    )
    
    # 6 
    ##################
    # 6, circle primitive
    regions.append(
        (((x - 120*scale)**2 + (y - 17.5*scale)**2) <= (7.5*scale + safety)**2)
    )
    # 6, vertical rectangle on top of the circle
    regions.append(
        (x - 112.5*scale + safety >= 0) and
        (x - 117.5*scale - safety <= 0) and
        (y - 17.5*scale + safety >= 0) and
        (y - 35*scale - safety <= 0)
    )
    # 6, horizontal rectangle attached to the right of the vertical rectangle
    regions.append(
        (x - 117.5*scale + safety >= 0) and
        (x - 123*scale - safety <= 0) and
        (y - 33*scale + safety >= 0) and
        (y - 35*scale - safety <= 0)
    )
    
    # 6 (second 6)
    ##################
    # 6 (second 6), circle primitive
    regions.append(
        (((x - 139.5*scale)**2 + (y - 17.5*scale)**2) <= (7.5*scale + safety)**2)
    )
    # 6 (second 6), vertical rectangle
    regions.append(
        (x - 132*scale + safety >= 0) and
        (x - 137*scale - safety <= 0) and
        (y - 17.5*scale + safety >= 0) and
        (y - 35*scale - safety <= 0)
    )
    # 6 (second 6), horizontal rectangle 
    regions.append(
        (x - 137*scale + safety >= 0) and
        (x - 143*scale - safety <= 0) and
        (y - 33*scale + safety >= 0) and
        (y - 35*scale - safety <= 0)
    )
    
    # 1
    ###############
    # 1, rectangle primitive
    regions.append(
        (x - 155*scale + safety >= 0) and
        (x - 160*scale - safety <= 0) and
        (y - 10*scale + safety >= 0) and
        (y - 35*scale - safety <= 0)
    )
    
    return any(regions)

def generate_occupancy_grid(workspace_dimension, spatial_resolution, scale, safety): 
    occupancy_grid_shape = (
        int(workspace_dimension[0] * spatial_resolution) + 1, 
        int(workspace_dimension[1] * spatial_resolution) + 1,
    )

    occupancy_grid = np.zeros(occupancy_grid_shape, dtype=bool)

    for x in range(occupancy_grid_shape[0]):
        for y in range(occupancy_grid_shape[1]):
            grid_coord = (x, y)
            frame_coord = coordinate_transformation_inverse_2dof(grid_coord, spatial_resolution)
            if collision(frame_coord[0], frame_coord[1], scale, safety):
                occupancy_grid[x, y] = True 
    
    return occupancy_grid

def generate_occupancy_grid_for_plotting(workspace_dimension, spatial_resolution, scale, safety): 
    occupancy_grid_shape = (
        int(workspace_dimension[0] * spatial_resolution) + 1, 
        int(workspace_dimension[1] * spatial_resolution) + 1,
    )

    occupancy_grid_for_plotting = np.zeros(occupancy_grid_shape, dtype=int)

    for x in range(occupancy_grid_shape[0]):
        for y in range(occupancy_grid_shape[1]):
            grid_coord = (x, y)
            frame_coord = coordinate_transformation_inverse_2dof(grid_coord, spatial_resolution)
            if collision(frame_coord[0], frame_coord[1], scale, 0.0):
                occupancy_grid_for_plotting[x, y] = 2  # Obstacle region but not in padding
            elif collision(frame_coord[0], frame_coord[1], scale, safety):
                occupancy_grid_for_plotting[x, y] = 1  # Padded region but not in obstacle region

    return occupancy_grid_for_plotting

def euclidean_ground_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def find_valid_neighbors(occupancy_grid, current, step_size, spatial_resolution, angular_resolution):
    # The output of this must be two lists in grid units only
    
    # Define a list of valid actions
    actions_list = [Action.LEFT60, Action.LEFT30, Action.STRAIGHT, Action.RIGHT30, Action.RIGHT60]

    # Define lists for valid neighbors and distances
    valid_neighbors_list = [] # grid frame
    distances_list = [] # grid distances

    # Convert to standard frame (mm, mm, deg) for collision checking
    current_standard_frame = coordinate_transformation_inverse(current, spatial_resolution, angular_resolution)

    for action in actions_list:
        
        # Get the dimensionless radius of expansion
        r = action.r 

        # The action angular displacement in degrees: 
        delta_theta = action.theta

        # The new x coordinate in mm
        new_node_standard_x = current_standard_frame[0] + r * step_size * np.cos(
            np.deg2rad(current_standard_frame[2] + delta_theta)
        )

        # The new y coordinate in mm
        new_node_standard_y = current_standard_frame[1] + r * step_size * np.sin(
            np.deg2rad(current_standard_frame[2] + delta_theta)
        )

        # The new theta coordinate in degrees
        new_node_standard_theta = current_standard_frame[2] + delta_theta 
        new_node_standard_theta = new_node_standard_theta % 360 # Wrap around 360 degrees

        # The new node in (mm, mm, deg) format
        new_node_standard = (
            new_node_standard_x, 
            new_node_standard_y, 
            new_node_standard_theta
        )
        
        # The new node in grid units
        new_node_grid = coordinate_transformation(new_node_standard, spatial_resolution, angular_resolution)
        
        # If the grid coordinate of the new node is out of bounds, skip
        if new_node_grid[0] < 0 or new_node_grid[0] >= occupancy_grid.shape[0]:
            continue

        if new_node_grid[1] < 0 or new_node_grid[1] >= occupancy_grid.shape[1]:
            continue 
        
        # Mark the new node as valid if its in bounds and not an obstacle
        if not occupancy_grid[new_node_grid[0], new_node_grid[1]]:
            valid_neighbors_list.append(new_node_grid)
            distances_list.append(r * step_size * spatial_resolution) # The cost is measured in grid units

    return valid_neighbors_list, distances_list

            



        
def backtrack(predecessors, start, goal):
    """
    Simple backtracking routine to extract the path from the branching dictionary
    """
    path = [goal]
    current = goal
    while predecessors[current] != None:
        parent = predecessors[current]
        path.append(parent)
        current = parent
    return path[::-1]



def astar(occupancy_grid, start, goal, scale_factor, safety_margin, step_size, spatial_resolution, angular_resolution, h=euclidean_ground_distance, logging=False): 

    # *Assume start and goal are in grid units*

    # Convert the start node to standard coordinate from for validity check
    start_standard_frame = coordinate_transformation_inverse(start, spatial_resolution, angular_resolution)

    # Convert the goal node to standard coordinate frame for validity check
    goal_standard_frame = coordinate_transformation_inverse(goal, spatial_resolution, angular_resolution)
    
    # Check the validity of the start node
    if collision(
        start_standard_frame[0], 
        start_standard_frame[1], 
        scale_factor, 
        safety_margin
    ):
        print("The start node is invalid")
        return None, None
    
    # Check the validity of the goal node
    if collision(
        goal_standard_frame[0], 
        goal_standard_frame[1], 
        scale_factor, 
        safety_margin

    ):
        print("The goal node is invalid")
        return None, None 
    
    # Construct lists to store open set and closed set history for animation
    if logging:
        search_array_history = []

    # Initialize A* data structures
    parents = dict()

    # Construct an array data structure to combine open set and closed set information. 
    # Unvisited and unopened = 0
    # Visited = 1
    # Open = 2
    search_array_shape = (
        occupancy_grid.shape[0], 
        occupancy_grid.shape[1], 
        int(360 / angular_resolution)
    )
    search_array = np.zeros(search_array_shape, dtype=np.int8)

    g_scores = dict()
    queue = PriorityQueue()
    goal_is_found = False
    
    # Handle the start node
    search_array[start[0], start[1], start[2]] = 2 # 2 -> open
    parents[start] = None 
    g_scores[start] = 0.0
    f_score_start = g_scores[start] + h(start, goal)
    queue.put((f_score_start, start))
    
    # Logging
    iteration = 1

    # Begin the A* main loop
    while not queue.empty():

        
        iteration += 1
        if logging and (iteration % 500 == 0):
            print(iteration)
            search_array_history.append(search_array)
        
        # POP the most promising node
        f_current, current = queue.get()

        # ASSUME that some queued nodes may be visited nodes. Skip them.
        if search_array[current[0], current[1], current[2]] == 1: # 1 -> visited
            continue

        # ONLY proceed to visit and process unvisited nodes:
        
        # Mark the current node as visited
        search_array[current[0], current[1], current[2]] = 1 # 1-> visited

        # Stop search if the goal is found
        if current[:2] == goal[:2]:
            print("A* has found the goal")
            found_goal = current # May have different orientation, but that's OK
            goal_is_found = True 
            break

        # If this current node is NOT the goal node:
        
        # Expand the neighbors of this current node:
        valid_neighbors_list, distances_list = find_valid_neighbors(occupancy_grid, current, step_size, spatial_resolution, angular_resolution)
        for i, neighbor in enumerate(valid_neighbors_list):

            # ASSUME that some NEIGHBORS may be VISITED already. Skip them.
            if search_array[neighbor[0], neighbor[1], neighbor[2]] == 1: # 1 -> visited
                continue


            # ASSUME that some NEIGHBORS may already be in the OPEN set. Process them, but IF AND ONLY IF a better partial plan would result.
            if search_array[neighbor[0], neighbor[1], neighbor[2]] == 2: #2 -> open
                g_current = g_scores[current] # g-score of current node
                g_tentative = g_current + distances_list[i]
                if g_tentative < g_scores[neighbor]:
                    g_scores[neighbor] = g_tentative
                    parents[neighbor] = current
                    f_score_neighbor = g_tentative + h(neighbor, goal)
                    queue.put((f_score_neighbor, neighbor))
            
            # ASSUME that some NEIGHBORS may be NOT in the OPEN SET and NOT in the CLOSED SET.
            if search_array[neighbor[0], neighbor[1], neighbor[2]] == 0: #0 -> unvisited and unseen
                search_array[neighbor[0], neighbor[1], neighbor[2]] = 2 # 2 -> open
                g_tentative = g_scores[current] + distances_list[i]
                parents[neighbor] = current 
                g_scores[neighbor] = g_tentative
                f_score_neighbor = g_tentative + h(neighbor, goal)
                queue.put((f_score_neighbor, neighbor))
    
    if goal_is_found:
        
        cost = g_scores[found_goal] # cost in grid units
        
        # Convert cost to mm of travel distance in the standard frame
        cost = cost / spatial_resolution

        path = backtrack(parents, start, found_goal)

        if logging:
            animate_search(occupancy_grid, search_array_history, path)
        return path, cost

    return None, None

if __name__ == "__main__":

    # This project is an extension of Project 2.
    # The obstacles from Project 2 will be used.
    # Recall the exact x, y dimensions of the workspace from Project 2
    # However they will be dilated by a fixed factor.
    ORIGINAL_WORKSPACE_DIMENSION = (180, 50)
    SCALE_FACTOR = 3.0

    # The resulting workspace dimension is this:
    NEW_WORKSPACE_DIMENSION = (
        ORIGINAL_WORKSPACE_DIMENSION[0] * SCALE_FACTOR,
        ORIGINAL_WORKSPACE_DIMENSION[1] * SCALE_FACTOR
    )

    # The Project 2 specifications also call for a spatial resolution of 0.5mm/ unit (2 units/mm) and for an angular resolution of 30 degrees per unit. So I am setting that here.
    # The coordinates of the workspace are assumed to be
    # (mm, mm, degrees) for (x, y, theta).
    SPATIAL_RESOLUTION = 2.0
    ANGULAR_RESOLUTION = 30.0

    # Define the step size for the branch expansion process for the A* component of the pathfinding operation. This will be in mm. 
    step_size = float(input("Enter the step size (mm): "))
    
    # Define a safety margin around obstacles
    safety_margin = float(input("Enter a safety margin-AKA the radius of the robot (mm): "))

    # Define the start pose (mm, mm, deg)
    x_start, y_start, theta_start = map(float, input("Enter the (mm, mm, deg) coordinates for the start pose, separated by only spaces:" ).split())
    start = (x_start, y_start, theta_start)

    # Define the goal pose (mm, mm, deg)
    x_goal, y_goal, theta_goal = map(float, input("Enter the (mm, mm, deg) coordinates for the goal pose, separated by only spaces:" ).split())
    goal = (x_goal, y_goal, theta_goal)
    
    # Convert the start and goal poses to the grid frame
    start = coordinate_transformation(start, SPATIAL_RESOLUTION, ANGULAR_RESOLUTION)

    goal = coordinate_transformation(goal, SPATIAL_RESOLUTION, ANGULAR_RESOLUTION)

    # Define an occupancy grid for Project 3 having the structure:
    # grid[i][j -> i -> units of mm * spatial_resolution
    #              j -> units of mm * spatial_resolution
    # The grid will be a Numpy bool array and will be constructed using 
    # a `collision` function having the hard-coded obstacle information 
    # specified in Project 2
    print("Constructing the occupancy grid...")
    occupancy_grid = generate_occupancy_grid(NEW_WORKSPACE_DIMENSION, SPATIAL_RESOLUTION, SCALE_FACTOR, safety_margin)
    print("The occupancy grid construction is complete.")
    
    # Used for only plotting / documentation - not necessary for pathfinding
    print("Constructing an occupancy grid for plotting to highlight the padded region...")
    #occupancy_grid_for_plotting = generate_occupancy_grid_for_plotting(NEW_WORKSPACE_DIMENSION, SPATIAL_RESOLUTION, SCALE_FACTOR, safety_margin)
    print("The occupancy grid for plotting is complete.")

    # Executes A* search, return path and cost
    print("Executing A* search from start to goal.")
    path, cost = astar(occupancy_grid, start, goal, SCALE_FACTOR, safety_margin, step_size, SPATIAL_RESOLUTION, ANGULAR_RESOLUTION, logging=False)
    if path:
        print(f"The path is as follows: {path}")
        print(f"The path length in mm is {cost / SPATIAL_RESOLUTION}")

    
    # Assume you already generated occupancy_grid_for_plotting and obtained a path
    occupancy_grid_for_plotting = generate_occupancy_grid_for_plotting(NEW_WORKSPACE_DIMENSION, SPATIAL_RESOLUTION, SCALE_FACTOR, safety_margin)

    plt.imshow(occupancy_grid_for_plotting.T, origin="lower", cmap="inferno", extent=[0, occupancy_grid.shape[0], 0, occupancy_grid.shape[1]])
    plt.title("Workspace with Path")

    # WORKING ON THIS NOW:
    xs_path = [node[0] for node in path]
    ys_path = [node[1] for node in path]
    us_path = [np.cos(np.deg2rad(node[2] * ANGULAR_RESOLUTION)) for node in path]
    vs_path = [np.sin(np.deg2rad(node[2] * ANGULAR_RESOLUTION)) for node in path]

    plt.quiver(xs_path, ys_path, us_path, vs_path, color='r', width=0.0002)
    plt.show()

    # Animates the search process
    # TODO

    # Converts the path to (mm, mm, deg) format and converts the cost to units of mm rather than grid units. 
    ### TODO

    ## Prints the path and cost to the console in both formats

    #plt.imshow(occupancy_grid_for_plotting.T, origin="lower",     extent=[0, occupancy_grid.shape[0], 0, occupancy_grid.shape[1]])
    #plt.title("Workspace in Grid Units -- 2 Grid Units per 1 mm")
    #plt.show()

