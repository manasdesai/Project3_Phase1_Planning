# ENPM661 (Project3_Phase1_Planning)
# A* Algorithm

## Team Members 
1. Manas Desai () ()
2. Ronen Aniti (112095116) (raniti)
3. Vaibhav Yuwaraj Shende (121206817) (svaibhav)



This is a Python implementation of a finding a path between initial coordinate and start coordinate through the obstacle canvas using the A* algorithm. 

![Animation Result](/astar_animation.gif)
**Figure 1.** This animation, produced with `a_star_manas_ronen_vaibhav.py`, shows how A* traverses the configuration space from start to goal. 

## Requirements

The following libraries are required to run the script:

- Python 3.x
  - `numpy`
  - `matplotlib`
  - `time`
  - `pdb`
  - `queue`
  - `enum`
  - `typing`

Note: Make sure these libraries are installed in your system.

## Running the Script
1. Clone the repository or download the script to your local machine.
2. Open a terminal and navigate to the directory where the script is located.
3. Run the script by executing the following command:

```
python3 a_star_manas_ronen_vaibhav.py
```

#### Input:

```
--> Example Input:

Enter the step size (mm): 5
Enter a safety margin around obstacles (mm): 5
Enter the robot radius (mm): 5
Enter the (mm, mm, deg) coordinates for the start pose, separated by only spaces:50 100 180
Enter the (mm, mm, deg) coordinates for the goal pose, separated by only spaces:400 25 18
```

#### Output:

If a path is found, it will be printed to the console, and a static Matplotlib plot of the result will be shown in a separate window. The execution time for the search will also be printed to the console. 

If a path is found and the script is run with `LOGGING` set to `True`, the following files will be created in the same directory:

* astar_animation.gif : Contains the gif of how the path is traversed.
* astar_animation.mp4 : Contains the video of how the path is traversed.

If no path is found, the phrase "No path found" will be printed to the console, along with the execution time for the search. 

### How It Works

The script `a_star_manas_ronen_vaibhav.py` works by following these steps. 

1. Prompt the user for necessary pathfinding parameters, including the desired safety margin around obstacles, the robot radius, and the start and goal poses. 
2. Generate a grid-based map of free-space and obstacle-space, with obstacle-space being adjusted to account for a safety margin around each obstacle polygon. 
3. Search the grid-based map, from start pose to goal pose, with the A* algorithm, considering a discrete action space: 

```
[LEFT60, LEFT30, STRAIGHT, RIGHT30, RIGHT60]
```
4. Print the found path to the console, along with the found path length, and also print the A* execution time to the console. 
5. Enable the user to generate Matplotlib animations of the A* search process by setting global variable `LOGGING` to `True`.    