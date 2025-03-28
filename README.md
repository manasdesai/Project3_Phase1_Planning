# ENPM661 (Project3_Phase1_Planning)
# A* Algorithm

## Team Members 
1. Manas Desai () ()
2. Ronen Aniti (112095116) (raniti)
3. Vaibhav Yuwaraj Shende (121206817) (svaibhav)



This is a Python implementation of a finding a path between initial coordinate and start coordinate through the obstacle canvas using the A* algorithm. 

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
Once the Path is found, the following files will be created in the same directory:

* astar_animation.gif : Contains the gif of how the path is traversed.
* astar_animation.mp4 : Contains the video of how the path is traversed.


### How It Works
#TODO