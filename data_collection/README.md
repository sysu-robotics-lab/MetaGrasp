### Collect new data in V-REP

To collect new grasp data in V-REP, we you should first run the scene (i.e., run  ```ur.ttt```) in simulator, 
then control the actions using the remote python api.

#### Step 1: Launch the V-REP simulator and load scene.
You first need to have access to the V-REP 3.4 simulator, the software is available at [here](http://www.coppeliarobotics.com/).
Then run the script to launch V-REP:
```bash
cd path_to_vrep
./vrep
```
Load the scene ```ur.ttt``` into V-REP, the scene file and object models are available at 
[here](https://drive.google.com/open?id=1YQfju1x6_Kj7Hc0hPD154YmPWP3SbZ5h)

#### Step 2: run control script to collect data.
```bash
python data_collection/corrective_grasp_trials.py \ 
--ip 127.0.0.1 \  # ip address to the vrep simulator.
--port 19997 \  # port to the vrep simulator.
--obj_id obj0000 \  # object handle in vrep.
--num_grasp 200 \  # the number of grasp trails.
--num_repeat 1 \  # the number of repeat time if the gripper successfully grasp an object.
--output data/3dnet  # directory to save data.
```

If you want to collect data which is not guided by antipodal rule, you can run the script:
```bash
python data_collection/grasp_trials_without_rule.py \ 
--ip 127.0.0.1 \  # ip address to the vrep simulator.
--port 19997 \  # port to the vrep simulator.
--obj_id obj0000 \  # object handle in vrep.
--num_grasp 200 \  # the number of grasp trails.
--output data/3dnet  # directory to save data.
```

### Data structure
A grasp sample consists of an RGB image of the whole workspace, 
the grasp label that illustrates which locations are graspable or not 
as well as how much degrees the gripper should rotate, 
and object coordinates that indicate the whole object pixel locations in the RGB image.

### Contents of directories
* **3dnet**
    * **0000**
        * **color**: The raw RGB images obtained from vision sensor.
            * **000000.png**
            * **000001_0000.png**
            * ......
        * **depth**: The raw depth images obtained from vision sensor.
            * **000000.png**
            * **000001_0000.png**
            * ......
        * **height_map_color**: The cropped RGB images that only contain the information of workspace.
            * **000000.png**
            * **000001_0000.png**
            * ......
        * **height_map_depth**: The cropped depth images that only contain the information of workspace.
            * **000000.png**
            * **000001_0000.png**
            * ......
        * **label**
            * **000000.bad.txt**: This file contains the grasp points in image space and corresponding grasp angles.
            * **000000.object_points.txt**: This file contains coordinates that belong to object.
            * **000000.png**: The visualization of the grasp angles.
            * **000001_0000.good.txt**
            * **000001_0000.object_points.txt**
            * **000001_0000.png**
            * ......
        * **background_color.png**
        * **background_depth.png**
        * **crop_background_color.png**
        * **crop_background_depth.png**
        * **file_name.txt**
    * **0001**
        * ......
    * ......
    
        