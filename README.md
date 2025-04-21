amg8833_i2c.py forked from https://github.com/makerportal/AMG8833_IR_cam (special thanks for Josh-Hrisko)  
this codebase is largely attributed to the work of our very own [ffluryy](https://github.com/orgs/cde2310grp6/people/ffluryy)

# casualty_location
This repo consists of three different crucial nodes/executables
 - casualty_locate
 - casualty_save
 - ir_pub


## casualty_locate
This node is based off of [AniArka's frontier exploration algorithm](https://github.com/AniArka/Autonomous-Explorer-and-Mapper-ros2-nav2) and uses custom map viewers made by [ffluryy](https://github.com/orgs/cde2310grp6/people/ffluryy) to conduct 'frontier based exploration' to paint the walls of the maze based on the temperature detected at the walls.
It receives a service call from mission_control to start operation, and when it is done, sends a callback via a ros topic.
Hence it can be manually called via command line for debugging purposes using
```
ros2 service call /casualty_state custom_msg_srv/srv/StartCasualtyService "{state: "LOCATE"}"
```


## casualty_save
This node receives casualty positional data from casualty_locate, then commands the Turtlebot to go to each casualty   
Upon reaching the casualty, casualty_save will
 1. call aligner_node
 2. call launcher service
Similar to casualty_locate, this node may be called via command line using
```
ros2 service call /casualty_state custom_msg_srv/srv/StartCasualtyService "{state: "SAVE"}"
```


## ir_pub
This node is essentially our amg8833 driver, which acts as a ROS2 publisher of the IR data for casualty_locate and aligner_node to use  
forked from https://github.com/makerportal/AMG8833_IR_cam (special thanks for Josh-Hrisko)
