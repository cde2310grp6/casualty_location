import rclpy
from rclpy.node import Node

from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion

# for self.robot_position
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

import math

from time import sleep

# for mission control
from custom_msg_srv.msg import CasualtySaveStatus, CasualtyLocateStatus
from custom_msg_srv.srv import StartCasualtyService

# for launcher service
from std_srvs.srv import Trigger 

# for casualty locations
from custom_msg_srv.msg import ArrayCasualtyPos

import casualty_location.rviz_marker as rviz_marker

# for map res, for rviz marker
from nav_msgs.msg import OccupancyGrid

# for spin to face target
from nav2_msgs.action import Spin
from rclpy.duration import Duration

import numpy as np


DIST_TO_CASUALTY = 3.0  # Distance to casualty before stopping to fire



class CasualtySaver(Node):
    def __init__(self):
        super().__init__('casualty_saver')

        # to navigate to the casualties
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # sub to launcher service
        self.launcher_service = self.create_client(Trigger, 'launch_ball')

        # sub to casualty locations
        self.casualty_loc = self.create_subscription(ArrayCasualtyPos, 'casualty_locations', self.casualty_callback, 10)
        self.waypoints = []

        # service for mission_control to initiate casualty_location
        self.start_casualty_service = self.create_service(StartCasualtyService, 'casualty_state', self.start_casualty_callback)
        self.mission_state = "STOPPED"

        # topic to tell mission_control when casualty_location complete
        self.cas_save_pub = self.create_publisher(CasualtySaveStatus, 'casualty_saved', 10)

        # get self.robot_position
        # self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # change to pose instead of odom
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)


        # for spin_face_target
        self.spin_client = ActionClient(self, Spin, 'spin')

        # get map res for rviz marker
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.map_data = None
        self.occGrid = None

        self.saving_in_progress = False
        self.robot_position = (0, 0)

         # for showing casualty locations in rviz
        self.cas_marker = rviz_marker.RvizMarker()

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.simulating = use_sim_time


        self.curr_target = None

        self.robot_yaw = None


    def start_casualty_callback(self, request, response):
        if request.state == "SAVE":
            if self.mission_state != "SAVE":
                self.mission_state = "SAVE"
                self.save_timer = self.create_timer(3.0, self.save_casualty)
        else:
            try:
                self.save_timer.cancel()
            except:
                # no timer to cancel
                pass

        return response

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_position = (x, y)
        # extract quaternion
        orientation_q = msg.pose.pose.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        )

        # convert to euler angles
        _, _, yaw = euler_from_quaternion(quaternion)
        self.robot_yaw = yaw  # in radians


    def map_callback(self, msg):
        if not self.map_data:
            self.cas_marker.update_map_consts(
                msg.info.resolution,
                msg.info.origin.position.x, msg.info.origin.position.y)
                
        self.map_data = msg
        data = np.array(self.map_data.data)
        width = self.map_data.info.width
        height = self.map_data.info.height
        self.occGrid = data.reshape((height, width))

    def casualty_callback(self, msg):
        self.get_logger().info("Received casualty locations")
        self.waypoints = msg.casualties

        # show casualties in rviz
        disp_cas = []
        for cas in self.waypoints:
            disp_cas.append( (cas.pose.position.y, cas.pose.position.x) )
            self.get_logger().info(f"Publishing casualty marker at {disp_cas}")
        self.cas_marker.publish_marker_array(disp_cas)

    def launch_now(self):
        self.get_logger().info("calling launcher")
        self.req = Trigger.Request()
        # Ensure service is available
        if not self.launcher_service.wait_for_service(timeout_sec=5.0):
            self.get_logger().warning('launcher service not available')
            return
        # Call the service asynchronously and handle the response when ready
        future = self.launcher_service.call_async(self.req)
        future.add_done_callback(self.launch_callback)

    def launch_callback(self, future):
        try:
            response = future.result()  # Get the response from the service call
            # Update saving_in_progress to continue saving the next target
            self.saving_in_progress = False
        except Exception as e:
            self.get_logger().error(f"launch service call failed: {e}")
    

    # this function could fail if the casualty_loc is not published in time
    def save_casualty(self):
        if not self.saving_in_progress and self.robot_yaw != None: 
            # if pose has not been published yet, do not continue
            if len(self.waypoints) > 0:
                self.saving_in_progress = True
                self.curr_target = self.waypoints.pop(0)

                self.curr_target = self.transform_to_valid_goal(self.curr_target)

                self.cas_marker.publish_marker((self.curr_target.pose.position.y, self.curr_target.pose.position.x))
                self.navigate_to(self.curr_target)
            elif len(self.waypoints) == 0:
                # all casualties have been saved
                self.get_logger().info("All casualties saved.")
                self.cas_save_pub.publish(CasualtySaveStatus(all_casualties_saved=True))
                self.saving_in_progress = False
                self.mission_state = "STOPPED"
                self.save_timer.destroy()
        else:
            pass
            # waiting for current task to be completed...

        if self.robot_yaw == None:
            self.get_logger().warn("Robot yaw not available. Waiting for pose to be published.")
            return


    def transform_to_valid_goal(self, waypoint):

        self.get_logger().info(f"Transforming waypoint to valid goal: {waypoint.pose.position.x}, {waypoint.pose.position.y}")
        # Get the dimensions of the occupancy grid
        rows, cols = self.occGrid.shape

        def dist_to_closest_wall(self, x, y):
            """
            Radiates outward from (x, y) and finds the closest obstacle in self.occGrid.
            An obstacle is defined as a cell with a value > 90.
            
            Args:
                x (float): The x-coordinate in world space.
                y (float): The y-coordinate in world space.

            Returns:
                float: The distance to the closest obstacle. Returns -1 if no obstacle is found.
            """
            if self.occGrid is None:
                self.get_logger().error("Occupancy grid is not available.")
                return -1

            grid_x = int(x)
            grid_y = int(y)

            # Perform a radial search
            max_radius = max(rows, cols)  # Limit the search to the size of the grid
            for radius in range(1, max_radius):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Skip cells outside the current radius
                        if abs(dx) != radius and abs(dy) != radius:
                            continue

                        nx, ny = grid_x + dx, grid_y + dy

                        # Check if the cell is within bounds
                        if 0 <= nx < cols and 0 <= ny < rows:
                            # Check if the cell is an obstacle
                            if self.occGrid[ny, nx] > 90:
                                # Calculate the distance to the obstacle
                                obstacle_x = nx
                                obstacle_y = ny
                                distance = math.sqrt((obstacle_x - x) ** 2 + (obstacle_y - y) ** 2)
                                self.get_logger().info(f"Closest obstacle found at ({obstacle_x}, {obstacle_y}) with distance {distance}")
                                return distance

            # If no obstacle is found, return -1
            self.get_logger().warning("No obstacle found within the occupancy grid.")
            return -1

            ## end of dist to closest wall

        x = waypoint.pose.position.x
        y = waypoint.pose.position.y

        # Perform a radial search to find a valid goal
        resolution = self.map_data.info.resolution
        step_size = resolution  # Move outward in steps of the map resolution
        max_radius = 150  # Limit the search radius to avoid infinite loops

        # Cardinal directions: (dx, dy) for up, down, left, right
        directions = [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0), (step_size, step_size), (-step_size, -step_size), (step_size, -step_size), (-step_size, step_size)] 

        for radius in range(10, max_radius):
            for dx, dy in directions:
                # Move outward in the current direction
                new_x = int(x + dx * radius)
                new_y = int(y + dy * radius)

                # Check the distance to the closest wall
                distance_to_wall = dist_to_closest_wall(self, new_x, new_y)
                self.get_logger().info(f"Distance to closest wall at ({new_x}, {new_y}): {distance_to_wall}")

                # If the distance exceeds 19, return the current waypoint
                if 0 < new_x < cols and 0 < new_y < rows and self.occGrid[new_y][new_x] != -1 and distance_to_wall > DIST_TO_CASUALTY:
                    self.get_logger().info(f"Valid goal found at ({new_x}, {new_y}) with distance {distance_to_wall}")
                    waypoint.pose.position.x = float(new_x)
                    waypoint.pose.position.y = float(new_y)
                    return waypoint

        # If no valid goal is found, return the original waypoint
        self.get_logger().warning("No valid goal found within the search radius. Returning the original waypoint.")
        return waypoint 
        

        



    def navigate_to(self, waypoint):
        x = waypoint.pose.position.x
        y = waypoint.pose.position.y

        self.nav_in_progress = True
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_msg.pose.position.y = y * self.map_data.info.resolution + self.map_data.info.origin.position.y

        # Compute yaw to face the target (low confidence area)
        robot_x, robot_y = self.robot_position
        yaw = math.atan2(y - robot_y, x - robot_x)  # Direct the robot towards the target
        quaternion = quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        self.get_logger().info(f"Navigating to: x={x}, y={y}")
        self.nav_to_pose_client.wait_for_server()
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)


    # feedback for request accepted/rejected
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected')
            self.nav_in_progress = False
            return

        self.get_logger().info('Goal accepted')

        # This is where you wait for the goal to complete
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.nav_result_callback)

    #########################################################################################3
    # near duplicate of spin_360, but spins to face the target

    def spin_face_target(self):
        if not self.spin_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Spin action server not available!")
            return

        x = self.curr_target.pose.position.x * self.map_data.info.resolution + self.map_data.info.origin.position.x
        y = self.curr_target.pose.position.y * self.map_data.info.resolution + self.map_data.info.origin.position.y

        robot_x, robot_y = self.robot_position
        target_yaw = math.atan2(y - robot_y, x - robot_x)  

        current_yaw = self.robot_yaw

        # compute relative angle (shortest rotation direction)
        delta_yaw = target_yaw - current_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))  # normalize to [-pi, pi]
       
        spin_goal = Spin.Goal()
        spin_goal.target_yaw = delta_yaw
        spin_goal.time_allowance = Duration(seconds=15.0).to_msg()

        self.get_logger().info(f"Robot position: ({robot_x}, {robot_y})")
        self.get_logger().info(f"Target position: ({x}, {y})")
        self.get_logger().info(f"Current yaw: {current_yaw} rad, {math.degrees(current_yaw)} deg")
        self.get_logger().info(f"Target yaw: {target_yaw} rad, {math.degrees(target_yaw)} deg")
        self.get_logger().info(f"spin_goal_target_yaw: {spin_goal.target_yaw}")
        self.get_logger().info(f"Computed yaw: {delta_yaw} rad, {math.degrees(delta_yaw)} deg")


        self.get_logger().info("Sending spin goal (FACE TO TARGET)...")
        future = self.spin_client.send_goal_async(spin_goal)
        future.add_done_callback(self.spin_face_target_callback)

    def spin_face_target_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Spin goal rejected')
            return

        self.get_logger().info('Spin goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.spin_face_target_result)

    def spin_face_target_result(self, future):
        result = future.result().result
        self.get_logger().info(f"Spin completed with result: {result}")
        self.nav_in_progress = False
        self.launch_now()
        if self.simulating:
            self.get_logger().info("Waiting for 3 seconds to simulate saving...")
            sleep(3.0)
            self.saving_in_progress = False
        


    #########################################################################################3

    # feedback for when goalPose is reached
    def nav_result_callback(self, future):
        result = future.result().result
        sleep(5.0)
        self.spin_face_target()
        # self.launch_now()
        # self.nav_in_progress = False






def main(args=None):
    rclpy.init(args=args)
    node = CasualtySaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()