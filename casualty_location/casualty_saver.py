import rclpy
from rclpy.node import Node

from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion

# for self.robot_position
from nav_msgs.msg import Odometry
import math

# for map res, for rviz marker
from nav_msgs.msg import OccupancyGrid

from time import sleep

# for mission control
from custom_msg_srv.msg import CasualtySaveStatus, CasualtyLocateStatus
from custom_msg_srv.srv import StartCasualtyService

# for launcher service
from std_srvs.srv import Trigger 

# for casualty locations
from custom_msg_srv.msg import ArrayCasualtyPos

import casualty_location.rviz_marker as rviz_marker



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
        self.cas_locate_pub = self.create_publisher(CasualtyLocateStatus, 'casualty_found', 10)

        # get self.robot_position
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # get map res for rviz marker
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.map_data = None

        # for showing casualty locations in rviz
        self.cas_marker = rviz_marker.RvizMarker()



        self.saving_in_progress = False
        self.robot_position = (0, 0)


    def start_casualty_callback(self, request, response):
        if request.state == "SAVE":
            if self.mission_state != "SAVE":
                self.mission_state = "SAVE"
                self.save_timer = self.create_timer(3.0, self.save_casualty)
        else:
            try:
                self.save_timer.cancel()
            except AttributeError:
                # no timer to cancel
                pass

        return response

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_position = (x, y)


    def map_callback(self, msg):
        if not self.map_data:
            self.cas_marker.update_map_consts(
                msg.info.resolution,
                msg.info.origin.position.x, msg.info.origin.position.y)
                
        self.map_data = msg

    def casualty_callback(self, msg):
        self.get_logger().info("Received casualty locations")
        self.waypoints = msg.casualties

        # show casualties in rviz
        for cas in self.waypoints:
            disp_cas = []
            disp_cas.append( (cas.pose.position.x, cas.pose.position.y) )
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
        except Exception as e:
            self.get_logger().error(f"launch service call failed: {e}")
    

    # this function could fail if the casualty_loc is not published in time
    def save_casualty(self):
        if not self.saving_in_progress:
            if len(self.waypoints) > 0:
                self.navigate_to(self.waypoints.pop(0)) #TODO: save location of current goal casualty to point towards it before launch
                self.saving_in_progress = True
            elif len(self.waypoints) == 0:
                # all casualties have been saved
                self.get_logger().info("All casualties saved.")
                self.saving_in_progress = False
                self.mission_state = "STOPPED"
                self.save_timer.cancel()
                self.save_timer.destroy()
        else:
            pass
            # waiting for current task to be completed...

    def navigate_to(self, waypoint):
        x = waypoint.pose.position.x
        y = waypoint.pose.position.y

        self.nav_in_progress = True
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y

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

    # feedback for when goalPose is reached
    def nav_result_callback(self, future):
        result = future.result().result
        self.launch_now()

        # Wait for the flare to finish launching
        # TODO: use launcher's service response to determine when the flare is launched
        sleep(8)  
               
        self.saving_in_progress = False





def main(args=None):
    rclpy.init(args=args)
    node = CasualtySaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()