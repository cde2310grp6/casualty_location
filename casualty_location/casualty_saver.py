import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action import FollowWaypoints
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry

from rclpy.duration import Duration

from time import sleep


# for mission control
from custom_msg_srv.msg import CasualtySaveStatus, CasualtyLocateStatus
from custom_msg_srv.srv import StartCasualtyService

# for launcher service
from std_srvs.srv import Trigger 

# for casualty locations
from custom_msg_srv.msg import ArrayCasualtyPos



class CasualtySaver(Node):
    def __init__(self):
        super().__init__('casualty_saver')

        # use waypoint follower to navigate to the casualty and complete tasks at waypoints
        self.waypoint_follow = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # Wait for the action server to be available
        if not self.waypoint_follow.wait_for_server(timeout_sec=10.0):
            node.get_logger().warn('Action server not available within timeout.')

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

    def start_casualty_callback(self, request, response):
        if request.state == "STOPPED":
            self.mission_state = "STOPPED"
        elif request.state == "LOCATE":
            self.mission_state = "LOCATE"
        elif request.state == "SAVE":
            if self.mission_state != "SAVE":
                self.mission_state = "SAVE"
                self.save_casualty()
            pass
        return response

    def casualty_callback(self, msg):
        self.waypoints = msg.casualties

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
        # Create the goal message
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = self.waypoints

        # Call the FollowWaypoints action asynchronously
        future = self.waypoint_follow.send_goal_async(goal_msg)
        future.add_done_callback(self.follow_waypoints_callback)

    def follow_waypoints_callback(self, future):
        try:
            result = future.result()  # Get the result from the action server
            self.get_logger().info(f"FollowWaypoints result: {result}")
        except Exception as e:
            self.get_logger().error(f"Failed to call FollowWaypoints action: {e}")
            save_casualty()






def main(args=None):
    rclpy.init(args=args)
    node = CasualtySaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()