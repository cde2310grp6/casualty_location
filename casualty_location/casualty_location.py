import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from scipy.interpolate import interp1d
import numpy as np
from collections import deque
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import math
from nav2_msgs.action import Spin
from rclpy.duration import Duration

SENSOR_FOV = 60.0  # Field of view in degrees
POSE_CACHE_SIZE = 100  # Number of poses to keep in the cache
ODOM_RATE = 29.0 # Rate of odometry updates in Hz
ORIGIN_CACHE_SIZE = 1 # Number of origin poses to keep in the cache
MAP_RATE = 0.5 # Rate of map updates in Hz
DELAY_IR = 0.5  # Delay in seconds for IR data processing

GAZEBO = True  # Set to True if running in Gazebo simulation

class BotPose(object):
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def __str__(self):
        return f"BotPose(x={self.x}, y={self.y}, yaw={self.yaw})"

class FinderNode(Node):
    def __init__(self):
        super().__init__('finder')
        self.get_logger().info("Casualty Finder Node Started")

        self.map_received = False
        self.pose_received = False
        self.map_data = None
        self.grid = None
        self.thermal_confidence = None
        self.nav_in_progress = False

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        if not GAZEBO:
            self.ir_sub = self.create_subscription(String, '/ir_data', self.ir_callback, 10)
            self.latest_ir_data = None
        else:
            self.latest_ir_data = [20] * 8

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')

        self.robot_position = (0, 0)
        self.robot_yaw = 0.0
        self.robot_cache = []
        self.pose_cache_full = False
        self.origin = (0, 0)
        self.origin_cache = []
        self.origin_cache_full = False
        self.resolution = 0.0
        self.pose_index = round(POSE_CACHE_SIZE - DELAY_IR * ODOM_RATE -1) # Index of the pose to use for navigation
        self.origin_index = 0 #round(ORIGIN_CACHE_SIZE - DELAY_IR * MAP_RATE -1) # Index of the origin to use for navigation

        self.visited_frontiers = set()
        self.ignored_frontiers = []
        self.previous_frontier = None

        self.fig, self.ax = None, None
        self.robot_marker, self.robot_arrow = None, None
        plt.ion()


        self.explore_timer = self.create_timer(5.0, self.explore)

    def map_callback(self, msg):
        self.map_data = msg
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        self.origin = (origin_x, origin_y)
        self.resolution = self.map_data.info.resolution
        self.origin_cache.append(self.origin)
        self.clean_origin_cache()

        if not self.map_received:
            self.map_received = True
            data = np.array(self.map_data.data)
            width = self.map_data.info.width
            height = self.map_data.info.height
            self.grid = data.reshape((height, width))
            self.thermal_confidence = np.zeros_like(self.grid, dtype=int)
            self.init_plot()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_position = (x, y)

        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        _, _, yaw = euler_from_quaternion(quaternion)
        self.robot_yaw = yaw

        self.pose_received = True
        pose = BotPose(x, y, yaw)
        self.robot_cache.append(pose)
        #self.get_logger().info(f"Robot Pose: {pose}")
        self.clean_pose_cache()

        if GAZEBO:
            self.paint_wall()
            self.update_plot()

    def ir_callback(self,msg):
        try:
            ir_values = eval(msg.data)
            self.latest_ir_data = ir_values
            self.paint_wall()
            self.update_plot()
            #self.get_logger().info(f"{ir_values}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse IR data: {e}")

    def clean_pose_cache(self):
        # Remove old poses from the cache
        if len(self.robot_cache) > POSE_CACHE_SIZE:
            self.robot_cache.pop(0)
            self.pose_cache_full = True

    def clean_origin_cache(self):
        # Remove old origins from the cache
        if len(self.origin_cache) > ORIGIN_CACHE_SIZE:
            self.origin_cache.pop(0)
            self.origin_cache_full = True

    def init_plot(self):
        if not self.map_received:
            return
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(self.grid, cmap='hot', origin='lower', vmin=-1, vmax=100)
        plt.show()

    def update_plot(self):
        if not self.map_received or not self.pose_received:
            return

        robot_x = int((self.robot_position[0] - self.origin[0]) / self.resolution)
        robot_y = int((self.robot_position[1] - self.origin[1]) / self.resolution)

        if self.robot_marker:
            self.robot_marker.remove()
        if self.robot_arrow:
            self.robot_arrow.remove()

        self.robot_marker = self.ax.plot(robot_x, robot_y, 'ro', markersize=10, label="Robot")[0]

        arrow_length = 3
        dx = arrow_length * math.cos(self.robot_yaw)
        dy = arrow_length * math.sin(self.robot_yaw)
        self.robot_arrow = self.ax.arrow(robot_x, robot_y, dx, dy, head_width=2, head_length=2, fc='r', ec='r')

        self.im.set_data(self.grid)
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def paint_wall(self):
        if not self.map_received or not self.pose_cache_full or not self.origin_cache_full:
            return

        cy = int((self.robot_cache[self.pose_index].x - self.origin[0]) / self.resolution)
        cx = int((self.robot_cache[self.pose_index].y - self.origin[1]) / self.resolution)
        rows, cols = self.map_data.info.height, self.map_data.info.width

        start_angle = math.degrees(self.robot_cache[self.pose_index].yaw) - SENSOR_FOV / 2
        end_angle = math.degrees(self.robot_cache[self.pose_index].yaw) + SENSOR_FOV / 2
        num_rays = int(SENSOR_FOV * 2)
        
        data = self.latest_ir_data
        x = np.linspace(0, len(data) - 1, len(data))  # Indices of the input data
        interpolator = interp1d(x, data, kind='linear', fill_value='extrapolate')  # Linear interpolation
        interpolated_data = interpolator(np.linspace(0, len(data) - 1, num_rays))  # Interpolated data
        angles = np.linspace(start_angle, end_angle, num_rays)

        for idx, angle in enumerate(angles):
            rad_angle = math.radians(angle)
            dx = math.cos(rad_angle)
            dy = math.sin(rad_angle)

            for step in range(1, 300):
                row = int(round(cx + step * dy))
                col = int(round(cy + step * dx))

                if 0 <= row < rows and 0 <= col < cols:
                    if self.grid[row][col] != 0:
                        # Assign value to the current cell
                        if self.grid[row][col] > 90:
                            self.grid[row][col] = interpolated_data[idx]
                        else:
                            self.grid[row][col] = (self.grid[row][col] + interpolated_data[idx]) / 2
                        
                        # Update the adjacent cells (neighbors)
                        for d_row in [-1, 0, 1]:
                            for d_col in [-1, 0, 1]:
                                # Skip the current cell itself
                                if d_row == 0 and d_col == 0:
                                    continue
                                
                                adj_row = row + d_row
                                adj_col = col + d_col

                                # Ensure the neighbor is within bounds
                                if 0 <= adj_row < rows and 0 <= adj_col < cols:
                                    if self.grid[adj_row][adj_col] != 0:
                                        # Update non-zero neighboring cells
                                        self.grid[adj_row][adj_col] = (self.grid[adj_row][adj_col] + interpolated_data[idx]) / 2
                        break
                else:
                    break

    def spin_360(self):
        if not self.spin_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Spin action server not available!")
            return

        spin_goal = Spin.Goal()
        spin_goal.target_yaw = 2 * math.pi  # 360 degrees
        spin_goal.time_allowance = Duration(seconds=15.0).to_msg()

        self.get_logger().info("Sending spin goal (360 degrees)...")
        future = self.spin_client.send_goal_async(spin_goal)
        future.add_done_callback(self.spin_response_callback)

    def navigate_to(self, x, y):
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


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")
            self.nav_in_progress = False
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f"Navigation result: {result}")
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")
        self.spin_360()
        #self.nav_in_progress = False

    def spin_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Spin goal rejected')
            return

        self.get_logger().info('Spin goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.spin_result_callback)

    def spin_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Spin completed with result: {result}")
        self.nav_in_progress = False

    def find_frontiers(self, map_array):
        frontiers = []
        rows, cols = map_array.shape
        visited = np.zeros_like(map_array, dtype=bool)
        queue = deque()

        robot_row = int((self.robot_position[1] - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        robot_col = int((self.robot_position[0] - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        self.get_logger().info(f"Robot map index: ({robot_row}, {robot_col})")

        queue.append((robot_row, robot_col))
        visited[robot_row, robot_col] = True

        while queue:
            r, c = queue.popleft()

            # Check if this cell is a wall (100) and hasn't been thermally scanned yet
            if self.grid[r, c] > 90:
                neighbors = self.grid[r-1:r+2, c-1:c+2].flatten()
                if 100 in neighbors and (r, c) not in self.ignored_frontiers:
                    frontiers.append((r, c))

            # Continue BFS to adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        self.get_logger().info(f"Found {len(frontiers)} frontiers (walls needing thermal scan)")
        return frontiers


    def choose_frontier(self, frontiers, map_array):
        """
        Choose the best frontier to explore based on distance, number of unknown cells around it, and direction.
        """
        robot_col = int((self.robot_position[0] - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        robot_row = int((self.robot_position[1] - self.map_data.info.origin.position.y) / self.map_data.info.resolution)

        chosen_frontier = None

        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue

            # Calculate distance to the robot
            distance = np.sqrt((robot_row - frontier[0])**2 + (robot_col - frontier[1])**2)

            # Calculate the number of unknown cells around the frontier
            r, c = frontier
            neighbors = map_array[r-1:r+2, c-1:c+2].flatten()
            unknown_count = np.sum(neighbors > 90)

            # Skip frontiers near already visited frontiers
            #Copilot: Increased threshold to 3
            skip = False
            for visited in self.visited_frontiers:
                if np.sqrt((visited[0] - r)**2 + (visited[1] - c)**2) < 2:  # Threshold distance
                    skip = True
                    break
            if skip:
                continue

            # Calculate the angular penalty
            frontier_angle = math.atan2(r - robot_row, c - robot_col)
            angular_penalty = min(abs(frontier_angle - self.robot_yaw), 2 * math.pi - abs(frontier_angle - self.robot_yaw)) / math.pi

            # Calculate a score based on distance, unknown count, and direction
            direction_penalty = 0
            if self.previous_frontier:
                prev_r, prev_c = self.previous_frontier
                direction_penalty = np.sqrt((prev_r - r)**2 + (prev_c - c)**2)

            score = 0.2 * distance + unknown_count  - 1 * angular_penalty # Extra emphasis on direction penalty (wastes a lot of time doing u turns)

            high_score = 0
            # Check if the score is better than the current best
            if score > high_score:
                chosen_frontier = frontier
                high_score = score

        if chosen_frontier:
            self.visited_frontiers.add(chosen_frontier)
            self.previous_frontier = chosen_frontier
            self.get_logger().info(f"Chosen frontier: {chosen_frontier}")
            # self.frontier_marker.publish_marker(chosen_frontier=chosen_frontier)
        else:
            self.get_logger().warning("No valid frontier found")

        return chosen_frontier
    

    def explore(self):
        if self.nav_in_progress:
            self.get_logger().info("Navigation in progress, cannot choose frontier")
            return None
        
        if self.map_data is None:
            self.get_logger().warning("No map data available")
            return

        # Convert map to numpy array
        map_array = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width))

        # Detect frontiers
        frontiers = self.find_frontiers(map_array)

        if not frontiers:
            self.get_logger().info("No frontiers found. Exploration complete!")
            return

        # Choose the closest frontier
        chosen_frontier = self.choose_frontier(frontiers, map_array)

        if not chosen_frontier:
            self.get_logger().warning("No frontiers to explore")
            return

        # Convert the chosen frontier to world coordinates
        goal_x = chosen_frontier[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_y = chosen_frontier[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y

        # Navigate to the chosen frontier
        self.navigate_to(goal_x, goal_y)


def main(args=None):
    rclpy.init(args=args)
    node = FinderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        plt.ioff()
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()
