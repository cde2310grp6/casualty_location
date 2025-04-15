import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
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
import random

# for mission control
from custom_msg_srv.msg import CasualtySaveStatus, CasualtyLocateStatus, ArrayCasualtyPos
from custom_msg_srv.srv import StartCasualtyService

SENSOR_FOV = 60.0  # Field of view in degrees
POSE_CACHE_SIZE = 20  # Number of poses to keep in the cache
ODOM_RATE = 20.0 # Rate of odometry updates in Hz
ORIGIN_CACHE_SIZE = 1 # Number of origin poses to keep in the cache
MAP_RATE = 0.5 # Rate of map updates in Hz
DELAY_IR = 0.5  # Delay in seconds for IR data processing

CASUALTY_COUNT = 1 # Number of casualties to find

DIST_TO_CASUALTY = 4.0 # Distance to casualty before stopping to fire

class BotPose(object):
    def __init__(self, x=0, y=0, yaw=0):
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
        self.occupancy_grid = None
        self.grid = None
        self.nav_in_progress = False
        self.exploring = True
        self.painting = False

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.GAZEBO = use_sim_time


        
        if not self.GAZEBO:
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
        self.origin_index = round(ORIGIN_CACHE_SIZE - DELAY_IR * MAP_RATE -1) # Index of the origin to use for navigation

        self.visited_frontiers = set()
        self.recent_frontier = None
        self.costmap = None

        self.casualties = []
        self.binary = None

        self.fig, self.ax = None, None
        self.robot_marker, self.robot_arrow = None, None
        
        self.is_spinning = False
        plt.ion()


        # service for mission_control to initiate casualty_location
        self.start_casualty_service = self.create_service(StartCasualtyService, 'casualty_state', self.start_casualty_callback)
        self.mission_state = "STOPPED"

        # topic to tell mission_control when casualty_location complete
        self.cas_locate_pub = self.create_publisher(CasualtyLocateStatus, 'casualty_found', 10)
        self.cas_pose_pub = self.create_publisher(ArrayCasualtyPos, 'casualty_locations', 10)

    def start_casualty_callback(self, request, response):
        if request.state == "STOPPED":
            self.mission_state = "STOPPED"
            try:
                self.explore_timer.cancel()
            except:
                pass
        elif request.state == "LOCATE":
            self.mission_state = "LOCATE"
            self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
            self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
            self.explore_timer = self.create_timer(1.0, self.explore)
            self.offset = BotPose()
            self.odom_pose = BotPose()

        else:
            self.mission_state = "STOPPED"
        return response
    
    def map_callback(self, msg):
        def clean_origin_cache(self):
        # Remove old origins from the cache
            if len(self.origin_cache) > ORIGIN_CACHE_SIZE:
                self.origin_cache.pop(0)
                self.origin_cache_full = True

        self.map_data = msg
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        self.origin = (origin_x, origin_y)
        self.resolution = self.map_data.info.resolution
        self.origin_cache.append(self.origin)
        clean_origin_cache(self)

        if not self.map_received:
            data = np.array(self.map_data.data)
            width = self.map_data.info.width
            height = self.map_data.info.height
            self.grid = data.reshape((height, width))
            self.occupancy_grid = data.reshape((height, width))
            self.map_received = True
            self.init_plot()

    def odom_callback(self, msg):

        def clean_pose_cache(self):
            # Remove old poses from the cache
            if len(self.robot_cache) > POSE_CACHE_SIZE:
                self.robot_cache.pop(0)
                self.pose_cache_full = True
    
        self.odom_pose.x = msg.pose.pose.position.x
        self.odom_pose.y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        _, _, self.odom_pose.yaw = euler_from_quaternion(quaternion)

        self.pose_received = True
        pose = BotPose(self.odom_pose.x-self.offset.x, self.odom_pose.y-self.offset.y, self.odom_pose.yaw-self.offset.yaw) #offset correction
        self.robot_position = (pose.x,pose.y) #corrected values
        self.robot_yaw = pose.yaw
        self.robot_cache.append(pose)
        clean_pose_cache(self)

        if self.GAZEBO:
            self.paint_wall()
            self.update_plot()


    ###
    ## This function is used to correct the pose of the robot based on the AMCL pose
    ## It calculates the offset between the odometry pose and the AMCL pose and stores it in self.offset
    ## i.e. AMCL_pose + offset == Odom_pose

    def pose_callback(self, msg):
        self.offset.x = self.odom_pose.x - msg.pose.pose.position.x
        self.offset.y = self.odom_pose.y - msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        _, _, amcl_yaw = euler_from_quaternion(quaternion)
        self.offset.yaw = self.odom_pose.yaw - amcl_yaw

    def ir_callback(self,msg):
        if not self.exploring:
            return
        try:
            ir_values = eval(msg.data)
            self.latest_ir_data = ir_values
            self.paint_wall()
            self.update_plot()
            #self.get_logger().info(f"{ir_values}")
        except Exception as e:
            self.get_logger().warning(f"Failed to parse IR data: {e}")

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

    def transform_to_valid_goal(self, waypoint):

        self.get_logger().info(f"Transforming waypoint to valid goal: {waypoint.pose.position.x}, {waypoint.pose.position.y}")
        # Get the dimensions of the occupancy grid
        rows, cols = self.occupancy_grid.shape

        def dist_to_closest_wall(self, x, y):
            """
            Radiates outward from (x, y) and finds the closest obstacle in self.grid.
            An obstacle is defined as a cell with a value > 90.
            
            Args:
                x (float): The x-coordinate in world space.
                y (float): The y-coordinate in world space.

            Returns:
                float: The distance to the closest obstacle. Returns -1 if no obstacle is found.
            """
            if self.occupancy_grid is None:
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
                            if self.occupancy_grid[ny, nx] > 10:
                                # Calculate the distance to the obstacle
                                obstacle_x = nx
                                obstacle_y = ny
                                distance = math.sqrt((obstacle_x - x) ** 2 + (obstacle_y - y) ** 2)
                                #self.get_logger().info(f"Closest obstacle found at ({obstacle_x}, {obstacle_y}) with distance {distance}")
                                return distance

            # If no obstacle is found, return -1
            self.get_logger().warning("No obstacle found within the occupancy grid.")
            return -1

            ## end of dist to closest wall

        x = waypoint.pose.position.x
        y = waypoint.pose.position.y

        # Perform a radial search to find a valid goal
        resolution = self.map_data.info.resolution
        #self.get_logger().info(f"Map resolution: {resolution}")
        step_size = resolution  # Move outward in steps of the map resolution
        max_radius = 150  # Limit the search radius to avoid infinite loops

        # Cardinal directions: (dx, dy) for up, down, left, right
        directions = [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0), (step_size, step_size), (-step_size, -step_size), (step_size, -step_size), (-step_size, step_size)] 
        random.shuffle(directions)
        for radius in range(10, max_radius):
            for dx, dy in directions:
                # Move outward in the current direction
                new_x = int(x + dx * radius)
                new_y = int(y + dy * radius)

                # Check the distance to the closest wall
                distance_to_wall = dist_to_closest_wall(self, new_x, new_y)
                #self.get_logger().info(f"Distance to closest wall at ({new_x}, {new_y}): {distance_to_wall}")

                # If the distance exceeds 19, return the current waypoint
                if 0 < new_x < cols and 0 < new_y < rows and 0.0 <= self.grid[new_y][new_x] < 10 and distance_to_wall > DIST_TO_CASUALTY:
                    #self.get_logger().info(f"Valid goal found at ({new_x}, {new_y}) with distance {distance_to_wall}")
                    waypoint.pose.position.x = float(new_x)
                    waypoint.pose.position.y = float(new_y)
                    return waypoint

        # If no valid goal is found, return the original waypoint
        self.get_logger().warning("No valid goal found within the search radius. Returning the original waypoint.")
        return waypoint 

    def paint_wall(self):
        # mission_control
        # do not proceed if not in LOCATE state
        if self.mission_state != "LOCATE" or self.painting == True:
            return

        # Only proceed to paint walls if the robot is spinning
        # it is most accurate when spinning rather than moving
        if not self.is_spinning:
            self.get_logger().info("Not spinning, not painting walls")
            return

        if not self.map_received or not self.pose_cache_full or not self.origin_cache_full:
            return
        self.painting = True
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

            for step in range(1,18):
                row = int(round(cx + step * dy))
                col = int(round(cy + step * dx))

                if 0 <= row < rows and 0 <= col < cols:
                    if self.grid[row][col] > 10:
                        if self.grid[row][col] > 90:
                            self.grid[row][col] = interpolated_data[idx]
                            # Check if the next cell is also a wall (100) and hasn't been thermally scanned yet
                        row = int(round(cx + (step+1) * dy)) 
                        col = int(round(cy + (step+1) * dx))
                        if 0 <= row < rows-1 and 0 <= col < cols-1:
                            if self.grid[row][col] > 90:
                                self.grid[row][col] = interpolated_data[idx]
                        break
                else:
                    break
        self.painting = False

    def spin_360(self):
        self.is_spinning = True

        if not self.spin_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Spin action server not available!")
            return

        spin_goal = Spin.Goal()
        spin_goal.target_yaw = 2 * math.pi  # 360 degrees
        spin_goal.time_allowance = Duration(seconds=20.0).to_msg()

        self.get_logger().info("Sending spin goal (360 degrees)...")
        future = self.spin_client.send_goal_async(spin_goal)
        future.add_done_callback(self.spin_response_callback)

        

    def navigate_to(self, x, y, yaw):
        self.nav_in_progress = True
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_msg.pose.position.y = y * self.map_data.info.resolution + self.map_data.info.origin.position.y

        # Compute yaw to face the target (low confidence area)
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
        self.is_spinning = False

    def generate_costmap(self, grid, visited_frontiers, threshold=90):
        """
        Generate a costmap where each cell's value reflects:
        - How large a connected unknown region it belongs to
        - How close it is to the robot
        """
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        costmap = np.zeros_like(grid, dtype=float)
        robot_c = int((self.robot_position[0] - self.origin[0]) / self.resolution)
        robot_r = int((self.robot_position[1] - self.origin[1]) / self.resolution)

        for r in range(rows):
            for c in range(cols):
                if visited[r, c] or grid[r, c] <= threshold:
                    continue

                # Flood-fill to get connected region
                queue = deque([(r, c)])
                region_cells = []
                visited[r, c] = True

                while queue:
                    cr, cc = queue.popleft()
                    region_cells.append((cr, cc))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and grid[nr, nc] > threshold):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                region_size = len(region_cells)

                # Assign score to all cells in this region
                for cr, cc in region_cells:
                    dist = np.sqrt((cr - robot_r)**2 + (cc - robot_c)**2)
                    score = (region_size - 5) / (dist + 1.0)  # +1 to avoid div by zero
                    costmap[cr, cc] = score

        for row, col in visited_frontiers:
            costmap[row, col] = 0  # Mark visited frontiers as invalid

        return costmap

    def choose_frontier(self, costmap):
        if self.recent_frontier is not None:
            for i in range(len(costmap)):
                for j in range(len(costmap[0])):
                    dist = np.sqrt((i - self.recent_frontier[0])**2 + (j - self.recent_frontier[1])**2)
                    if dist < 5: #change this to change ignore radius
                        costmap[i, j] = 0
        max_value_index = np.argmax(costmap)
        row = max_value_index // costmap.shape[1]
        col = max_value_index % costmap.shape[1]
        if costmap[row, col] <= 0:
            return None
        self.visited_frontiers.add((row, col))
        self.recent_frontier = [row, col]
        return (row, col)

    def init_costmap_plot(self):
        """Initialize the costmap plot."""
        self.costmap_fig, self.costmap_ax = plt.subplots(figsize=(8, 8))
        self.costmap_im = self.costmap_ax.imshow(
            np.zeros((1, 1)), cmap='hot', origin='lower', vmin=0, vmax=1
        )
        self.costmap_ax.set_title("Costmap")
        plt.show(block=False)

    def update_costmap_plot(self, costmap):
        """Update the costmap plot."""
        if not hasattr(self, 'costmap_im'):
            self.init_costmap_plot()

        self.costmap_im.set_data(costmap)
        self.costmap_im.set_extent([0, costmap.shape[1], 0, costmap.shape[0]])
        self.costmap_im.set_clim(vmin=np.min(costmap), vmax=np.max(costmap))
        self.costmap_fig.canvas.draw()
        self.costmap_fig.canvas.flush_events()

    def explore(self):
        if not self.exploring:
            return

        if self.nav_in_progress:
            self.get_logger().info("Navigation in progress, cannot choose frontier")
            return None

        if self.map_data is None:
            self.get_logger().warning("No map data available")
            return

        # mission_control
        # do not proceed if not in LOCATE state
        if self.mission_state != "LOCATE":
            return

        # Generate costmap and choose a frontier
        self.costmap = self.generate_costmap(self.grid, self.visited_frontiers, threshold=90)
        # Update the costmap plot
        self.update_costmap_plot(self.costmap)

        chosen_frontier = self.choose_frontier(self.costmap)
        if chosen_frontier is None:
            self.get_logger().info("No frontiers found. Location complete!")
            self.find_casualties()
            self.exploring = False
            msg = CasualtyLocateStatus()
            msg.all_casualties_found = True
            self.cas_locate_pub.publish(msg)
            return

        # Convert the chosen frontier to world coordinates
        goalPose = PoseStamped()
        goalPose.pose.position.x = float(chosen_frontier[1])
        goalPose.pose.position.y = float(chosen_frontier[0])

        # Transform to a valid goal
        goalPose = self.transform_to_valid_goal(goalPose)

        # Calculate yaw angle to face the goal
        goal_x = goalPose.pose.position.x
        goal_y = goalPose.pose.position.y
        dx = goal_x - chosen_frontier[0]
        dy = goal_y - chosen_frontier[1]
        yaw = math.atan2(dy, dx)  # Calculate yaw angle

        # Log the calculated yaw
        self.get_logger().info(f"Calculated yaw angle to goal: {math.degrees(yaw)} degrees")

        # Update the goalPose orientation with the calculated yaw
        quaternion = quaternion_from_euler(0, 0, yaw)
        goalPose.pose.orientation = Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )

        # Navigate to the chosen frontier
        self.navigate_to(goal_x, goal_y, yaw)

    def find_casualties(self):
        self.grid = np.where(self.grid > 90, 0, self.grid)  # Ignore unexplored/wall cells

        def find_top_hotspots(grid_array, count=CASUALTY_COUNT, neighborhood=2, suppress_radius=10):
            rows, cols = grid_array.shape
            grid_avg = np.zeros_like(grid_array)

            # Calculate local average around each pixel
            for row in range(rows):
                for col in range(cols):
                    total = 0
                    div = 0
                    for offset_i in range(-neighborhood, neighborhood + 1):
                        for offset_j in range(-neighborhood, neighborhood + 1):
                            i, j = row + offset_i, col + offset_j
                            if 0 <= i < rows and 0 <= j < cols:
                                if grid_array[i][j] > 10:
                                    total += grid_array[i][j]
                                    div += 1
                    grid_avg[row][col] = total / div if div != 0 else 0

            hotspots = []

            # Find top 'count' hotspots
            for _ in range(count):
                idx = np.argmax(grid_avg)
                hottest_i = idx // cols
                hottest_j = idx % cols
                hottest_value = grid_avg[hottest_i][hottest_j]
                hotspots.append((hottest_i, hottest_j, hottest_value))

                # Suppress surrounding region
                for offset_i in range(-suppress_radius, suppress_radius + 1):
                    for offset_j in range(-suppress_radius, suppress_radius + 1):
                        i, j = hottest_i + offset_i, hottest_j + offset_j
                        if 0 <= i < rows and 0 <= j < cols:
                            grid_avg[i][j] = 0

            return hotspots

        # Find hotspots
        hotspots = find_top_hotspots(self.grid)

        if len(hotspots) == CASUALTY_COUNT:
            for hotspot in hotspots:
                cY, cX, value = hotspot
                self.get_logger().info(f"Casualty at map index: ({cY}, {cX}) with value: {value}")
                cas_x = cX
                cas_y = cY
                self.casualties.append((cas_x, cas_y))
            
            self.publish_casualties()

            self.mission_state = "STOPPED"
            # update mission_control
            msg = CasualtyLocateStatus()
            msg.all_casualties_found = True
            self.cas_locate_pub.publish(msg)


            return

        self.get_logger().warning("Could not find the desired number of casualties")

    def publish_casualties(self):
        msg = ArrayCasualtyPos()

        for x, y in self.casualties:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Identity quaternion

            msg.casualties.append(pose)

        self.cas_pose_pub.publish(msg)
        self.get_logger().info(f"Published {len(msg.casualties)} casualties.")

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
