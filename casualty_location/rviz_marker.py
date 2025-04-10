import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class RvizMarker(Node):
    def __init__(self):
        super().__init__('rviz_marker_node')
        self.marker_pub = self.create_publisher(Marker, 'frontier_chosen_marker', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'frontier_all_marker', 10)

        self.heatmap_pub = self.create_publisher(MarkerArray, 'heatmap_marker', 10)
        self.prev_heat_array = None

        # used for deleting previous markers
        self.prev_marker_array = None

    def update_map_consts(self, map_res, map_origin_x, map_origin_y):
        self.map_res = map_res
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.get_logger().info(f"Map consts updated: {self.map_res}, {self.map_origin_x}, {self.map_origin_y}")

    def create_marker(self, 
            position,
            frame_id="map", 
            marker_id=0, 
            marker_type=Marker.SPHERE,  
            orientation = (0.0, 0.0, 0.0, 1.0), 
            scale = (1.0, 1.0, 1.0), 
            color = (0.0, 1.0, 1.0, 0.8) #RGB-Alpha
            ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontier"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[1] * self.map_res + self.map_origin_x)
        marker.pose.position.y = float(position[0] * self.map_res + self.map_origin_y)
        marker.pose.position.z = 0.0 # we don't have z axis in this project
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    def publish_marker(self, chosen_frontier):
        marker_x = chosen_frontier[1] * self.map_res + self.map_origin_x
        marker_y = chosen_frontier[0] * self.map_res + self.map_origin_y
        self.get_logger().info(f"Publishing marker at frontier {chosen_frontier} coords: {marker_x}, {marker_y}")
        self.marker_pub.publish(self.create_marker(
            position=chosen_frontier, 
            color=(0.6, 0.0, 0.1, 1.0), 
            scale=(0.6, 0.6, 0.6))
            )   

    def delete_marker_array(self, marker_array):
        for marker in marker_array.markers:
            marker.action = Marker.DELETE
        self.marker_array_pub.publish(marker_array)

    def publish_marker_array(self, frontiers_list):
        self.get_logger().info(f"Publishing markers")
        marker_array = MarkerArray()

        # set id for each marker
        # apparently causes issues if all ids are the same
        # only show every 50th marker (reduce render load)
        if len(frontiers_list) > 100: frontiers_list = frontiers_list[::100]
        for i, frontier in enumerate(frontiers_list):
            marker_array.markers.append(self.create_marker(marker_id=i, position=frontier))
            # self.get_logger().info(f"Marker {i} at frontier {frontier[1] * self.map_res + self.map_origin_x}, {frontier[0] * self.map_res + self.map_origin_x}")

        if self.prev_marker_array != None:
            self.delete_marker_array(self.prev_marker_array)

        self.prev_marker_array = marker_array
        self.marker_array_pub.publish(marker_array)

    # function specifically designed for casualty_location to plot the heatmap on RViz
    # check publish_heat
    # heat_marker_array is a list of tuples (y, x, temp)
    def publish_heatmap(self, heat_marker_array):
        # self.get_logger().info(f"RViz: displaying heatmap")
        marker_array = MarkerArray()

        # the following code maps the temp from heat_marker_array into a range of colours
        # hot is red, blue is cold

        #TODO: confirm if these are correct
        temp_min = 25.0  # Minimum temperature
        temp_max = 45.0  # Maximum temperature
        color_low = (0.0, 0.0, 1.0, 0.8)  # Blue (low temperature)
        color_high = (1.0, 0.0, 0.0, 0.8)  # Red (high temperature)

        # Set markers for each heat point
        for i, heatpoint in enumerate(heat_marker_array):
            y = heatpoint[0]
            x = heatpoint[1]
            temp = heatpoint[2]

            # Normalize temperature to [0, 1]
            normalized_temp = min(max((temp - temp_min) / (temp_max - temp_min), 0.0), 1.0)

            # Interpolate between low and high colors
            if temp == -1.0:
                r = 0.0
                g = 0.3
                b = 0.0
                a = 0.2
            else:
                r = color_low[0] + normalized_temp * (color_high[0] - color_low[0])
                g = color_low[1] + normalized_temp * (color_high[1] - color_low[1])
                b = color_low[2] + normalized_temp * (color_high[2] - color_low[2])
                a = color_low[3] + normalized_temp * (color_high[3] - color_low[3])

            # Add marker to the array
            marker_array.markers.append(
                self.create_marker(
                    marker_id=i,
                    position=(y, x),
                    color=(r, g, b, a)
                )
            )

        # Delete previous markers if they exist
        if self.prev_heat_array is not None:
            self.delete_marker_array(self.prev_heat_array)

        # Publish the new marker array
        self.prev_heat_array = marker_array
        self.marker_array_pub.publish(marker_array)