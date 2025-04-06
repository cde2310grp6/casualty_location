import time
import sys
import casualtydetection.amg8833_i2c_driver as amg8833_i2c
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Debugging values removed
DETECTION_OFFSET = 5  # Offset from average temperature to detect objects

class IRPubNode(Node):
    def __init__(self):
        super().__init__('ir_pub_node')
        self.publisher = self.create_publisher(String, 'ir_data', 10)

        self.sensor = self.initialize_sensor()
        if self.sensor is None:
            self.get_logger().error("No AMG8833 Found - Check Your Wiring")
            sys.exit(1)

        timer_period = 0.1  # 10 fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def initialize_sensor(self):
        """ Try initializing the AMG8833 sensor at different I2C addresses. """
        sensor = None
        t0 = time.time()
        while (time.time() - t0) < 1:  # Try for 1 second
            try:
                sensor = amg8833_i2c.AMG8833(addr=0x69)
                return sensor
            except Exception as e:
                self.get_logger().warning(f"Sensor init failed at 0x69: {e}")
            try:
                sensor = amg8833_i2c.AMG8833(addr=0x68)
                return sensor
            except Exception as e:
                self.get_logger().warning(f"Sensor init failed at 0x68: {e}")
            time.sleep(0.1)
        return None

    def timer_callback(self):
        pix_to_read = 64
        pix_res = (8, 8)
        grid_size = 32

        status, pixels = self.sensor.read_temp(pix_to_read)

        if not status:
            z = np.reshape(pixels, pix_res)
            z_interp = cv2.resize(z, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)

            # Get the highest value from each column
            max_in_columns = np.max(z_interp, axis=0)  # Max value per column

            # Publish the result as a 1D array
            msg = String()
            msg.data = str(max_in_columns.tolist())  # Convert to list and publish
            self.publisher.publish(msg)

        else:
            self.get_logger().error("Sensor read failed")

def main(args=None):
    rclpy.init(args=args)
    node = IRPubNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
