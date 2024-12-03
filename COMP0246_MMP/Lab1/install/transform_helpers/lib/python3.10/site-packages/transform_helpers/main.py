import rclpy
from rclpy.node import Node
# TODO: Import the message type that holds data describing robot joint angle states
# this tutorial may have hints: https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/Using-URDF-with-Robot-State-Publisher.html#publish-the-state
from sensor_msgs.msg import JointState
# TODO: Import the class that publishes coordinate frame transform information
# this tutorial may have hints: https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
from tf2_ros import TransformBroadcaster
# TODO: Import the message type that expresses a transform from one coordinate frame to another
# this same tutorial from earlier has hints: https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
from geometry_msgs.msg import TransformStamped

import numpy as np
from numpy.typing import NDArray

from transform_helpers.utils import rotmat2q

# Modified DH Params for the Franka FR3 robot arm
# https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
# meters
a_list = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d_list = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]

# radians
alpha_list = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
theta_list = [0] * len(alpha_list)

DH_PARAMS = np.array([a_list, d_list, alpha_list, theta_list]).T

BASE_FRAME = "base"
FRAMES = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8"]

def get_transform_n_to_n_minus_one(n: int, theta: float) -> NDArray:
    # this function calculates the transform to go from n to n-1 
    # using modified denavit hartenberg parameters

    transform_matrix = np.zeros((4,4))

    n_minus_one = n - 1

    # # TODO: implement this function
    # # note that it may be helpful to refer to documentation on modified denavit hartenberg parameters:
    # # https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters

    # Get DH params
    a_n_minus_one = DH_PARAMS[n_minus_one, 0]  # Connecting rod length a_n-1
    d_n = DH_PARAMS[n_minus_one, 1]  # offset d_n
    alpha_n_minus_one = DH_PARAMS[n_minus_one, 2]  # Torsion Angle alpha_n-1
    theta_n = theta


    transform_matrix[0, 0] = np.cos(theta_n)
    transform_matrix[0, 1] = -np.sin(theta_n)
    transform_matrix[0, 2] = 0
    transform_matrix[0, 3] = a_n_minus_one
    
    transform_matrix[1, 0] = np.sin(theta_n) * np.cos(alpha_n_minus_one)
    transform_matrix[1, 1] = np.cos(theta_n) * np.cos(alpha_n_minus_one)
    transform_matrix[1, 2] = -np.sin(alpha_n_minus_one)
    transform_matrix[1, 3] = -d_n * np.sin(alpha_n_minus_one)
    
    transform_matrix[2, 0] = np.sin(theta_n) * np.sin(alpha_n_minus_one)
    transform_matrix[2, 1] = np.cos(theta_n) * np.sin(alpha_n_minus_one)
    transform_matrix[2, 2] = np.cos(alpha_n_minus_one)
    transform_matrix[2, 3] = d_n * np.cos(alpha_n_minus_one)
    
    transform_matrix[3, 0] = 0
    transform_matrix[3, 1] = 0
    transform_matrix[3, 2] = 0
    transform_matrix[3, 3] = 1

    # a, d, alpha, _ = DH_PARAMS[n - 1]
    # transform_matrix = np.array([
    #     [np.cos(theta), -np.sin(theta), 0, a],
    #     [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
    #     [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
    #     [0, 0, 0, 1]
    # ])
    
    return transform_matrix



class ForwardKinematicCalculator(Node):

    def __init__(self):
        super().__init__('fk_calculator')

        # # TODO: create a subscriber to joint states, can you find which topic
        # # this publishes on by using ros2 topic list while running the example?
        # raise NotImplementedError
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # self.prefix = ""
        self.prefix = "my_robot/"
    
    # callback function
    def joint_state_callback(self, msg: JointState):
        # Process the received joint states and call publish_transforms
        self.publish_transforms(msg)

    def publish_transforms(self, msg: JointState):


        self.get_logger().debug(str(msg))

        # note our frames list is longer than the number of joints, so some special handling is required
        for i in range(len(FRAMES) - 1, -1, -1):
            frame_id = self.prefix + FRAMES[i]
            if i != 0:
                parent_id = self.prefix + FRAMES[i - 1]
            else:
                parent_id = self.prefix + BASE_FRAME
            theta = None
            if i != len(FRAMES) - 1 and i != 0:
                # joint msg has 7 entries, not base or static flange
                # 'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
                theta = msg.position[i - 1]
            elif i == len(FRAMES) - 1:
                # flange joint with the static transform and theta of zero
                theta = 0
            else:
                theta = 0

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_id
            t.child_frame_id = frame_id

            if i != 0:
                transform = get_transform_n_to_n_minus_one(i, theta)
            else:
                transform = np.eye(4)

            quat = rotmat2q(transform[:3, :3])

            # # TODO: set the translation and rotation in the message we have created
            # # you can check the documentation for the message type for ros2
            # # to see what members it has
            # raise NotImplementedError

            t.transform.translation.x = transform[0, 3]
            t.transform.translation.y = transform[1, 3]
            t.transform.translation.z = transform[2, 3]

            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w

            # Publish
            self.tf_broadcaster.sendTransform(t)
    



def main(args=None):
    rclpy.init(args=args)

    # TODO: initialize our class and start it spinning
    # this example may be helpful: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html#write-the-subscriber-node

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    
    fk_calculator = ForwardKinematicCalculator()
    try:
        rclpy.spin(fk_calculator) # Start the node's event loop
    except KeyboardInterrupt:
        pass
    finally:
        fk_calculator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()