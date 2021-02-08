import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import math
import matplotlib.pyplot as plt

from planning_utils import a_star, heuristic, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop()  # pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def raytrace(self, p1, p2):
        cells = []

        x_axis = np.arange(p1[0], p2[0] + 1)
        if p1[0] != p2[0]:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        else:  # both points are on the same vertical
            for i in np.arange(p1[1], p2[1]):
                cells.append((p1[0], i))
            return cells

        y0 = p1[1] - m * p1[0]
        f = np.array(list(map(lambda x: m * x + y0, x_axis)))

        x = 0
        y = 0

        while x + 1 < len(f):
            cells.append((x + p1[0], y + p1[1]))
            if f[x + 1] > y + p1[1] + 1:
                y += 1
            else:
                x += 1

        return cells

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 2
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        f = open('colliders.csv', 'r+')
        l = f.readline()
        f.close()

        start_pos = dict((x.strip(), float(y.strip()))
                         for x, y in (element.split(' ')
                                      for element in l.split(', ')))

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(start_pos['lon0'], start_pos['lat0'], 0)

        # TODO: retrieve current global position
        global_position = self.global_position

        # Retrieve your current position in geodetic coordinates from
        # self._latitude, self._longitude and self._altitude. Then
        # use the utility function global_to_local() to convert to local
        # position (using self.global_home as well, which you just set)

        # TODO: convert to current local position using global_to_local()
        local_position = global_to_local([self._longitude, self._latitude, self._altitude], self.global_home)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        grid_start = (-north_offset + math.floor(self.local_position[0]),
                      -east_offset + math.floor(self.local_position[1]))

        # Set goal as some arbitrary position on the grid
        # TODO: adapt to set goal as latitude / longitude position and convert

        # Test path 1: just a straight line from starting point
        goal_lat = 37.7924
        goal_lon = -122.3974

        # Test path 2: around the corner
        # goal_lat = 37.793933
        # goal_lon = -122.397336

        # Test path 3
        goal_lat = 37.793532
        goal_lon = -122.397781

        goal_lat = 37.797366
        goal_lon = -122.394869

        goal_local = global_to_local([goal_lon, goal_lat, 0], self.global_home)
        grid_goal = (-north_offset + math.floor(goal_local[0]),
                     -east_offset + math.floor(goal_local[1]))

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)

        # TODO: prune path to minimize number of waypoints

        # we start trying to fly a straight line from start to goal.
        segments = [(0, len(path) - 1)]

        while segments:
            current_segment = segments.pop()
            ray_points = self.raytrace(path[current_segment[0]], path[current_segment[1]])
            if all(grid[p[0], p[1]] == 0 for p in ray_points):
                for p in [path[i] for i in range(current_segment[0] + 1, current_segment[1] - 1)]:
                    path.remove(p)
            else:  # if we're not successful, we halve the way into two equal segments and try again
                mid_id = current_segment[0] + math.floor((current_segment[1] - current_segment[0]) / 2)
                if current_segment[0] != mid_id:
                    segments.append((current_segment[0], mid_id))
                if mid_id + 1 != current_segment[1]:
                    segments.append((mid_id + 1, current_segment[1]))

        # Plot the execution path
        plt.imshow(grid, cmap='Greys', origin='lower')
        plt.plot(grid_start[1], grid_start[0], 'x')
        plt.plot(grid_goal[1], grid_goal[0], 'o')

        plt.title = "2D path seen from above"
        pruned_path = np.array(path)
        plt.plot(pruned_path[:, 1], pruned_path[:, 0], 'g')
        plt.scatter(pruned_path[:, 1], pruned_path[:, 0])
        plt.grid(True)
        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show()

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
