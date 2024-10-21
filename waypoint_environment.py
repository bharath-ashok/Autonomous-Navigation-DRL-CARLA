import glob
import sys
import random
import time
import numpy as np
import logging
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import KDTree
sys.path.append('./carla')

try:
    sys.path.append(glob.glob('./carla/dist/carla-0.9.14-py3.7-win-amd64.egg')[0])
except IndexError:
    print("Couldn't import Carla egg properly")
import carla

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SECONDS_PER_EPISODE = 20
FIXED_DELTA_SECONDS = 0.01
SHOW_PREVIEW = False
NO_RENDERING = True
SYNCHRONOUS_MODE = True
HEIGHT = 480
WIDTH = 640

class CarEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    front_camera = None
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4
    im_width = WIDTH
    im_height = HEIGHT

    def __init__(self):
        super(CarEnv, self).__init__()
        self.actor_list = []
        self.vehicle = None
        self.collision_hist = []
        self.step_counter = 0
        self.simulation_time = 0
        self.spawn_position = None
        self.start_waypoint = None
        self.dest_waypoint = None
        self.observation = None
        
        obs_dim = 3
        low = np.array([-1, 0, -1], dtype=np.float32)
        high = np.array([1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([9, 4])

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(15.0)
        self.client.load_world('Town02')
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = NO_RENDERING
        self.settings.synchronous_mode = SYNCHRONOUS_MODE
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.world.tick()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find('vehicle.tesla.model3')

        # Waypoint system setup
        town_map = self.world.get_map()
        self.waypoints = town_map.generate_waypoints(3)
        waypoint_positions = np.array([(wp.transform.location.x, wp.transform.location.y) for wp in self.waypoints])
        self.waypoint_tree = KDTree(waypoint_positions)

        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Failed to destroy actor {actor.id}: {e}")

    def get_observation(self, vehicle_transform, dest_waypoint, velocity):
        speed = self.get_speed(velocity) if self.vehicle and self.vehicle.is_alive else 0
        lateral_distance = self.get_lateral_distance(vehicle_transform, dest_waypoint)
        heading = self.get_relative_heading(vehicle_transform, dest_waypoint)

        obs = np.array([lateral_distance, speed, heading], dtype=np.float32)
        return obs

    def step(self, action):
        self.step_counter += 1
        steer, throttle = action
        steer_mapping = {0: -0.9, 1: -0.25, 2: -0.1, 3: -0.05, 4: 0.0, 5: 0.05, 6: 0.1, 7: 0.25, 8: 0.9}
        steer = steer_mapping.get(steer, 0.0)
        throttle_mapping = {0: (0.0, 1.0), 1: (0.3, 0.0), 2: (0.7, 0.0), 3: (1.0, 0.0)}
        throttle_val, brake_val = throttle_mapping.get(throttle, (1.0, 0.0))

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_val, steer=steer, brake=brake_val))
        self.world.tick()

        velocity = self.vehicle.get_velocity()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        speed = self.get_speed(velocity)
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.dest_waypoint)

        reward = self.calculate_reward(speed, lateral_distance)
        done = False

        # Check proximity to waypoint
        distance_to_waypoint = vehicle_location.distance(self.dest_waypoint.transform.location)
        waypoint_threshold = 0.4
        if distance_to_waypoint < waypoint_threshold:
            reward += 10
            self.dest_waypoint = self.next_waypoint(self.dest_waypoint)
            logger.info(f'Passed waypoint, moving to next. Reward: {reward}')

        # Handle collisions
        if len(self.collision_hist) != 0:
            done = True
            reward -= 1000
            logger.info('Collision detected, ending episode')
            self.cleanup()

        # Time limit reached
        if self.simulation_time > SECONDS_PER_EPISODE:
            done = True
            reward += 100
            logger.info(f'Reward: {reward}, Episode time limit reached')
            self.cleanup()

        # Get new observation
        if not done:
            self.observation = self.get_observation(vehicle_transform, self.dest_waypoint, velocity)

        return self.observation, reward, done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []
        self.simulation_time = 0

        self.vehicle = self.spawn_vehicle()
        self.actor_list.append(self.vehicle)
        self.world.tick()

        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        self.start_waypoint = self.initial_waypoint_location(self.spawn_position)
        self.dest_waypoint = self.next_waypoint(self.start_waypoint)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        self.observation = self.get_observation(vehicle_transform, self.dest_waypoint, velocity)
        return self.observation, {}

    def spawn_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)
        vehicle = None
        while vehicle is None:
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
            except Exception as e:
                logger.error(f"Failed to spawn vehicle: {e}")
                time.sleep(0.2)

        logger.info(f'Spawning vehicle at: {transform.location}')
        self.spawn_position = transform.location
        return vehicle

    def initial_waypoint_location(self, initial_pos):
        _, closest_index = self.waypoint_tree.query([initial_pos.x, initial_pos.y])
        closest_waypoint = self.waypoints[closest_index]
        return closest_waypoint

    def next_waypoint(self, start_waypoint, look_ahead=3.0):
        next_wps = start_waypoint.next(look_ahead)
        return next_wps[0] if next_wps else None

    def get_speed(self, v, max_speed=60.0):
        speed = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
        return np.clip(speed / max_speed, -1, 1)

    def get_lateral_distance(self, vehicle_transform, waypoint, max_distance=3.0):
        waypoint_location = waypoint.transform.location
        waypoint_forward_vector = waypoint.transform.get_forward_vector()
        vehicle_location = vehicle_transform.location
        vehicle_vector = np.array([vehicle_location.x - waypoint_location.x, vehicle_location.y - waypoint_location.y])
        forward_vector = np.array([waypoint_forward_vector.x, waypoint_forward_vector.y])

        lateral_distance = np.cross(forward_vector, vehicle_vector) / np.linalg.norm(forward_vector)
        return np.clip(lateral_distance / max_distance, -1, 1)

    def get_relative_heading(self, vehicle_transform, waypoint):
        vehicle_yaw = vehicle_transform.rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw
        relative_heading = (waypoint_yaw - vehicle_yaw + 180) % 360 - 180
        return np.clip(relative_heading / 180.0, -1, 1)

    def collision_data(self, event):
        if len(self.collision_hist) >= 10:
            self.collision_hist.pop(0)
        self.collision_hist.append(event)
    
    def calculate_reward(self, speed, lateral_distance):
        reward = speed * (1 - abs(lateral_distance))
        return reward
