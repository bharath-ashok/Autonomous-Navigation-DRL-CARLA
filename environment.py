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
    sys.path.append(glob.glob('./carla/dist/carla-0.9.15-py3.7-win-amd64.egg')[0])
except IndexError:
    print("Couldn't import Carla egg properly")
import carla

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SECONDS_PER_EPISODE = 25
FIXED_DELTA_SECONDS = 0.02
SHOW_PREVIEW = True
NO_RENDERING = False
SYNCHRONOUS_MODE = True
SPIN = 10
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
        obs_dim = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([9, 4])

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.client.load_world('Town02')
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = NO_RENDERING
        self.settings.synchronous_mode = SYNCHRONOUS_MODE
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find('vehicle.tesla.model3')

        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()
        
        self.waypoints = self.generate_optimized_waypoints(3.0)
        waypoint_positions = np.array([(wp.transform.location.x, wp.transform.location.y) for wp in self.waypoints])
        self.waypoint_tree = KDTree(waypoint_positions)        
        self.start_waypoint = None
        self.dest_waypoint = None
        self.observation = None
        self.last_positions = []

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Failed to destroy actor {actor.id}: {e}")

    def next_waypoint(self, start_waypoint):
        next_wps = start_waypoint.next(2.0)
        if next_wps:
            return next_wps[0]
        else:
            # Handle the case where no next waypoint is found
            logger.warning('No next waypoint found.')
            return None

    def get_observation(self, vehicle_transform, dest_waypoint):
        veh_loc = vehicle_transform.location
        veh_rot = vehicle_transform.rotation
        wpt_loc = dest_waypoint.transform.location
        wpt_rot = dest_waypoint.transform.rotation

        distance_x = veh_loc.x - wpt_loc.x
        distance_y = veh_loc.y - wpt_loc.y
        direction_difference = (veh_rot.yaw - wpt_rot.yaw) % 360

        obs = np.array([
            distance_x,
            distance_y,
            direction_difference,
        ], dtype=np.float32)
        return obs

    def step(self, action):
        vehicle_transform = self.vehicle.get_transform()
        if self.SHOW_CAM:
            self.spectator.set_transform( 
                carla.Transform(vehicle_transform.location + carla.Location(z=20), carla.Rotation(yaw=-180, pitch=-90))
            )

        self.step_counter += 1
        steer, throttle = action

        # Map steering actions
        steer_mapping = {
            0: -0.9, 1: -0.25, 2: -0.1, 3: -0.05, 4: 0.0,
            5: 0.05, 6: 0.1, 7: 0.25, 8: 0.9
        }
        steer = steer_mapping.get(steer, 0.0)  # Default to 0.0 if steer is not in the mapping

        # Map throttle and apply control
        throttle_mapping = {
            0: (0.0, 1.0), 1: (0.3, 0.0), 2: (0.7, 0.0), 3: (1.0, 0.0)
        }
        throttle_val, brake_val = throttle_mapping.get(throttle, (1.0, 0.0))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_val, steer=steer, brake=brake_val))

        reward = 0
        done = False
        vehicle_location = vehicle_transform.location

        # --- Reward based on proximity to the next waypoint ---
        distance_to_waypoint = vehicle_location.distance(self.dest_waypoint.transform.location)        
        waypoint_threshold = 1.0  # You can adjust this threshold based on desired precision
        if distance_to_waypoint < waypoint_threshold:
            reward += 50  # Reward for reaching the waypoint
            self.dest_waypoint = self.next_waypoint(self.dest_waypoint)  # Set the next waypoint as destination
            logger.info(f'Passed waypoint, moving to next. Reward: {reward}')

        # --- Penalize for lack of movement ---
        # If the vehicle hasn't moved from its last recorded position, penalize it
        if self.last_positions:
            distance_moved = vehicle_location.distance(self.last_positions[-1])
            if distance_moved < 0.25:  # Movement threshold (adjust if necessary)
                reward -= 10  # Penalize for staying idle
                logger.info(f'No movement detected, penalizing. Current reward: {reward}')
        
        self.last_positions.append(vehicle_location)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)

        # --- Penalize for collisions ---
        if len(self.collision_hist) != 0:
            done = True
            reward -= 100  # Strong penalty for collisions
            logger.info('Collision detected, ending episode')
            self.cleanup()  
        # --- Check if episode time limit reached ---
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            reward -= 10  # Penalize slightly for not finishing the episode successfully
            logger.info(f'Reward: {reward}, Episode time limit reached')
            self.cleanup()

        # Update the observation
        self.observation = self.get_observation(vehicle_transform, self.dest_waypoint)

        return self.observation, reward, done, False, {}
    
    def collision_data(self, event):
        self.collision_hist.append(event)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []
        self.start_waypoint = None
        self.dest_waypoint = None

        try:
            self.vehicle = self.spawn_vehicle()
            self.actor_list.append(self.vehicle)
        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            raise
        time.sleep(0.1)

        # Retrieve initial vehicle position
        try:
            initial_pos = self.vehicle.get_transform().location
        except Exception as e:
            logger.error(f"Failed to get vehicle transform: {e}")
            raise

        logger.info(f'Initial position: {initial_pos}')

        # Initialize waypoints
        try:
            self.start_waypoint = self.initial_waypoint(initial_pos)
            self.dest_waypoint = self.start_waypoint.next(2.0)[0]
        except Exception as e:
            logger.error(f"Failed to initialize waypoints: {e}")
            raise

        if self.start_waypoint and self.dest_waypoint:
            logger.info(f'Current waypoint: {self.start_waypoint.transform.location}')
            logger.info(f'Target waypoint: {self.dest_waypoint.transform.location}')
        else:
            logger.error('Failed to set up waypoints.')
            raise RuntimeError('Waypoints initialization failed.')
        
        # Apply control to ensure the vehicle is stationary
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        try:
            camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
            if self.SHOW_CAM:
                trans = self.vehicle.get_transform()
                self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20), carla.Rotation(yaw=-180, pitch=-90)))

            # Initialize collision sensor
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))
        except Exception as e:
            logger.error(f"Failed to setup sensors or camera: {e}")
            raise

        # Initialize episode timing and tracking
        self.episode_start = time.time()
        self.step_counter = 0
        # Get the initial observation
        try:
            self.observation = self.get_observation(self.vehicle.get_transform(), self.dest_waypoint)
        except Exception as e:
            logger.error(f"Failed to get initial observation: {e}")
            raise

        self.last_positions = [initial_pos]
        return self.observation, {}
    
    def get_speed(self, v):
        if not isinstance(v, carla.Vector3D):
            logger.error("Invalid velocity vector provided.")
            return 0
        speed = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
        return speed

    
    def spawn_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("No spawn points available.")
            raise RuntimeError("No spawn points available.")

        transform = random.choice(spawn_points)
        logger.info(f'Spawning vehicle at: {transform.location}')

        vehicle = None
        while vehicle is None:
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
            except carla.CarlaException as e:
                logger.error(f"Failed to spawn vehicle: {e}")
                time.sleep(1)  # Brief delay before retrying

        return vehicle
    
    def initial_waypoint(self, initial_pos):
        # Query the KD-tree for the closest waypoint to initial_pos
        _, closest_index = self.waypoint_tree.query([initial_pos.x, initial_pos.y])
        closest_waypoint = self.waypoints[closest_index]
        return closest_waypoint

    def generate_optimized_waypoints(self, base_distance):
        """
        Generate waypoints with increased density at turns and intersections.
        """
        town_map = self.world.get_map()
        
        # Start by generating waypoints at a small interval
        fine_waypoints = town_map.generate_waypoints(0.5)
        optimized_waypoints = []

        for i in range(len(fine_waypoints) - 1):
            wp = fine_waypoints[i]
            next_wp = fine_waypoints[i + 1]
            
            optimized_waypoints.append(wp)  # Add current waypoint
            
            if self.is_turn_or_intersection(wp, next_wp, turn_threshold=20):
                # If a turn or intersection, keep the finer waypoint resolution
                continue
            
            # Skip waypoints to achieve the desired base distance
            if i % int(base_distance) != 0:
                optimized_waypoints.pop()  # Remove unnecessary waypoints
        
        return optimized_waypoints

    def is_turn_or_intersection(self, wp, next_wp, turn_threshold):
        """
        Detect if a waypoint is at a turn or intersection.
        """
        yaw_diff = abs(self.calculate_yaw_difference(wp.transform.rotation.yaw, next_wp.transform.rotation.yaw))
        return yaw_diff > turn_threshold or wp.is_intersection
    
    @staticmethod
    def calculate_yaw_difference(yaw1, yaw2):
        """
        Calculate the absolute difference in yaw, accounting for angle wrapping.
        """
        diff = abs(yaw1 - yaw2) % 360
        return diff if diff <= 180 else 360 - diff

    def get_lateral_distance(self, vehicle, waypoint):
        # Get vehicle and waypoint positions
        vehicle_location = vehicle.get_transform().location
        waypoint_location = waypoint.transform.location

        # Get the forward vector of the waypoint (direction the road is facing)
        waypoint_forward_vector = waypoint.transform.get_forward_vector()

        # Calculate the vector from the waypoint to the vehicle
        vehicle_vector = vehicle_location - waypoint_location

        # Convert the CARLA location objects to NumPy arrays for easier vector math
        waypoint_forward_np = np.array([waypoint_forward_vector.x, waypoint_forward_vector.y])
        vehicle_vector_np = np.array([vehicle_vector.x, vehicle_vector.y])

        # Normalize the waypoint's forward vector to get the direction
        waypoint_forward_np = waypoint_forward_np / np.linalg.norm(waypoint_forward_np)

        # Project the vehicle vector onto the waypoint's forward vector
        projection_length = np.dot(vehicle_vector_np, waypoint_forward_np)

        # Get the projection point (the closest point on the waypoint's direction line)
        projection_point = projection_length * waypoint_forward_np

        # Calculate the lateral distance as the distance between the vehicle's position and the projection point
        lateral_vector = vehicle_vector_np - projection_point
        lateral_distance = np.linalg.norm(lateral_vector)

        return lateral_distance
