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

SECONDS_PER_EPISODE = 25
FIXED_DELTA_SECONDS = 0.01
SYNCHRONOUS_MODE = True
SPIN = 10
HEIGHT = 480
WIDTH = 640

class CarEnv(gym.Env):
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
        low = np.array([-1, 0, -1], dtype=np.float32)  # Set speed to [0, 1] and other values to [-1, 1]
        high = np.array([1, 1, 1], dtype=np.float32)
        self.NO_RENDERING = False

        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([15, 7])  # No change to the action space

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(15.0)
        self.client.load_world('Town02')
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = self.NO_RENDERING
        self.settings.synchronous_mode = SYNCHRONOUS_MODE
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.world.tick()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find('vehicle.tesla.model3')

        self.spectator = None # Initialize the spectator
        if not self.NO_RENDERING:
            self.spectator = self.world.get_spectator()
        
        town_map = self.world.get_map()
        self.waypoints = town_map.generate_waypoints(3)
        waypoint_positions = np.array([(wp.transform.location.x, wp.transform.location.y) for wp in self.waypoints])
        self.waypoint_tree = KDTree(waypoint_positions)
        self.spawn_position = None        
        self.start_waypoint = None
        self.dest_waypoint = None
        self.observation = None
        self.episode = 0
        self.waypoint_threshold = 2.0
        self.debug = False

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")
        self.world.tick()

    def get_observation(self, vehicle_transform, dest_waypoint, velocity):

        if self.vehicle and self.vehicle.is_alive:
            speed = self.get_speed(velocity)
        else:
            speed = 0  # Default speed if vehicle is not available

        lateral_distance = self.get_lateral_distance(vehicle_transform, dest_waypoint)
        heading = self.get_relative_heading(vehicle_transform, dest_waypoint)

        if self.step_counter % 100 == 0 and self.debug:
            logger.info(f"lateral_distance: {lateral_distance}, speed: {speed}, heading: {heading}")
        obs = np.array([
            lateral_distance,
            speed,
            heading,
        ], dtype=np.float32)
        if self.step_counter % 100 == 0:
             print(f"Observation: {obs}")
        return obs

    def step(self, action):
        self.step_counter += 1
        self.world.tick()
        self.simulation_time += FIXED_DELTA_SECONDS

        steer, throttle = action
        # Map steering actions
        # Generate equally spaced values from -1 to 1 for steering
        steer_values = np.linspace(-1, 1, 15)
        steer_mapping = {i: steer_values[i] for i in range(15)}
        if steer not in steer_mapping:
            logger.error(f"Invalid steer value: {steer}. Defaulting to 0.0")
        steer = steer_mapping.get(steer, 0.0)  # Default to 0.0 if steer is not in the mapping

        # Map throttle and apply control
        throttle_mapping = {
            0: (0.0, 1.0), 1: (0.0, 0.0), 2: (0.2, 0.0), 3: (0.4, 0.0), 4: (0.6, 0.0), 5: (0.8, 0.0), 6: (1.0, 0.0)
        }
        throttle_val, brake_val = throttle_mapping.get(throttle, (1.0, 0.0))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_val, steer=steer, brake=brake_val))
        if self.step_counter % 100 == 0 and self.debug:
            print(f"Action: {action}, Steer: {steer}, Throttle: {throttle_val}, Brake: {brake_val}")
        
        velocity = self.vehicle.get_velocity()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        speed = self.get_speed(velocity)
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.dest_waypoint)
        relative_yaw = self.get_relative_heading(vehicle_transform, self.dest_waypoint)
        if self.step_counter % 100 == 0 and self.debug:
            print(f"Vehicle transform: {vehicle_transform}, Yaw:{vehicle_transform.rotation.yaw}, WP_Yaw:{self.dest_waypoint.transform.rotation.yaw} relative_yaw: {relative_yaw}")
            print(f"destination waypoint: {self.dest_waypoint.transform}")
        reward = self.calculate_reward(speed, lateral_distance, relative_yaw)
        done = False
        # --- Reward based on proximity to the next waypoint ---
        distance_to_waypoint = vehicle_location.distance(self.dest_waypoint.transform.location)        
        if distance_to_waypoint < self.waypoint_threshold:
            if distance_to_waypoint < 0.75:
                reward += 3  # Reward for reaching the waypoint
            self.dest_waypoint = self.next_waypoint(self.dest_waypoint)  # Set the next waypoint as destination
            # self.world.debug.draw_string(self.dest_waypoint.transform.location, 'O', draw_shadow=False,
            #     color=carla.Color(255, 0, 0), life_time=5, persistent_lines=True)
            logger.info(f'Passed waypoint. Reward: {reward}, Next waypoint: {self.dest_waypoint.transform.location}')

        elif distance_to_waypoint > 3:
            reward -= 5  # Strong penalty for going far away from the waypoint
            waypoint = self.initial_waypoint_location(vehicle_location)
            self.dest_waypoint = self.next_waypoint(waypoint)  # Set the next waypoint as destination
            # Update the destination waypoint to the next waypoint from the starting waypoint
            if self.dest_waypoint is None:
                logger.error('No valid next waypoint found, terminating episode.')
                done = True  # Set done to True if no valid waypoint is available

        # --- Penalize for collisions ---
        if len(self.collision_hist) != 0:
            reward -= 100  # Strong penalty for collisions
            logger.info('Collision detected, resetting vehicle.')
            # Reset vehicle position to the starting waypoint
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            self.vehicle.set_transform(self.start_waypoint.transform)
            time.sleep(0.1)  # Brief delay to ensure the reset takes effect
            # Update the destination waypoint to the next waypoint from the starting waypoint
            self.dest_waypoint = self.next_waypoint(self.start_waypoint)
            logger.info(f"Next waypoint after collision: {self.dest_waypoint.transform.location}")
            self.collision_hist.clear()

        # --- Check if episode time limit reached ---
        if self.simulation_time > SECONDS_PER_EPISODE:
            done = True
            logger.info(f'Episode time limit reached')
            self.cleanup()
        # Update the observation
        if not done:
            self.observation = self.get_observation(vehicle_transform, self.dest_waypoint, velocity)
            if self.step_counter % 100 == 0:
                print(f"Reward: {reward}")
        return self.observation, reward, done, False, {}
    
    def collision_data(self, event):
        if len(self.collision_hist) >= 10:  # Keep only the last 10 collisions
            self.collision_hist.pop(0)
        self.collision_hist.append(event)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []
        self.start_waypoint = None
        self.dest_waypoint = None
        self.simulation_time = 0
        self.episode += 1
        logger.info(f"Starting episode {self.episode}")

        self.vehicle = self.spawn_vehicle()
        self.actor_list.append(self.vehicle)
        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        # Initialize waypoints
        try:
            self.start_waypoint = self.initial_waypoint_location(self.spawn_position)
            self.dest_waypoint = self.next_waypoint(self.start_waypoint)

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
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        self.world.tick()
        try:
            #camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
            if self.spectator is not None:
                spectator_transform =  self.vehicle.get_transform()
                spectator_transform.rotation.pitch -= 60.0
                spectator_transform.location += carla.Location( z = 15.0)
                self.spectator.set_transform(spectator_transform)

            # Initialize collision sensor
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))
        except Exception as e:
            logger.error(f"Failed to setup sensors or camera: {e}")
            raise

        # Initialize episode timing and tracking
        self.step_counter = 0
        # Get the initial observation
        try:
            self.observation = self.get_observation(vehicle_transform, self.dest_waypoint, velocity)
        except Exception as e:
            logger.error(f"Failed to get initial observation: {e}")
            raise
        return self.observation, {}
    
    def spawn_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available.")
        # respawn = [35,36,17,30,43,9,4,22,40,76,10,21,29,80,59,74,10,64,68,81,51,75]
        # spawn_points = [spawn_points[i] for i in respawn]   
        transform = random.choice(spawn_points)
        vehicle = None
        retries = 10  # or any reasonable limit
        while vehicle is None and retries > 0:
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
            except Exception as e:
                retries -= 1
                logger.error(f"Failed to spawn vehicle: {e}")
                time.sleep(0.2)  # Brief delay before retrying
        if vehicle is None:
            raise RuntimeError("Vehicle could not be spawned after multiple attempts.")
        logger.info(f'Spawned vehicle at: {transform.location}')
        self.spawn_position = transform.location
        return vehicle
    
    def initial_waypoint_location(self, initial_pos):
        # Query the KD-tree for the closest waypoint to initial_pos
        _, closest_index = self.waypoint_tree.query([initial_pos.x, initial_pos.y])
        closest_waypoint = self.waypoints[closest_index]
        return closest_waypoint
   
    def next_waypoint(self, start_waypoint, look_ahead=3.0):
        next_wps = start_waypoint.next(look_ahead)
        if next_wps:
            return next_wps[0]
        else:
            # Handle the case where no next waypoint is found
            logger.warning('No next waypoint found.')
            return None
    
    def get_speed(self, v, max_speed=60.0):
        if not isinstance(v, carla.Vector3D):
            logger.error("Invalid velocity vector provided.")
            return 0
        speed = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
        normalized_speed = np.clip(speed / max_speed, -1, 1)  # Normalize speed to [-1, 1]
        return normalized_speed

    def get_lateral_distance(self, vehicle_transform, waypoint, max_distance=3.0):
        # Get vehicle and waypoint positions
        waypoint_location = waypoint.transform.location
        waypoint_forward_vector = waypoint.transform.get_forward_vector()
        vehicle_location = vehicle_transform.location

        # Convert CARLA location objects to NumPy arrays
        waypoint_location_np = np.array([waypoint_location.x, waypoint_location.y])
        waypoint_forward_np = np.array([waypoint_forward_vector.x, waypoint_forward_vector.y])
        vehicle_location_np = np.array([vehicle_location.x, vehicle_location.y])

        # Normalize the waypoint's forward vector (this is the direction of the lane)
        norm = np.linalg.norm(waypoint_forward_np)
        if norm == 0:
            return 0  # Prevent division by zero if the forward vector length is zero
        waypoint_forward_np /= norm

            # Vector from the waypoint (center of the lane) to the vehicle
        vehicle_vector_np = vehicle_location_np - waypoint_location_np

        # Project the vehicle vector onto the waypoint's forward vector (lane direction)
        projection_length = np.dot(vehicle_vector_np, waypoint_forward_np)
        projection_point = projection_length * waypoint_forward_np

        lateral_vector = vehicle_vector_np - projection_point
        lateral_distance = np.linalg.norm(lateral_vector)

        # Normalize the lateral distance to the range [-1, 1]
        normalized_lateral_distance = np.clip(lateral_distance / max_distance, -1, 1)

        # Check if the vehicle is to the left or right of the lane center by using a cross product
        cross_product = np.cross(waypoint_forward_np, vehicle_vector_np)
        if cross_product < 0:
            # If the cross product is negative, the vehicle is to the right of the lane
            normalized_lateral_distance *= -1

        return normalized_lateral_distance

    def get_relative_heading(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=3.0):
        # Extract vehicle and waypoint yaw in degrees
        vehicle_yaw = vehicle_transform.rotation.yaw % 360
        waypoint_yaw = waypoint.transform.rotation.yaw % 360
        
        current_waypoint = waypoint  # Start from the current waypoint
        yaw_differences_sum = 0  # Accumulator for the sum of yaw differences
        turn_direction = self.turn_direction(vehicle_transform, waypoint, num_lookahead, distance_lookahead)
        
        # Sum up yaw differences from the current waypoint and future waypoints
        for i in range(1, num_lookahead + 1):
            future_waypoint = self.next_waypoint(current_waypoint, distance_lookahead)
            if not future_waypoint:
                break  # Stop if no valid future waypoint
            
            future_yaw = future_waypoint.transform.rotation.yaw 
            yaw_diff = abs(current_waypoint.transform.rotation.yaw - future_yaw) % 360
            yaw_diff = yaw_diff if yaw_diff <= 180 else 360 - yaw_diff
            
            # Accumulate the yaw difference
            yaw_differences_sum += yaw_diff
            current_waypoint = future_waypoint

        # Normalize vehicle yaw and relative heading
        vehicle_yaw = vehicle_yaw % 360     
        relative_yaw =(vehicle_yaw - waypoint_yaw) % 360
        if turn_direction != 0:
            relative_heading = (yaw_differences_sum - turn_direction * relative_yaw) % 360
        else:
            relative_heading = (yaw_differences_sum -  relative_yaw) % 360
        # Normalize to [-180, 180] range
        if relative_heading > 180:
            relative_heading -= 360

        if turn_direction != 0:
            normalized_heading = turn_direction * (relative_heading / 180.0)
        else:
            # If turn direction is 0 (i.e., straight), we just normalize relative_heading without adjustment
            normalized_heading = relative_heading / 180.0
        
        return np.clip(normalized_heading, -1, 1)

    def turn_direction(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=3.0):
        vehicle_yaw = vehicle_transform.rotation.yaw % 360
        current_waypoint = waypoint

        # Look ahead at future waypoints
        for i in range(1, num_lookahead + 1):
            future_waypoint = self.next_waypoint(current_waypoint, distance_lookahead)
            if not future_waypoint:
                break
            current_waypoint = future_waypoint

        # Check if there is a valid future waypoint
        if future_waypoint:
            future_yaw = future_waypoint.transform.rotation.yaw % 360
            yaw_diff = (future_yaw - vehicle_yaw + 180) % 360 - 180
            
            # Return direction: 1 for left (counterclockwise), -1 for right (clockwise)
            return 1 if yaw_diff > 0 else -1
        return 0
    
    def calculate_reward(self, speed, lateral_distance, relative_yaw):
        reward = 0

        # Penalize for not moving or moving very slowly
        if speed <= 0.1:  # Vehicle almost stationary
            reward -= 10.0  # Stronger penalty for being stationary
        else:
            # Reward for maintaining a speed within 10-25 km/h (normalized speed 0.17 - 0.42)
            if 0.17 <= speed <= 0.34:
                reward += 0
            elif speed > 0.34:  # Penalize over-speeding (beyond ~30 km/h)
                reward -= (speed - 0.34) ** 2 / 50  # Higher penalty for over-speeding
            else:  # Penalize for going too slowly
                reward -= (0.17 - speed) ** 2 / 50

        # Penalize for deviation in lateral distance from the center line
        lateral_threshold = 0.1 # Threshold for lateral distance
        if abs(lateral_distance) < lateral_threshold:
            reward += 0  # Reward for staying within lateral threshold
        else:
            # Exponential penalty that increases sharply as lateral distance increases
            lateral_penalty = 10 / (1 + np.exp(-15 * (abs(lateral_distance) - lateral_threshold)))
            reward -= lateral_penalty

        # Penalize for deviation in heading (relative yaw)
        relative_heading_threshold = 10  # Degrees
        if abs(relative_yaw) < relative_heading_threshold:
            reward += 0 # Reward for maintaining heading within threshold
        else:
            # Exponential penalty for large deviations in heading
            heading_penalty = 10 / (1 + np.exp(-15 * (abs(relative_yaw) - relative_heading_threshold)))
            reward -= heading_penalty

        return reward

    def calculate_relative_yaw(self, vehicle_transform, waypoint):
        # Extract yaw angles from vehicle and waypoint
        vehicle_yaw = vehicle_transform.rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw

        # Calculate the relative yaw, keeping it within the range [-180, 180] to avoid discontinuities
        relative_yaw = (vehicle_yaw - waypoint_yaw + 180) % 360 - 180
        return relative_yaw
