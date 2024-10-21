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
        low = np.array([-1, 0, -1], dtype=np.float32)  # Set speed to [0, 1] and other values to [-1, 1]
        high = np.array([1, 1, 1], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([9, 4])  # No change to the action space


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

        if self.SHOW_CAM:
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

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")

    def get_observation(self, vehicle_transform, dest_waypoint, velocity):

        if self.vehicle and self.vehicle.is_alive:
            speed = self.get_speed(velocity)
        else:
            speed = 0  # Default speed if vehicle is not available

        lateral_distance = self.get_lateral_distance(vehicle_transform, dest_waypoint)
        heading = self.get_relative_heading(vehicle_transform, dest_waypoint)

        if self.step_counter % 100 == 0:
            logger.info(f"lateral_distance: {lateral_distance}, speed: {speed}, heading: {heading}")
        obs = np.array([
            lateral_distance,
            speed,
            heading,
        ], dtype=np.float32)
        return obs

    def step(self, action):
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
        
        self.world.tick()
        self.simulation_time += FIXED_DELTA_SECONDS
        velocity = self.vehicle.get_velocity()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        speed = self.get_speed(velocity)
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.dest_waypoint)
        relative_heading = self.get_relative_heading(vehicle_transform, self.dest_waypoint)

        reward = self.calculate_reward(speed, lateral_distance, relative_heading)
        done = False
  
        # --- Reward based on proximity to the next waypoint ---
        distance_to_waypoint = vehicle_location.distance(self.dest_waypoint.transform.location)        
        waypoint_threshold = 0.4  # You can adjust this threshold based on desired precision
        if distance_to_waypoint < waypoint_threshold:
            #reward += 10  # Reward for reaching the waypoint
            self.dest_waypoint = self.next_waypoint(self.dest_waypoint)  # Set the next waypoint as destination
            logger.info(f'Passed waypoint. Reward: {reward}')

        # --- Penalize for collisions ---
        if len(self.collision_hist) != 0:
            reward -= 100  # Strong penalty for collisions
            self.vehicle.set_transform(carla.Transform(self.start_waypoint.transform.location))
            self.dest_waypoint = self.next_waypoint(self.start_waypoint)
            logger.info('Collision detected, vehicle reset.') 

        # --- Check if episode time limit reached ---
        if self.simulation_time > SECONDS_PER_EPISODE:
            done = True
            reward += 100  # Penalize slightly for not finishing the episode successfully
            logger.info(f'Reward: {reward}, Episode time limit reached')
            self.cleanup()
        # Update the observation
        if not done:
            self.observation = self.get_observation(vehicle_transform, self.dest_waypoint, velocity)
            if self.step_counter % 100 == 0:
                logger.info(f"vehicle_transform: {vehicle_transform}, self.dest_waypoint: {self.dest_waypoint}, velocity: {velocity}")
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
        self.world.tick()
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
        try:
            #camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
            if self.SHOW_CAM:
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
            logger.error("No spawn points available.")
            raise RuntimeError("No spawn points available.")

        transform = random.choice(spawn_points)
        vehicle = None
        while vehicle is None:
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
            except Exception as e:
                logger.error(f"Failed to spawn vehicle: {e}")
                time.sleep(0.2)  # Brief delay before retrying

            logger.info(f'Spawning vehicle at: {transform.location}')
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
    
    def get_speed(self, v, max_speed=45.0):
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

        # Convert CARLA location objects to NumPy arrays for easier vector calculations
        waypoint_location_np = np.array([waypoint_location.x, waypoint_location.y])
        waypoint_forward_np = np.array([waypoint_forward_vector.x, waypoint_forward_vector.y])
        vehicle_location_np = np.array([vehicle_location.x, vehicle_location.y])

        # Normalize the waypoint's forward vector (this is the direction of the lane)
        norm = np.linalg.norm(waypoint_forward_np)
        if norm == 0:
            return 0  # Prevent division by zero if the forward vector length is zero
        waypoint_forward_np = waypoint_forward_np / norm

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

    
    def get_relative_heading(self, vehicle_transform, waypoint):
        # Extract vehicle and waypoint yaw in degrees
        vehicle_yaw = vehicle_transform.rotation.yaw
        current_waypoint = waypoint  # Start from the current waypoint
        num_lookahead = 5  # Number of waypoints to look ahead
        distance_lookahead = 3.0
        yaw_differences_sum = 0  # Accumulator for the sum of yaw differences
        turn_direction = self.turn_directon(vehicle_transform, waypoint, num_lookahead, distance_lookahead)
        
        for i in range(1, num_lookahead + 1):
            # Fetch the next waypoint 'i' steps ahead
            future_waypoint = self.next_waypoint(current_waypoint, distance_lookahead)
            
            if not future_waypoint:
                break  # Stop if there's no more valid waypoints
        
            # Check if the current waypoint is part of a turn
            future_yaw = future_waypoint.transform.rotation.yaw
            yaw_diff = abs(current_waypoint.transform.rotation.yaw - future_yaw) % 360
            yaw_diff = yaw_diff if yaw_diff <= 180 else 360 - yaw_diff
            
            # Accumulate the yaw difference
            yaw_differences_sum += yaw_diff
            
            # Move to the next waypoint for the next iteration
            current_waypoint = future_waypoint

        vehicle_yaw = vehicle_yaw % 360            
        # Calculate relative heading as the difference between the yaw sum and the vehicle's yaw
        relative_heading = (yaw_differences_sum - vehicle_yaw) % 360
        
        # Normalize to [-180, 180] range
        if relative_heading > 180:
            relative_heading -= 360

        # Adjust the heading based on the turn direction
        normalized_heading = turn_direction * (relative_heading / 180.0)
        return np.clip(normalized_heading, -1, 1)

    def turn_directon(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=3.0):
        # Extract yaw angles from waypoints
        vehicle_yaw = vehicle_transform.rotation.yaw  # Vehicle's current yaw
        current_waypoint = waypoint  # Start from the current waypoint
        future_waypoint = None
        for i in range(1, num_lookahead + 1):
            future_waypoint = self.next_waypoint(current_waypoint, distance_lookahead)
            if not future_waypoint:
                break  # Stop if there's no valid future waypoint
            current_waypoint = future_waypoint  # Move to the next waypoint

        if future_waypoint:
            # Get the yaw of the future waypoint
            future_yaw = future_waypoint.transform.rotation.yaw
            
            # Calculate the yaw difference between the vehicle and the future waypoint
            yaw_diff = (future_yaw - vehicle_yaw) % 360
            
            # Normalize the yaw difference to the [-180, 180] range
            if yaw_diff > 180:
                yaw_diff -= 360
            if yaw_diff > 10:  # If the future yaw is greater than the vehicle's yaw by a threshold
                return 1
            else:
                return -1
    
    def calculate_reward(self, speed, lateral_distance, relative_heading):
        reward = 0        
        # Penalize the vehicle for not moving or moving very slowly
        if speed <= 0.1:  # Consider speed close to zero as 'not moving'
            reward -= 10.0  # Severe penalty for not moving
        else:
            # Reward based on normalized speed (in the range [0, 1])
            if 0.33 <= speed <= 0.67:  # Speed range of about 15-30 km/h
                reward += 5.0
            elif speed > 0.67:  # Over-speeding beyond 40 km/h
                reward -= (speed - 0.67) ** 2 / 100  # Penalize for going too fast
            else:  # Speed is too slow but non-zero
                reward -= (0.33 - speed) ** 2 / 100  # Penalize for moving too slowly
            
        # Reward for lateral distance
        lateral_threshold = 0.2
        if abs(lateral_distance) < lateral_threshold:
            reward += 5  # Strong reward for being within the desired lateral distance
        else:
            # Penalize based on distance from the center line
            lateral_penalty = 5 / (1 + np.exp(-10 * (abs(lateral_distance) - lateral_threshold)))
            reward -= lateral_penalty

        relative_heading_threshold = 0.1
        if abs(relative_heading) < relative_heading_threshold:
            reward += 5
        else:
            heading_penalty = 5 / (1 + np.exp(-10 * (abs(relative_heading) - relative_heading_threshold)))
            reward -= heading_penalty

        return reward


#  BELOW NOT USED -->>
    def get_future_headings(self, vehicle_transform, waypoints):
        headings = []
        for waypoint in waypoints[:3]:  # Get the heading to the next 3 waypoints
            heading = self.get_relative_heading(vehicle_transform, waypoint.transform)
            headings.append(self.normalize_heading_angle(heading))
        return headings  # Return a list of normalized headings

    def normalize_heading_angle(self, relative_heading):
        # Normalize the heading angle to the range [-1, 1]
        return relative_heading / 180.0
    