import glob
import sys
import random
import time
import math
import numpy as np
import logging
import gymnasium as gym
from gymnasium import spaces
from threading import Event

try:
    sys.path.append(glob.glob('./carla/dist/carla-0.9.14-py3.7-win-amd64.egg')[0])
except IndexError:
    print("Couldn't import Carla egg properly")
import carla
try:
    sys.path.insert(0,r'C:\Users\ashokkumar\source\repos\AD\carla')
except IndexError:
    pass
from agents.navigation.global_route_planner import GlobalRoutePlanner   


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SECONDS_PER_EPISODE = 25
FIXED_DELTA_SECONDS = 0.01
NO_RENDERING = True
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
        obs_dim = 4
        low = np.array([-1, 0, -1, -1], dtype=np.float32)  # Set speed to [0, 1] and other values to [-1, 1]
        high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

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

        self.spectator = None # Initialize the spectator
        if not NO_RENDERING:
            self.spectator = self.world.get_spectator()
        
        ## Route planner
        self.map = self.world.get_map()
        self.sampling_resolution = 1.0
        self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution=self.sampling_resolution)
        self.spawn_points = self.map.get_spawn_points()      
        self.route = None  
        self.start_waypoint = None
        self.dest_waypoint = None
        self.curr_waypoint = None
        self.previous_waypoints = []
        self.observation = None
        self.debug = False

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")
        self.world.tick()

    def get_observation(self, vehicle_transform, curr_waypoint, velocity):

        if self.vehicle and self.vehicle.is_alive:
            speed = self.get_speed(velocity)
        else:
            speed = 0  # Default speed if vehicle is not available
        lateral_distance = self.get_lateral_distance(vehicle_transform, curr_waypoint)
        heading = self.get_relative_heading(vehicle_transform, curr_waypoint)
        normalized_yaw = self.calculate_relative_yaw(vehicle_transform, curr_waypoint)
        obs = np.array([
            lateral_distance,
            speed,
            heading,
            normalized_yaw
        ], dtype=np.float32)
        if self.step_counter % 100 == 0:
             print(f"Observation: {obs}")
        return obs

    def step(self, action):
        self.step_counter += 1
        self.world.tick()
        self.simulation_time += FIXED_DELTA_SECONDS
        steer, throttle = action
        if throttle < 0:
            brake_val = -throttle
            throttle_val = 0
        else:
            brake_val = 0
            throttle_val = throttle
        steer = float(steer)
        throttle_val = float(throttle_val)
        brake_val = float(brake_val)
        ## TODO in BatchSync
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_val, steer=steer, brake=brake_val))
        if self.step_counter % 100 == 0:
            print(f"Action: {action}")
        velocity = self.vehicle.get_velocity()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        speed = self.get_speed(velocity)
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.curr_waypoint)
        relative_heading = self.get_relative_heading(vehicle_transform, self.curr_waypoint)
        # print(f"curren_waypoint: {self.curr_waypoint.transform.location}")
        # print(f"Speed: {speed}, Lateral distance: {lateral_distance}, Relative heading: {relative_heading}")
        if self.step_counter % 100 == 0 and self.debug:
            print(f"Vehicle transform: {vehicle_transform}, Yaw:{vehicle_transform.rotation.yaw}, WP_Yaw:{self.curr_waypoint.transform.rotation.yaw} relative_yaw: {relative_heading}")
            print(f"destination waypoint: {self.curr_waypoint.transform.location}")
        reward = self.calculate_reward(speed, lateral_distance, relative_heading, vehicle_location)
        done = False

        # # --- Reward based on proximity to the waypoint ---
        # distance_to_waypoint = vehicle_location.distance(self.curr_waypoint.transform.location)        
        # # print(f"Distance to waypoint: {distance_to_waypoint}")
        # if distance_to_waypoint < 0.5:
        #     reward += 3  # Reward for reaching the waypoint
        if self.find_current_waypoint(vehicle_transform.location) is not None:
            self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)
        # self.world.debug.draw_string(self.curr_waypoint.transform.location, 'O', draw_shadow=False,
        #     color=carla.Color(255, 0, 0), life_time=5, persistent_lines=True)
        if self.curr_waypoint is self.dest_waypoint:
            try:
                self.route = self.generate_route(start_waypoint=self.curr_waypoint)
            except Exception as e:
                logger.error(f"Failed to generate route: {e}")
                raise

        # --- Penalize for collisions ---
        if len(self.collision_hist) != 0:
            reward -= 100.0
            self.handle_collision()
            self.collision_hist.clear()

        # --- Check if episode time limit reached ---
        if self.simulation_time > SECONDS_PER_EPISODE:
            done = True
            logger.info(f'Episode time limit reached')
            self.cleanup()
        # Update the observation
        if not done:
            self.observation = self.get_observation(vehicle_transform, self.curr_waypoint, velocity)
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
        self.previous_waypoints = []
        self.route = None
        self.start_waypoint = None
        self.curr_waypoint = None
        self.dest_waypoint = None
        self.simulation_time = 0
        while self.route is None:
            self.route = self.generate_route()
            if self.route is not None:
                logger.info(f'route length: {len(self.route)}, distance: {self.sampling_resolution*len(self.route)}')
            else:
                logger.error('Failed to generate route. Retrying...')
        self.vehicle = self.spawn_vehicle()
        self.actor_list.append(self.vehicle)
        self.world.tick()
        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        # Apply control to ensure the vehicle is stationary
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        if self.route is not None:
            self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)
            # print(f"Current waypoint: {self.curr_waypoint.transform.location}")
        else:
            logger.error('Failed to find current waypoint.')
            raise RuntimeError('Waypoints initialization failed.')
        try:
            #camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
            if self.spectator is not None:
                spectator_transform =  self.vehicle.get_transform()
                spectator_transform.rotation.pitch -= 60.0
                spectator_transform.location += carla.Location( z = 25.0)
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
            self.observation = self.get_observation(vehicle_transform, self.curr_waypoint, velocity)
        except Exception as e:
            logger.error(f"Failed to get initial observation: {e}")
            raise
        return self.observation, {}
    
    def spawn_vehicle(self):
        transform = self.start_waypoint
        vehicle = None
        retries = 10  # or any reasonable limit
        while vehicle is None and retries > 0:
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
            except Exception as e:
                retries -= 1
                logger.error(f"Failed to spawn vehicle: {e}")
                time.sleep(0.4)  # Brief delay before retrying
        if vehicle is None:
            raise RuntimeError("Vehicle could not be spawned after multiple attempts.")
        logger.info(f'Spawned vehicle at: {transform.location}')
        return vehicle
    
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

    def get_relative_heading(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=1):

        current_waypoint = waypoint  # Start from the current waypoint
        yaw_differences_sum = 0  # Accumulator for the sum of yaw differences
        turn_direction = self.turn_direction(vehicle_transform, waypoint, num_lookahead, distance_lookahead)
        
        # Sum up yaw differences from the current waypoint and future waypoints
        for i in range(1, num_lookahead + 1):
            future_waypoint = self.find_next_waypoint(current_waypoint)
            if not future_waypoint:
                break  # Stop if no valid future waypoint
            
            future_yaw = future_waypoint.transform.rotation.yaw 
            yaw_diff = abs(current_waypoint.transform.rotation.yaw - future_yaw) 
            yaw_diff = yaw_diff if yaw_diff <= 180 else 360 - yaw_diff
            
            # Accumulate the yaw difference
            yaw_differences_sum += yaw_diff
            current_waypoint = future_waypoint

        # Normalize vehicle yaw and relative heading
        # vehicle_yaw = vehicle_yaw % 360     
        # relative_yaw =(vehicle_yaw - waypoint_yaw) % 360
        # For now only consider the waypoint yaw
        # relative_yaw =  0  # relative_yaw if relative_yaw <= 180 else 360 - relative_yaw
        # if turn_direction != 0:
        #     relative_heading = (yaw_differences_sum - turn_direction * relative_yaw) % 360
        # else:
        #     relative_heading = (yaw_differences_sum -  relative_yaw) % 360
        # # Normalize to [-180, 180] range
        relative_heading = (yaw_differences_sum) 
        if relative_heading > 180:
            relative_heading -= 360

        normalized_heading = turn_direction * (relative_heading / 180.0)

        return np.clip(normalized_heading, -1, 1)

    def turn_direction(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=1):
        vehicle_yaw = vehicle_transform.rotation.yaw % 360
        current_waypoint = waypoint

        # Look ahead at future waypoints
        for _ in range(num_lookahead):
            future_waypoint = self.find_next_waypoint(current_waypoint, distance_lookahead)
            if not future_waypoint:
                break
            current_waypoint = future_waypoint

        # Check if there is a valid future waypoint
        if future_waypoint:
            future_yaw = future_waypoint.transform.rotation.yaw % 360
            yaw_diff = (future_yaw - vehicle_yaw + 180) % 360 - 180
            
            # Return direction: 1 for left (counterclockwise), -1 for right (clockwise)
        return 1 if yaw_diff >= 0 else -1
    
    def calculate_reward(self, speed, lateral_distance, relative_heading, vehicle_location):
        reward = 0.5  # Baseline reward to encourage movement

        # Strong penalty for being stationary
        if speed == 0.0:
            reward -= 5.0  # Reduced penalty to avoid discouraging initial movement

        k = 20  # Steepness of the sigmoid curve

        if 0.17 <= speed <= 0.42:  # Desired speed range (10-25 km/h)
            reward += 2.0  # Reward for staying within optimal speed range
        elif speed < 0.17:  # Underspeed penalty
            reward += -2 / (1 + np.exp(-k * (0.17 - speed)))
        elif speed > 0.42:  # Overspeed penalty
            reward += -2 / (1 + np.exp(-k * (speed - 0.42)))


        # Penalize for lateral distance from the center line
        lateral_penalty = 15 / (1 + np.exp(-15 * abs(lateral_distance)))
        reward -= lateral_penalty

        # Penalize for heading deviation

        heading_penalty = 15 * (abs(relative_heading) ** 2)
        reward -= heading_penalty

        distance_travelled = self.start_waypoint.location.distance(vehicle_location)
        reward += 0.5 * distance_travelled        

        return reward

    def generate_route(self,start_waypoint=None):

        if start_waypoint is None:
            self.start_waypoint = random.choice(self.spawn_points)
            # print(f"Start waypoint: {self.start_waypoint.location}")
        else:
            self.start_waypoint = start_waypoint
        self.dest_waypoint = random.choice(self.spawn_points)
        distance = self.start_waypoint.location.distance(self.dest_waypoint.location)
        while distance < 50:
            self.dest_waypoint = random.choice(self.spawn_points)
            distance = self.start_waypoint.location.distance(self.dest_waypoint.location)
        try:
            route = self.route_planner.trace_route(self.start_waypoint.location, self.dest_waypoint.location)
        except Exception as e:
            logger.error(f"Failed to generate route: {e}")
            return None
        return route 
    
    def calculate_relative_yaw(self, vehicle_transform, waypoint):
        # Extract yaw angles from vehicle and waypoint
        # Get vehicle yaw in world coordinates
        vehicle_yaw = vehicle_transform.rotation.yaw % 360
        vehicle_yaw = vehicle_yaw if vehicle_yaw <= 180 else vehicle_yaw - 360  # Normalize to [-180, 180]

        # Get road heading from current waypoint
        road_yaw = waypoint.transform.rotation.yaw % 360
        road_yaw = road_yaw if road_yaw <= 180 else road_yaw - 360  # Normalize to [-180, 180]

        # Calculate relative yaw to align with road orientation
        relative_yaw = (vehicle_yaw - road_yaw) % 360
        relative_yaw = relative_yaw if relative_yaw <= 180 else relative_yaw - 360  # Normalize again to [-180, 180]
        # Normalize relative yaw to [-1, 1]
        return relative_yaw / 180.0

    def find_current_waypoint(self, vehicle_location):
        if not self.route or len(self.route) == 0:
            logger.error('Route not found or is empty')
            return None
    # Find the waypoint with the minimum distance to the vehicle
        closest_waypoint = min(self.route,
                               key=lambda route_entry: route_entry[0].transform.location.distance(vehicle_location),)
        return closest_waypoint[0]  # Return only the waypoint, not the roadinfo
    
    def find_next_waypoint(self, current_waypoint, distance_lookahead=1):
        for i, (waypoint, _) in enumerate(self.route):
            dist = waypoint.transform.location.distance(current_waypoint.transform.location)
            if dist == 0:    
                if i + distance_lookahead < len(self.route):
                    return self.route[i + distance_lookahead][0]
                else:
                    return None
        return None and RuntimeError("Waypoints initialization failed.")

    def close(self):
        self.cleanup()
        self.world.tick()
        logger.info("Environment closed")

    def handle_collision(self):
        # Clear collision history to avoid repeated penalties for the same collision
        self.collision_hist.clear()
        logger.info("Resetting vehicle to starting position.")
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        self.vehicle.set_transform(self.start_waypoint)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        self.world.tick()
        time.sleep(0.5)  # Ensure reset takes effect