import glob
import sys
import random
import time
import numpy as np
import logging
import gymnasium as gym
from gymnasium import spaces
import cv2

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

SECONDS_PER_EPISODE = 45
FIXED_DELTA_SECONDS = 0.02
NO_RENDERING = True 
SYNCHRONOUS_MODE = True
SHOW_CAM = False
N_CHANNELS = 3
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
        # Example for using image as input normalised to 0..1 (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                    shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([ 0, -1]), high=np.array([ 1, 1]), dtype=np.float32)

        self.client = carla.Client('localhost', 2004)
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
        self.Show_cam = SHOW_CAM
        
        ## Route planner
        self.map = self.world.get_map()
        self.sampling_resolution = 1.0
        self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution=self.sampling_resolution)
        self.spawn_points = self.map.get_spawn_points()      
        self.route = None  
        self.start_waypoint = None
        self.dest_waypoint = None
        self.curr_waypoint = None
        self.debug = False

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")
        if self.Show_cam:
            cv2.destroyAllWindows()
        self.world.tick()

    def step(self, action):
        self.step_counter += 1
        self.world.tick()
        self.simulation_time += FIXED_DELTA_SECONDS
        throttle_action , steer_action = action
        throttle_action = float(throttle_action)
        steer = float(steer_action)
        brake = 0
        velocity = self.vehicle.get_velocity()
        speed = self.get_speed(velocity)
        
        current_index = next((i for i, (waypoint, _) in enumerate(self.route) if waypoint == self.curr_waypoint), -1)

        ## TODO in BatchSync
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_action, steer=steer, brake=brake))
        if self.step_counter % 250 == 0:
            logger.info(f"Action, Speed:{action[0]:.2f}, {action[1]:.2f}, {speed:.2f},waypoint: {current_index}/{len(self.route)} ")
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.curr_waypoint)
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.curr_waypoint)
        relative_yaw = self.calculate_relative_yaw(vehicle_transform, self.curr_waypoint)

        # storing camera to return at the end in case the clean-up function destroys it
        cam = self.front_camera
		# showing image
        if self.Show_cam:
            cv2.imshow('Sem Camera', cam)
            cv2.waitKey(1)

        if self.step_counter % 250 == 0 and self.debug:
            print(f"Vehicle transform: {vehicle_transform}, Yaw:{vehicle_transform.rotation.yaw}, WP_Yaw:{self.curr_waypoint.transform.rotation.yaw} relative_yaw: {relative_yaw}")
            print(f"destination waypoint: {self.curr_waypoint.transform.location}")
        reward = self.calculate_reward( speed, lateral_distance, relative_yaw, vehicle_location)
        done = False
        # # --- Reward based on proximity to the waypoint ---
                # Calculate the Euclidean distance to the waypoint
        distance_to_waypoint = vehicle_location.distance(self.curr_waypoint.transform.location)
        # Normalize the distance to be between 0 and 1
        normalized_distance = min(distance_to_waypoint / 2, 1.0)
        # Return a reward based on how close the vehicle is to the path
        # Use a weight to control the influence of this reward
        reward += 0.1*(1.0 - normalized_distance)
        if self.find_current_waypoint(vehicle_transform.location) is not None:
            self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)

        if self.curr_waypoint is self.dest_waypoint:
            try:
                self.route = self.generate_route(start_waypoint=self.curr_waypoint)
            except Exception as e:
                logger.error(f"Failed to generate route: {e}")
                raise
        # --- Penalize for collisions ---
        if len(self.collision_hist) != 0:
            reward -= 25.0
            self.handle_collision(self.collision_hist[0], self.vehicle)
            self.collision_hist.clear()

        # --- Check if episode time limit reached ---
        if self.simulation_time > SECONDS_PER_EPISODE:
            done = True
            logger.info(f'Episode time limit reached')
            self.cleanup()
        return cam/255.0, reward, done, False, {}
        
    def collision_data(self, event):
        if len(self.collision_hist) >= 10:  # Keep only the last 10 collisions
            self.collision_hist.pop(0)
        self.collision_hist.append(event)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []
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
        try:
            self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
            self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
            self.sem_cam.set_attribute("fov", f"90")
            camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
            self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
            self.actor_list.append(self.sensor)
            self.sensor.listen(lambda data: self.process_img(data))
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
        self.world.tick()
        vehicle_transform = self.vehicle.get_transform()
        self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)
        while self.front_camera is None:
            time.sleep(0.01)
        if self.Show_cam:
            cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Sem Camera', self.front_camera)
            cv2.waitKey(1)
        # Initialize episode timing and tracking
        self.step_counter = 0

        return self.front_camera/255.0, {}
    
    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

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
                time.sleep(0.1)  # Brief delay before retrying
        if vehicle is None:
            raise RuntimeError("Vehicle could not be spawned after multiple attempts.")
        logger.info(f'Spawned vehicle at: {transform.location}')
        return vehicle
    
    def get_speed(self, v, max_speed=60.0):
        if self.vehicle:
            speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            return speed
        return 0.0

    def get_lateral_distance(self, vehicle_transform, waypoint, max_distance=4.0):
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

    def get_relative_heading(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=2):

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

        if turn_direction == 0:
            normalized_heading = relative_heading / 180.0
        else:
            normalized_heading = turn_direction * (relative_heading / 180.0)

        return np.clip(normalized_heading, -1, 1)

    def turn_direction(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=2):
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
        return 0

    def calculate_reward( self, speed, lateral_distance, relative_heading, vehicle_location, speed_weight=1.0, lateral_weight=1.0, yaw_weight=1.0, distance_weight=0.1): #TODO was 1.0, 1.0, 1.2, 0.1
        reward = 0.0
        speed = speed / 60.0  # Normalize speed to [0, 1]
        # Reward for the desired speed range
        k = 20
        if 0.25 <= speed <= 0.50: 
            reward += speed_weight*0.0
        # Penalty for underspeed
        elif speed < 0.25:
            reward += speed_weight*(-5 / (1 + np.exp(-k * (0.25 - speed))))
        # Penalty for overspeed
        elif speed > 0.50:
            reward += speed_weight*( -5 / (1 + np.exp(-k * (speed - 0.50))))
        
        # Penalize for lateral distance from the center line
        lateral_penalty = lateral_weight * 5 * (lateral_distance ** 2)
        reward -= lateral_penalty

        absolute_heading = np.abs(relative_heading)
        heading_penalty = yaw_weight * (10 / (1 + np.exp(-10 * absolute_heading)) - 5)
        reward -= heading_penalty

        # Distance = vehicle_location.distance(self.start_waypoint.location)
        # reward += distance_weight*(Distance/10.0)

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
        vehicle_yaw = vehicle_transform.rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw
        # Calculate relative yaw
        relative_yaw = waypoint_yaw - vehicle_yaw
        # Normalize relative yaw to be within [-180, 180]
        def normalize_angle(angle):
            # Ensure the angle is within [-180, 180]
            return ((angle + 180) % 360) - 180
        relative_yaw = normalize_angle(relative_yaw)
        # Scale to [-1, 1]
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
        return None

    def close(self):
        self.cleanup()
        self.world.tick()
        logger.info("Environment closed")

    def handle_collision(self, event, vehicle):
        """
        Handle different collision scenarios and apply appropriate penalties or reset strategies.
        
        :param event: The collision event containing details about the collision.
        :param actor: The actor involved in the collision, usually the environment vehicle.
        :param vehicle: The ego vehicle in the simulation.
        """

        # Extract basic collision data
        collision_actor = event.other_actor
        collision_impulse = event.normal_impulse

        # Define thresholds or categories for collision severity
        low_severity_threshold = 500  # Example impulse threshold for low severity
        high_severity_threshold = 1500  # Example threshold for high severity

        # Determine severity based on collision impulse
        severity = np.linalg.norm([collision_impulse.x, collision_impulse.y, collision_impulse.z])

        # Check the type of collision actor involved
        collision_object_type = collision_actor.type_id

        # Apply different strategies based on collision severity and object type
        if severity < low_severity_threshold:
            logger.info(f'Low severity collision with {collision_object_type}. Minor penalties applied.')
            #penalty = -10  # Minor penalty for low-severity collision
            vehicle.set_transform(self.get_nearest_safe_position())
            # Optionally, slightly adjust vehicle position if needed
        elif severity < high_severity_threshold:
            logger.info(f'Moderate severity collision with {collision_object_type}. Moderate penalties applied.')
            #penalty = -50  # Moderate penalty for medium-severity collision
            vehicle.set_transform(self.get_nearest_safe_position())
            # Consider stabilizing the vehicle or applying a small reset
        else:
            logger.warning(f'High severity collision with {collision_object_type}. Heavy penalties applied.')
            #penalty = -100  # Major penalty for high-severity collision
            # Reset or significantly modify the vehicle's state
            vehicle.set_transform(self.get_nearest_safe_position())
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            self.world.tick()

        # Handle specific collision objects if needed
        if 'vehicle' in collision_object_type:
            logger.info('Collision with another vehicle.') 
            # Implement further handling if needed for vehicle-to-vehicle collisions
        elif 'pedestrian' in collision_object_type:
            logger.warning('Collision with a pedestrian! Severe penalty applied.')
            #penalty = -200  # Severe penalty for pedestrian collision

        # Return the computed penalty value to be used in the reward calculation
        return None

    def get_nearest_safe_position(self):
        """
        Find and return the nearest safe position to reset the vehicle to.
        
        :param vehicle: The ego vehicle in the simulation.
        :return: A carla.Transform object representing a safe position for the vehicle.
        """
        # Placeholder implementation: return to the start waypoint or predefined safe spot
        # Modify this to use actual map and state data to find a real nearest safe point
        start_waypoint = self.curr_waypoint
        return start_waypoint.transform