import glob
import sys
import random
import time
import math
import numpy as np
import logging
import gymnasium as gym
from gymnasium import spaces

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
NO_RENDERING = False 
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
        obs_dim = 6
        low = np.array([-1, 0, -1, -1, 0, 0], dtype=np.float32)  # Set speed to [0, 1] and other values to [-1, 1]
        high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([ -1 , -1]), high=np.array([ 1, 1]), dtype=np.float32)

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
        self.traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        self.traffic_light_info = []
        self.route = None  
        self.start_waypoint = None
        self.dest_waypoint = None
        self.curr_waypoint = None
        self.previous_waypoints = []
        self.observation = None
        self.debug = False
        self.integral_error = 0.0
        self.previous_error = 0.0

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")
        self.world.tick()

    def get_observation(self, vehicle_transform,  curr_waypoint, velocity):

        speed = self.get_speed(velocity)
        speed = speed / 60.0  # Normalize speed to [0, 1]
        lateral_distance = self.get_lateral_distance(vehicle_transform, curr_waypoint)
        heading = self.get_relative_heading(vehicle_transform, curr_waypoint)
        normalized_yaw = self.calculate_relative_yaw(vehicle_transform, curr_waypoint)
        traffic_light_state, traffic_light_distance = self.get_traffic_light_info(vehicle_transform, curr_waypoint)
        traffic_light_state_normalized = {
            'Red': 1,
            'Yellow': 1,
            'Green': 0
        }.get(traffic_light_state, 0)
        traffic_light_distance = np.clip(traffic_light_distance/50.0, 0, 1)  # Normalize traffic light distance
        obs = np.array([
            lateral_distance,
            speed,
            heading,
            normalized_yaw,
            traffic_light_state_normalized,
            traffic_light_distance
        ], dtype=np.float32)
        if self.step_counter % 250 == 0:
            logger.info(f"Observation: {[f'{x:.2f}' for x in obs], traffic_light_state}")
        return obs

    def step(self, action):
        self.step_counter += 1
        self.world.tick()
        self.simulation_time += FIXED_DELTA_SECONDS
        throttle_action , brake_action, steer_action = action
        throttle = float(throttle_action)
        brake = float(brake_action)
        steer = float(steer_action)
        velocity = self.vehicle.get_velocity()
        speed = self.get_speed(velocity)
        
        current_index = next((i for i, (waypoint, _) in enumerate(self.route) if waypoint == self.curr_waypoint), -1)

        ## TODO in BatchSync
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        if self.step_counter % 250 == 0:
            logger.info(f"Action:{action[0]:.2f}, {action[1]:.2f},speed: {speed:.2f},waypoint: {current_index}/{len(self.route)} ")
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.curr_waypoint)
        relative_yaw = self.calculate_relative_yaw(vehicle_transform, self.curr_waypoint)
        traffic_light_state, traffic_light_distance = self.get_traffic_light_info(vehicle_transform, self.curr_waypoint)
        if self.step_counter % 250 == 0 and self.debug:
            print(f"Vehicle transform: {vehicle_transform}, Yaw:{vehicle_transform.rotation.yaw}, WP_Yaw:{self.curr_waypoint.transform.rotation.yaw} relative_yaw: {relative_yaw}")
            print(f"destination waypoint: {self.curr_waypoint.transform.location}")

        reward = self.calculate_reward( speed, lateral_distance, relative_yaw, vehicle_location, traffic_light_state, traffic_light_distance)
        done = False
 
        distance_to_waypoint = vehicle_location.distance(self.curr_waypoint.transform.location)
        # Normalize the distance to be between 0 and 1
        normalized_distance = min(distance_to_waypoint / 2, 1.0)

        reward += 0.1*(1.0 - normalized_distance)

        if self.find_current_waypoint(vehicle_transform.location) is not None:
            self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)

        if self.curr_waypoint.transform.location.distance(self.dest_waypoint.location) < 5.0:
            try:
                self.route = self.generate_route(self.curr_waypoint)
                for tl in self.traffic_lights:
                    tl_waypoint = tl.get_stop_waypoints()
                    if tl_waypoint:
                        tl_info = (tl, tl_waypoint[0])
                        self.traffic_light_info.append(tl_info)
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
        self.traffic_light_info = []
        if self.route is None:
            self.route = self.generate_route()
            logger.info(f'route length: {len(self.route)}, distance: {self.sampling_resolution*len(self.route)}')
        self.traffic_light_info = self.update_traffic_light_info(self.traffic_lights, self.route)
        self.vehicle = self.spawn_vehicle()
        self.actor_list.append(self.vehicle)
        self.world.tick()
        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)
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

    def calculate_reward( self, speed, lateral_distance, relative_yaw, vehicle_location, traffic_light_state=None, traffic_light_distance=None, 
                             speed_weight=1.0, lateral_weight=1.0, yaw_weight=1.0, traffic_weight=1.0):
        
        reward = 0.0
        speed = speed / 60.0  # Normalize speed to [0, 1]
        # Reward for the desired speed range
        k = 20
        if 0.25 <= speed <= 0.5:
            reward += speed_weight * 0.0
        # Penalty for underspeed
        elif speed < 0.25:
            reward += speed_weight * (-5 / (1 + np.exp(-k * (0.25 - speed))))
        # Penalty for overspeed
        elif speed > 0.5:
            reward += speed_weight * (-5 / (1 + np.exp(-k * (speed - 0.5))))
        
        # Penalize for lateral distance from the center line
        lateral_penalty = lateral_weight * 5 * (lateral_distance ** 2)
        reward -= lateral_penalty

        absolute_yaw = np.abs(relative_yaw)
        yaw_penalty = yaw_weight * (10 / (1 + np.exp(-10 * absolute_yaw)) - 5)
        reward -= yaw_penalty

        traffic_light_state = {'RED': 1,'YELLOW': 1,'GREEN': 0}.get(traffic_light_state, 0)
        traffic_light_distance = np.clip(traffic_light_distance/50.0, 0, 1)  # Normalize traffic light distance
        if traffic_light_state == 1:
            penalty_factor = 1 - traffic_light_distance  # Closer distance, higher penalty
            reward -= traffic_weight * penalty_factor * speed ** 2  # Quadratic speed penalty
                
                # Gradual reward for stopping
            if speed < 0.1:
                reward += traffic_weight * (1 - penalty_factor) * 5  # Higher reward closer to the light

        elif traffic_light_state == 0:  # GREEN light
                # Gradual reward for smooth passing
                reward += traffic_weight * (1 - traffic_light_distance) * speed  # Reward scales with distance and speed
        
        return reward

    def generate_route(self,start_waypoint=None, max_retries=5):
        retries = 0
        if start_waypoint is None:
            self.start_waypoint = random.choice(self.spawn_points)
        else:
            self.start_waypoint = start_waypoint.transform
        while retries < max_retries:
            potential_dest = random.choice(self.spawn_points)
            distance = self.start_waypoint.location.distance(potential_dest.location)
            if distance >= 50:
                self.dest_waypoint = potential_dest
                try:
                    route = self.route_planner.trace_route(self.start_waypoint.location, self.dest_waypoint.location)
                    return route
                except Exception as e:
                    logger.error(f"Attempt {retries + 1}: Failed to generate route: {e}")
            retries += 1
        logger.error(f"Failed to generate route after {max_retries} retries")
        return None 
    
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
        collision_actor = event.other_actor
        collision_impulse = event.normal_impulse

        severity = self._calculate_collision_severity(collision_impulse)
        collision_type = collision_actor.type_id

        self._apply_response_based_on_severity(severity, collision_type, vehicle)
        self._log_collision_details(severity, collision_type)

    def _apply_response_based_on_severity(self, severity, collision_type, vehicle):
        """Apply vehicle response based on collision severity."""
        low_severity_threshold = 500
        high_severity_threshold = 1500

        if severity < low_severity_threshold:
            vehicle.set_transform(self.get_nearest_safe_position())
        elif severity < high_severity_threshold:
            vehicle.set_transform(self.get_nearest_safe_position())
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        else:
            vehicle.set_transform(self.get_nearest_safe_position())
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        self.world.tick()

    def _log_collision_details(self, severity, collision_type):
        """Log the collision details and apply specific penalties."""
        low_severity_threshold = 500
        high_severity_threshold = 1500

        if severity < low_severity_threshold:
            logger.info(f'Low severity collision with {collision_type}. Minor penalties applied.')
        elif severity < high_severity_threshold:
            logger.info(f'Moderate severity collision with {collision_type}. Moderate penalties applied.')
        else:
            logger.warning(f'High severity collision with {collision_type}. Heavy penalties applied.')

        if 'vehicle' in collision_type:
            logger.info('Collision with another vehicle.')
        elif 'pedestrian' in collision_type:
            logger.warning('Collision with a pedestrian! Severe penalty applied.')

    def _calculate_collision_severity(self, impulse):
        """Calculate the severity of a collision based on the impulse magnitude."""
        return np.linalg.norm([impulse.x, impulse.y, impulse.z])

    def get_nearest_safe_position(self):
        if self.curr_waypoint:
            return self.curr_waypoint.transform
        logger.error("Current waypoint not defined; returning to start position.")
        return self.start_waypoint.transform if self.start_waypoint else None
        
    def get_traffic_light_info(self, vehicle_transform, current_waypoint):
        vehicle_location = vehicle_transform.location
        # Find the segment of the route that is ahead of the vehicle
        route_tl = self._get_route_segment(current_waypoint, length=51)
        # Check for any traffic lights that impact this segment
        for tl, tl_waypoint in self.traffic_light_info:
            if any(self._is_waypoint_close(waypoint, tl_waypoint) for waypoint, _ in route_tl):
                stop_distance = tl_waypoint.transform.location.distance(vehicle_location)
                if  5 < stop_distance  < 50 :
                    tl.set_state(carla.TrafficLightState.Red)
                if stop_distance == 0:
                    tl.set_state(carla.TrafficLightState.Green)
                if tl.get_state() == carla.TrafficLightState.Red:
                    return 'Red', stop_distance
                elif tl.get_state() == carla.TrafficLightState.Yellow:
                    return 'Yellow', stop_distance
                elif tl.get_state() == carla.TrafficLightState.Green:
                    return 'Green', stop_distance
        return 0, 0  # Default if no traffic light affects the route

    def _get_route_segment(self, current_waypoint, length=51):
        """Returns a sublist of self.route starting from the current_waypoint."""
        if not self.route:
            logger.error("Route is empty or not set.")
            return []
        try:
            start_index = next(
                i for i, (waypoint, _) in enumerate(self.route)
                if waypoint == current_waypoint
            )
            end_index = min(start_index + length, len(self.route))
            return self.route[start_index:end_index]
        except StopIteration:
            logger.error("Current waypoint not found in the route.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error occurred while getting route segment: {e}")
            return []
        
    def _is_waypoint_close(self, waypoint, target_waypoint, threshold=2.0):
        """Checks if a waypoint is within a certain distance (threshold) of the target waypoint."""
        distance = waypoint.transform.location.distance(target_waypoint.transform.location)
        return distance < threshold

    def update_traffic_light_info(self,traffic_lights, route):
        """
        Updates the information of traffic lights that are within 2 meters
        of any waypoint in the current route.
        """
        traffic_light_info = []
        for tl in traffic_lights:
            tl_waypoints = tl.get_stop_waypoints()

            if tl_waypoints:
                tl_location = tl_waypoints[0].transform.location
                # Check if the traffic light waypoint is within 2 meters of any waypoint in the route
                for waypoint, _ in route:
                    if waypoint.transform.location.distance(tl_location) < 2.0:
                        tl_info = (tl, tl_waypoints[0])
                        if tl_info not in traffic_light_info:
                            traffic_light_info.append(tl_info)
        
        return traffic_light_info
    
    def traffic_light_reward(self, traffic_light_state, distance_to_light, speed, time_to_light_change, traffic_weight):
        """
        Reward function for teaching an RL model to navigate traffic lights.
        
        Parameters:
        - traffic_light_state (str): The current state of the light ('RED', 'YELLOW', 'GREEN').
        - distance_to_light (float): Distance to the traffic light (meters).
        - speed (float): Current speed of the vehicle (m/s).
        - time_to_light_change (float): Time (seconds) until the light changes state (optional).
        - traffic_weight (float): Weight parameter to scale traffic light-related rewards/penalties.
        
        Returns:
        - reward (float): Calculated reward for the current state-action pair.
        """
        reward = 0
        max_speed = 60.0  
        stopping_distance = 5.0 

        distance_normalized = np.clip(distance_to_light / 50.0, 0, 1)
        speed_normalized = np.clip(speed / max_speed, 0, 1)
        
        if traffic_light_state in ['RED', 'YELLOW']:
            if distance_to_light < 2.0 and speed > 0.1:
                reward -= traffic_weight * 20  

            if distance_to_light <= stopping_distance:
                penalty_factor = 1 - (distance_to_light / stopping_distance)
                reward -= traffic_weight * penalty_factor * (speed_normalized ** 2) * 10

            # Bonus for slowing down or stopping in time
            if distance_to_light <= stopping_distance and speed < 0.1:
                reward += traffic_weight * (1 - distance_normalized) * 10  # Higher reward closer to light

        elif traffic_light_state == 'GREEN':
            # Bonus for maintaining appropriate speed near green lights
            if stopping_distance < distance_to_light <= 50.0:
                reward += traffic_weight * (1 - distance_normalized) * speed_normalized * 5


        # Bonus for planning ahead (e.g., starting to slow down for a red light)
        if traffic_light_state in ['RED', 'YELLOW'] and time_to_light_change < 5.0:
            if distance_to_light > stopping_distance and speed > 0.1:
                reward += traffic_weight * 2

        # General penalties/rewards
        if speed > max_speed:
            reward -= traffic_weight * 10  # Penalize speeding
        if speed == 0 and traffic_light_state == 'GREEN':
            reward -= traffic_weight * 10  # Penalize stopping unnecessarily at a green light

        return reward
