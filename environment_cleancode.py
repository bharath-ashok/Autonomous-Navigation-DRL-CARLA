import glob
import logging
import random
import sys
import time

import carla
import gymnasium as gym
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner
from gymnasium import spaces

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SECONDS_PER_EPISODE = 45
FIXED_DELTA_SECONDS = 0.02
NO_RENDERING = True
SYNCHRONOUS_MODE = True
SPIN = 10
HEIGHT = 480
WIDTH = 640


def append_carla_egg_path():
    try:
        sys.path.append(glob.glob('./carla/dist/carla-0.9.14-py3.7-win-amd64.egg')[0])
    except IndexError:
        logger.error("Couldn't import Carla egg properly")


def insert_carla_repo_path():
    try:
        sys.path.insert(0, r'C:\Users\ashokkumar\source\repos\AD\carla')
    except IndexError:
        pass


append_carla_egg_path()
insert_carla_repo_path()


class CarEnv(gym.Env):
    """Custom Environment for autonomous car control using CARLA."""
    
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4

    def __init__(self):
        super(CarEnv, self).__init__()

        # Initialize environment variables
        self.actor_list = []
        self.vehicle = None
        self.collision_hist = []
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        self.client = self._init_carla_client()
        self.world = self.client.get_world()
        self._setup_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find('vehicle.tesla.model3')
        self._initialize_route_planner()

        self.spectator = self.world.get_spectator() if not NO_RENDERING else None

    def _init_carla_client(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(15.0)
        client.load_world('Town02')
        return client

    def _setup_world(self):
        settings = self.world.get_settings()
        settings.no_rendering_mode = NO_RENDERING
        settings.synchronous_mode = SYNCHRONOUS_MODE
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        self.world.tick()

    def _initialize_route_planner(self):
        self.map = self.world.get_map()
        self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution=1.0)
        self.spawn_points = self.map.get_spawn_points()
        self.route = None
        self.start_waypoint = None
        self.dest_waypoint = None
        self.curr_waypoint = None

    def cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")
        self.world.tick()

    def get_observation(self, vehicle_transform, curr_waypoint, velocity):
        speed_normalized = self.get_speed(velocity) / 60.0
        lateral_distance = self.get_lateral_distance(vehicle_transform, curr_waypoint)
        heading = self.get_relative_heading(vehicle_transform, curr_waypoint)
        normalized_yaw = self.calculate_relative_yaw(vehicle_transform, curr_waypoint)

        observation = np.array([
            lateral_distance,
            speed_normalized,
            heading,
            normalized_yaw
        ], dtype=np.float32)

        if self.step_counter % 250 == 0:
            logger.info(f"Observation: {[f'{x:.2f}' for x in observation]}")
        return observation

    def step(self, action):
        self.step_counter += 1
        self.world.tick()
        self.simulation_time += FIXED_DELTA_SECONDS

        throttle, steer = map(float, action)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0))

        if self.step_counter % 250 == 0:
            logger.info(f"Action taken - Throttle: {throttle:.2f}, Steer: {steer:.2f}")

        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        reward = self._calculate_step_reward(vehicle_transform, velocity)

        done = self.simulation_time > SECONDS_PER_EPISODE
        if done:
            self.cleanup()
            logger.info('Episode time limit reached')

        self.observation = self.get_observation(vehicle_transform, self.curr_waypoint, velocity) if not done else None

        return self.observation, reward, done, False, {}

    def _calculate_step_reward(self, vehicle_transform, velocity):
        speed = self.get_speed(velocity)
        location = vehicle_transform.location
        distance_to_waypoint = location.distance(self.curr_waypoint.transform.location)
        reward = self.calculate_reward(speed, *self._get_dynamics(vehicle_transform))
        normalized_distance = min(distance_to_waypoint / 2, 1.0)
        reward += 0.1 * (1.0 - normalized_distance)

        self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)
        self._evaluate_collisions(reward)

        if self.curr_waypoint == self.dest_waypoint:
            self.route = self.generate_route(self.curr_waypoint)

        return reward

    def _get_dynamics(self, vehicle_transform):
        lateral_distance = self.get_lateral_distance(vehicle_transform, self.curr_waypoint)
        relative_yaw = self.calculate_relative_yaw(vehicle_transform, self.curr_waypoint)
        return lateral_distance, relative_yaw

    def _evaluate_collisions(self, reward):
        if self.collision_hist:
            reward -= 25.0
            self.handle_collision(self.collision_hist[0], self.vehicle)
            self.collision_hist.clear()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cleanup_environment()
        self.initialize_environment()
        return self.observation, {}

    def cleanup_environment(self):
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []
        self.previous_waypoints = []
        self.route = None
        self.start_waypoint = None
        self.curr_waypoint = None
        self.dest_waypoint = None
        self.simulation_time = 0

    def initialize_environment(self):
        while self.route is None:
            self.route = self.generate_route()

        self.vehicle = self.spawn_vehicle()
        self.actor_list.append(self.vehicle)
        self.world.tick()
        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        self.curr_waypoint = self.find_current_waypoint(vehicle_transform.location)
        self.setup_sensors()

        self.step_counter = 0
        self.observation = self.get_observation(vehicle_transform, self.curr_waypoint, velocity)

    def spawn_vehicle(self):
        transform = self.start_waypoint
        for _ in range(10):  # Retry spawning up to 10 times
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
                logger.info(f'Spawned vehicle at: {transform.location}')
                return vehicle
            except Exception as e:
                logger.error(f"Failed to spawn vehicle: {e}")
                time.sleep(0.1)

        raise RuntimeError("Vehicle could not be spawned after multiple attempts.")

    def setup_sensors(self):
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        if self.spectator:
            self.update_spectator_view()

    def update_spectator_view(self):
        spectator_transform = self.vehicle.get_transform()
        spectator_transform.rotation.pitch -= 60.0
        spectator_transform.location += carla.Location(z=25.0)
        self.spectator.set_transform(spectator_transform)

    def calculate_reward(self, speed, lateral_distance, relative_yaw, vehicle_location):
        reward = self._speed_reward(speed)
        reward -= self._lateral_penalty(lateral_distance)
        reward -= self._heading_penalty(relative_yaw)
        return reward

    def _speed_reward(self, speed):
        speed_normalized = speed / 60.0
        if 0.25 <= speed_normalized <= 0.5:
            return 0.0
        k = 20
        if speed_normalized < 0.25:
            return -5 / (1 + np.exp(-k * (0.25 - speed_normalized)))
        elif speed_normalized > 0.5:
            return -5 / (1 + np.exp(-k * (speed_normalized - 0.5)))
        return 0.0

    def _lateral_penalty(self, lateral_distance):
        return 5 * (lateral_distance ** 2)

    def _heading_penalty(self, relative_heading):
        absolute_heading = np.abs(relative_heading)
        return 10 / (1 + np.exp(-10 * absolute_heading)) - 5

    def collision_data(self, event):
        if len(self.collision_hist) >= 10:
            self.collision_hist.pop(0)
        self.collision_hist.append(event)

    def calculate_relative_yaw(self, vehicle_transform, waypoint):
        vehicle_yaw = vehicle_transform.rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw
        relative_yaw = self._normalize_angle(waypoint_yaw - vehicle_yaw)
        return relative_yaw / 180.0

    def _normalize_angle(self, angle):
        return ((angle + 180) % 360) - 180

    def get_speed(self, velocity, max_speed=60.0):
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed if self.vehicle else 0.0

    def get_lateral_distance(self, vehicle_transform, waypoint, max_distance=4.0):
        vehicle_location = self._vector_from_location(vehicle_transform.location)
        waypoint_location = self._vector_from_location(waypoint.transform.location)
        waypoint_direction = self._normalize_vector(self._vector_from_location(waypoint.transform.get_forward_vector()))

        vehicle_projection = np.dot(vehicle_location - waypoint_location, waypoint_direction) * waypoint_direction
        lateral_vector = vehicle_location - waypoint_location - vehicle_projection
        lateral_distance = np.linalg.norm(lateral_vector)
        return self._calculate_normalized_lateral_distance(lateral_distance, waypoint_direction, vehicle_location, vehicle_projection)

    def _vector_from_location(self, location):
        return np.array([location.x, location.y])

    def _normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def _calculate_normalized_lateral_distance(self, lateral_distance, waypoint_direction, vehicle_vector, vehicle_projection):
        normalized_distance = np.clip(lateral_distance / 4.0, -1, 1)
        cross_product = np.cross(waypoint_direction, vehicle_vector - vehicle_projection)
        return -normalized_distance if cross_product < 0 else normalized_distance

    def get_relative_heading(self, vehicle_transform, waypoint):
        turn_direction = self.turn_direction(vehicle_transform, waypoint)
        relative_heading = self._calculate_relative_heading(turn_direction, waypoint, vehicle_transform)
        return np.clip(relative_heading / 180.0, -1, 1)

    def _calculate_relative_heading(self, turn_direction, waypoint, vehicle_transform):
        heading_difference = self._sum_yaw_differences(waypoint) 
        relative_heading = heading_difference if turn_direction == 0 else turn_direction * heading_difference
        return (relative_heading % 360) - 180 if (relative_heading % 360) > 180 else relative_heading

    def _sum_yaw_differences(self, waypoint, num_lookahead=5):
        yaw_differences_sum = 0
        current_waypoint = waypoint
        
        for _ in range(num_lookahead):
            next_waypoint = self.find_next_waypoint(current_waypoint)
            if not next_waypoint:
                break

            current_yaw = current_waypoint.transform.rotation.yaw
            next_yaw = next_waypoint.transform.rotation.yaw
            yaw_difference = self._get_yaw_difference(current_yaw, next_yaw)
            yaw_differences_sum += yaw_difference
            current_waypoint = next_waypoint

        return yaw_differences_sum

    def _get_yaw_difference(self, yaw1, yaw2):
        diff = abs(yaw1 - yaw2)
        return diff if diff <= 180 else 360 - diff

    def turn_direction(self, vehicle_transform, waypoint, num_lookahead=5, distance_lookahead=2):
        current_waypoint = waypoint

        for _ in range(num_lookahead):
            future_waypoint = self.find_next_waypoint(current_waypoint, distance_lookahead)
            if not future_waypoint:
                break
            current_waypoint = future_waypoint

        if future_waypoint:
            vehicle_yaw = vehicle_transform.rotation.yaw % 360
            future_yaw = future_waypoint.transform.rotation.yaw % 360
            yaw_diff = (future_yaw - vehicle_yaw + 180) % 360 - 180
            return 1 if yaw_diff >= 0 else -1
        return 0

    def generate_route(self, start_waypoint=None):
        self.start_waypoint = random.choice(self.spawn_points) if start_waypoint is None else start_waypoint
        self.dest_waypoint = random.choice(self.spawn_points)
        
        while self._is_destination_too_close(self.start_waypoint, self.dest_waypoint):
            self.dest_waypoint = random.choice(self.spawn_points)
        
        return self._trace_route()

    def _is_destination_too_close(self, start_waypoint, dest_waypoint):
        return start_waypoint.location.distance(dest_waypoint.location) < 50

    def _trace_route(self):
        try:
            return self.route_planner.trace_route(self.start_waypoint.location, self.dest_waypoint.location)
        except Exception as e:
            logger.error(f"Failed to generate route: {e}")
            return None

    def find_current_waypoint(self, vehicle_location):
        if not self.route or len(self.route) == 0:
            logger.error('Route not found or is empty')
            return None

        return min(self.route, key=lambda entry: entry[0].transform.location.distance(vehicle_location))[0]

    def find_next_waypoint(self, current_waypoint, distance_lookahead=1):
        current_index = self.route.index((current_waypoint,))

        if current_index + distance_lookahead < len(self.route):
            return self.route[current_index + distance_lookahead][0]
        return None

    def close(self):
        self.cleanup()
        self.world.tick()
        logger.info("Environment closed")

    def handle_collision(self, event, vehicle):
        collision_actor = event.other_actor
        collision_object_type = collision_actor.type_id
        severity = np.linalg.norm([event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z])

        if severity < 500:
            self._handle_impact(vehicle, "Low", collision_object_type)
        elif severity < 1500:
            self._handle_impact(vehicle, "Moderate", collision_object_type)
        else:
            self._handle_impact(vehicle, "High", collision_object_type, emergency_brake=True)

    def _handle_impact(self, vehicle, severity_level, collision_object_type, emergency_brake=False):
        logger.info(f"{severity_level} severity collision with {collision_object_type}.")
        new_position = self.get_nearest_safe_position()
        vehicle.set_transform(new_position)
        if emergency_brake:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            self.world.tick()

    def get_nearest_safe_position(self):
        return self.curr_waypoint.transform