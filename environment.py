import glob
import sys
import random
import time
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import KDTree

try:
    sys.path.append(glob.glob('./carla/dist/carla-0.9.14-py3.7-win-amd64.egg')[0])
except IndexError:
    print("Couldn't import Carla egg properly")
import carla

SECONDS_PER_EPISODE = 8
FIXED_DELTA_SECONDS = 0.02
SHOW_PREVIEW = True
NO_RENDERING = False
SYNCHRONOUS_MODE = False
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

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
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

        # Generate waypoints
        town_map = self.world.get_map()
        all_waypoints = town_map.generate_waypoints(5) #note distance between waypoints in meters
        waypoints = []
        for wp in all_waypoints:
            if len(waypoints) == 0:
                waypoints.append(wp) #first waypoint is added regardless to start the list
            else:
                found = False
                for uwp in waypoints: #check for same located waypoints and ignore if found
                    if abs(uwp.transform.location.x - wp.transform.location.x) < 0.1 \
                                    and abs(uwp.transform.location.y - wp.transform.location.y)<0.1 \
                                    and abs(uwp.transform.rotation.yaw - wp.transform.rotation.yaw)<20:  #this checks same direction
                        found = True
                        break
                if not found:
                    waypoints.append(wp)

        #set Waypoints
        self.waypoints = waypoints
        # Convert waypoints to a list of positions for KDTree
        waypoint_positions = [(wp.transform.location.x, wp.transform.location.y) for wp in self.waypoints]
        self.waypoint_tree = KDTree(waypoint_positions)
        self.start_waypoint = None
        self.dest_waypoint = None
        self.observation = None

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for actor in self.actor_list:
            try:
                actor.destroy()
            except Exception as e:
                print(f"Failed to destroy actor {actor.id}: {e}")

    def next_waypoint(self, start_waypoint):
        next_wps = start_waypoint.next(2.0)
        return next_wps[0]

    def get_observation(self, vehicle_transform, dest_waypoint):
        veh_loc = vehicle_transform.location
        veh_rot = vehicle_transform.rotation
        wpt_loc = dest_waypoint.transform.location
        wpt_rot = dest_waypoint.transform.rotation
        distance_x = veh_loc.x - wpt_loc.x
        distance_y = veh_loc.y - wpt_loc.y
        direction_difference = (veh_rot.yaw - wpt_rot.yaw) % 180
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

        # map steering actions
        if steer ==0:
            steer = - 0.9
        elif steer ==1:
            steer = -0.25
        elif steer ==2:
            steer = -0.1
        elif steer ==3:
            steer = -0.05
        elif steer ==4:
            steer = 0.0 
        elif steer ==5:
            steer = 0.05
        elif steer ==6:
            steer = 0.1
        elif steer ==7:
            steer = 0.25
        elif steer ==8:
            steer = 0.9
        # map throttle and apply steer and throttle	
        if throttle == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=steer, brake = 1.0))
        elif throttle == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=steer, brake = 0.0))
        elif throttle == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=steer, brake = 0.0))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steer, brake = 0.0))

        reward = 0
        done = False
        vehicle_location = vehicle_transform.location

        distance_travelled  = self.dest_waypoint.transform.location.distance(vehicle_location)
        #TEST
        waypoint_distance  = vehicle_location.distance(self.dest_waypoint.transform.location)
        if self.step_counter % 50 == 0:
            print('steer input from model:', steer, ', throttle: ', throttle, 'distance travelled:', distance_travelled, ', waypoint distance: ', waypoint_distance)
        # TODO- test
        reward  += distance_travelled

        #punish for collision
        if len(self.collision_hist) != 0:
            done = True
            reward -= 100
            print('collision detected, cleaning up')
            self.cleanup()

        if waypoint_distance < 1.0:  # Threshold to consider the waypoint reached
            self.start_waypoint = self.dest_waypoint
            self.dest_waypoint= self.next_waypoint(self.start_waypoint)
            reward += 10
            print('waypoint reached, next waypoint set, reward: ', reward)
            self.world.debug.draw_arrow(self.start_waypoint.transform.location,self.dest_waypoint.transform.location, life_time=-10)
        kmh = self.get_speed(self.vehicle.get_velocity())
        if kmh < 10:
            reward -= 3
        elif kmh < 15:
            reward -= 1
        elif kmh > 40:
            reward -= 5
        else:
            reward += 1
        #TODO Later
        # # track steering lock duration to prevent "chasing its tail"

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            reward += 10
            print('reward: ', reward)
            print('Episode time limit reached')
            self.cleanup()

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

        self.vehicle = self.spawn_vehicle()
        self.actor_list.append(self.vehicle)
        #TODO - print current wp and load into a self.init_wp
        time.sleep(0.2)

        initial_pos = self.vehicle.get_transform().location
        print('Initial position: ', initial_pos)
        self.start_waypoint = initial_waypoint(self, initial_pos)
        print('Current waypoint: ', self.start_waypoint.transform.location)

        self.dest_waypoint = self.start_waypoint.next(2.0)[0]
        print('Target waypoint: ', self.dest_waypoint.transform.location)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # setup sensors and camera
        camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
        # showing camera at the spawn point
        if self.SHOW_CAM:
            trans = self.vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20), carla.Rotation(yaw=-180, pitch=-90)))

        # collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        self.episode_start = time.time()
        self.step_counter = 0
        self.observation = self.get_observation(self.vehicle.get_transform(), self.dest_waypoint)
        return self.observation, {}
    
    def get_speed(self, v):
        return int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
    
    def spawn_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        print('Number of spawn points: ', len(spawn_points))
        transform = random.choice(spawn_points)
        print('Spawning vehicle at: ', transform.location)
        vehicle = None
        while vehicle is None:
            try:
                vehicle = self.world.spawn_actor(self.model_3, transform)
            except:
                pass
        return vehicle
    
def initial_waypoint(self, initial_pos):
    # Query the KD-tree for the closest waypoint to initial_pos
    _, closest_index = self.waypoint_tree.query([initial_pos.x, initial_pos.y])
    closest_waypoint = self.waypoints[closest_index]
    return closest_waypoint
