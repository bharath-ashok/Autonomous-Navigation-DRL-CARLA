import glob
import sys
import os
import random
import time
import numpy as np
import math 
import gymnasium as gym
from gymnasium import spaces

try:
    sys.path.append(glob.glob('./carla/dist/carla-0.9.14-py3.7-win-amd64.egg')[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')
import carla

SECONDS_PER_EPISODE = 10  # seconds per episode maybe needs to be removed later
FIXED_DELTA_SECONDS = 0.02 # seconds per frame
SHOW_PREVIEW = True # Show preview for debugging
NO_RENDERING = False # No rendering for training
SYNCRONOUST_MODE = False # Synchronous mode for training
SPIN = 10   # angle of spin for initail spawn
HEIGHT = 480 # height of image
WIDTH = 640 # width of image


class CarEnv(gym.Env):

    SHOW_CAM = SHOW_PREVIEW     # render camera
    front_camera = None
    CAMERA_POS_Z = 1.3      # camera position similar to tesla model 3
    CAMERA_POS_X = 1.4    # camera position similar to tesla model 3
    im_width = WIDTH
    im_height = HEIGHT

    def __init__(self):
        super(CarEnv, self).__init__()
        self.vehicle = None
        obs_dim = 3  # observation space dimension
        #vector to next waypoint (dx, dy) and heading angle
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

		# we setup discrete actions
        self.action_space = spaces.MultiDiscrete([9, 4])  # First variable with 9 possible actions for steering
        # Second variable with 4 possible actions for throttle/braking
        
        # Connect to carla server and load the world
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town02')
        self.world = self.client.get_world()

        # Set the world settings and blueprint library
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = NO_RENDERING
        self.settings.synchronous_mode = SYNCRONOUST_MODE
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find('vehicle.tesla.model3')

        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()

        # Generate waypoints
        town_map = self.world.get_map()
        all_waypoints = town_map.generate_waypoints(0.5) #note distance between waypoints in meters

        # make unique waypoints
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


    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
    
    
    def step(self, action):
        self.step_counter +=1
        steer = action[0]
        throttle = action[1]
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

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        distance_travelled, direction_difference = self.distance_from_waypoint(self.target_waypoint)
        # print steer and throttle every 50 steps
        if self.step_counter % 50 == 0:
            print('steer input from model:',steer,', throttle: ',throttle, ', km/h: ', kmh, ', distance travelled: ', distance_travelled)

        # start defining reward from each step
        reward = 0
        done = False
        vehicle_location  = self.vehicle.get_transform().location

    
        #TEST
        waypoint_distance  = vehicle_location.distance(self.target_waypoint.transform.location)

        # TODO- test
        reward = -waypoint_distance

        #punish for collision
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 100
            self.cleanup()

        if waypoint_distance < 1.0:  # Threshold to consider the waypoint reached
            self.current_waypoint = self.target_waypoint
            self.next_waypoint()
            reward = reward + 10  # Reward for reaching the waypoint
        # #reward for acceleration
        # if kmh < 10:
        #     reward = reward - 3
        # elif kmh < 15:
        #     reward = reward -1
        # elif kmh > 40:
        #     reward = reward - 5 #punish for going to fast
        # else:
        #     reward = reward + 1
        
        # reward for making distance
        if distance_travelled < 30:
            reward = reward - 1
        elif distance_travelled < 50:
            reward =  reward + 1
        else:
            reward = reward + 2

        #TODO Later
        # # track steering lock duration to prevent "chasing its tail"
        # # punish for steer lock up
        # if lock_duration>3:
        #     reward = reward - 150
        #     done = True
        #     self.cleanup()
        # elif lock_duration > 1:
        #     reward = reward - 20


        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            reward = reward + 10
            self.cleanup()

        observation = self.get_observation()

        return observation, reward, done, False, {}
    
    def get_observation(self):
        #TODO AttributeError: 'list' object has no attribute 'transform' correct "waypoint"
        veh_trans = self.vehicle.get_transform() 
        veh_loc = veh_trans.location
        veh_rot = veh_trans.rotation
        wpt = self.target_waypoint
        wpt_loc = wpt.transform.location
        wpt_rot = wpt.transform.rotation
      
        distance_x = (veh_loc.x - wpt_loc.x)
        distance_y = (veh_loc.y - wpt_loc.y)
        direction_difference = ( veh_rot.yaw - wpt_rot.yaw) % 180

        obs = np.array([
            distance_x,
            distance_y,
            direction_difference,
        ], dtype=np.float32)

        return obs 

    def collision_data(self, event):
        self.collision_hist.append(event)

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.collision_hist = []
        self.actor_list = []
        transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = None
        while self.vehicle is None:
            try:
        # TODO check vehicle spawn and init pos
                vehicle = self.world.spawn_actor(self.model_3, transform)
                self.vehicle = vehicle
            except:
                pass
        self.actor_list.append(self.vehicle)
        #TODO - print currencnt wp and load into a self.init_wp
        initial_pos = self.vehicle.get_location()
        curr_waypoint = initial_pos
        curr_distance = 1000
        for wp in self.waypoints:
            dist = curr_waypoint.distance(wp.transform.location)
            if dist < curr_distance:
                curr_distance =  dist
                selected_wp = wp

        self.current_waypoint = selected_wp
        self.target_waypoint = self.next_waypoint()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(0.5)

        # setup sensors and camera
        camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
        trans = self.vehicle.get_transform()
        # showing camera at the spawn point
        if self.SHOW_CAM:
            self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20),carla.Rotation(yaw =-180, pitch=-90)))

        # collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        self.episode_start = time.time()
        self.step_counter = 0

        self.world.tick()
        
        return self.get_observation() , {}
    

    def distance_from_waypoint(self, wp):     
        vehicle_transform = self.vehicle.get_transform()
        distance_to_wp = wp.transform.location.distance(vehicle_transform.location)
        direction_difference = (vehicle_transform.rotation.yaw - wp.transform.rotation.yaw) % 180
        return distance_to_wp, direction_difference

    def next_waypoint(self):
        next_wp = self.current_waypoint.next(5.0)[0]
        return next_wp
    