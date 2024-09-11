"""
CARLA-ESMINI Integration Script
"""

import argparse
import os
import sys
import time
from datetime import datetime
import glob
import numpy as np
import random


try:
    sys.path.append(glob.glob('./carla/dist/carla-0.9.14-py3.7-win-amd64.egg')[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')

try:
    sys.path.insert(0,r'C:\Users\ashokkumar\source\repos\AD\carla')
except IndexError:
    pass
import carla

from agents.navigation.behavior_agent import BehaviorAgent  # type: ignore 
from agents.navigation.basic_agent import BasicAgent  # type: ignore 


def main():
    actor_list = []
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(15.0)
        world = client.get_world()
        world = client.load_world('Town02')

        # Set up CARLA environment
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # Get blueprint library for spawning vehicles
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        bp = blueprint_library.find('vehicle.tesla.model3')

        town_map = world.get_map()

        # Generate waypoints
        all_waypoints = town_map.generate_waypoints(10)  #note distance between waypoints in meters
        waypoints = []
        for wp in all_waypoints:
            if len(waypoints)==0:
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

        # Display waypoints
        for wp in waypoints:
            world.debug.draw_string(wp.transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=300.0,
                persistent_lines=True)
        print('Number of waypoints:',len(waypoints))

        transform = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(bp, transform)  
        actor_list.append(ego_vehicle)
        spectator = world.get_spectator()
        spectator_transform =  ego_vehicle.get_transform()
        spectator_transform.location += carla.Location(z = 10.0)
        world.get_spectator().set_transform(spectator_transform)
        actor_list.append(spectator)

        my_waypoint = ego_vehicle.get_transform().location
        curr_distance = 1000
        for wp in waypoints:
            dist = my_waypoint.distance(wp.transform.location)
            if dist < curr_distance:
                curr_distance =  dist
                selected_wp = wp
        # initialize agent
        agent = BasicAgent(ego_vehicle)
        waypoint_counter = 0

        next_wp = selected_wp.next(5.0)[0]
        destination = next_wp.transform.location
        agent.set_destination(destination)

        while True:
                if agent.done():
                    next_wp = selected_wp.next(5.0)[0]
                    destination = next_wp.transform.location
                    agent.set_destination(destination)
                    world.debug.draw_string(next_wp.transform.location, '^', draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0), life_time=2.0,
                    persistent_lines=True)

                    print("The target has been reached, setting next waypoint")
                    selected_wp = next_wp
                    waypoint_counter += 1
                if waypoint_counter > 100:
                    print("We have reached the end of the waypoints")
                    break
                ego_vehicle.apply_control(agent.run_step())
                world.tick()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Exiting gracefully...")
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

    finally:
        time.sleep(5)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == '__main__':

    main()