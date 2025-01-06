### Abstract

Autonomous navigation is a critical component in the development of self-driving vehicles. This thesis explores the application of \ac{DRL}  for autonomous navigation within the CARLA simulator, an open-source simulation platform designed for autonomous driving research. The work focuses on training agents to make optimal driving decisions in dynamic urban environments without human intervention. Deep learning models were combined with reinforcement learning techniques so the vehicle could perceive its surroundings, predict outcomes, and take appropriate actions to navigate safely.

The study evaluates the performance of a state-of-the-art \ac{DRL} algorithm, \ac{PPO}, while actively addressing and overcoming challenges like sparse rewards, training stability, and generalization to unseen scenarios. A custom reward function was crafted to prioritize collision avoidance, lane-keeping, smooth acceleration, and steering, ensuring the agent adheres to realistic driving behavior. Experimental results demonstrated that the \ac{DRL}-based agent achieved promising performance in various simulated driving tasks, including obstacle avoidance, lane-following, and intersection handling. Furthermore, the agent exhibited robust performance in novel and complex environments, highlighting its capacity to generalize and adapt efficiently.

This thesis contributes to the understanding of integrating \ac{DRL} for autonomous navigation in simulation-based environments and highlights the CARLA simulator’s role as a robust testing ground. The findings lay the groundwork for further advancements in sim-to-real transfer and scalable training methods for autonomous vehicles.
