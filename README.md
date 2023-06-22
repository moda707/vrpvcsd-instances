# vrpvcsd-instances
This repo provides VRP-VCSD instances generated to test the method proposed in the paper titled "Off-line approximate dynamic programming for the vehicle routing problem with a highly variable customer basis and stochastic demands". Test instances are named as "A_B" where A refers to the density level and B is the vehicle capacity in that instance. 

# Test instances
Test instances with VeryLow and Low density contain 10 customer configurations. In the other instances, each contains 500 customer configurations.

# Train instances
We also provide the code to generate instances for the training purpose. The file "instance_generator_vrpvcsd.py" contains 4 functions to generate and load customer configuration and demand scenarios.
