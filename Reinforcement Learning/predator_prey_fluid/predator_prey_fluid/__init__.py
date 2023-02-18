from gym.envs.registration import register

register(
	id="predator_prey_fluid/PredatorPreyFluid-v0",
	entry_point="predator_prey_fluid.envs:PredatorPreyEnv",
)