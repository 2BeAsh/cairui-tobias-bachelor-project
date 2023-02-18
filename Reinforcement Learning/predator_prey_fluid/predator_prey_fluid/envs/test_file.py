import tianshou as ts
import numpy as np
import torch

# Lav dictionary og konverter til batch
dict_ex = {"agent" : np.arange(10), "target" : np.arange(10, 20)}
batch_ex = ts.data.Batch(dict_ex)
print(batch_ex)  # Giver det forventede

# Konverter til pytorch tensor eller numpy array.
# batch_conv1 er kopieret fra Tianshous "Deep Q Network" eksempel, men giver fejl.
# batch_conv2 og 3 er fra Tianshous "Understanding Batch"

obs = torch.tensor(batch_ex, dtype=torch.float)
batch_conv2 = batch_ex.to_torch(dtype=torch.int, device='cpu')

#print(batcc_conv1)  # Giver fejl
print(batch_conv2)  # Giver None
print(obs)
#print(batch_conv3)  # Giver None

"""
class PredatorPreyEnvArray(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, render_mode=None, size=2):
        self.size = size  # Boundary size
        self.window_size = 512  # PyGame window size
        self.max_steps = 1000  # Max allowed steps to catch prey


        # Observations are dictionary with agent's and target's location
        # Each location is a 2d array with x and y coordinate
        self.observation_space = spaces.Box(0, self.size, shape=(2, 2), dtype=np.float32)

        # One action, the angle value (constant velocity)
        self.action_space = spaces.Box(0, 2*np.pi, shape=(1,), dtype=np.float32)

        self.render_mode = render_mode

        self.window = None
        self.clock = None


    # Translate environment's state into an observation
    def _get_obs(self):
        loc_list = [np.array(self._agent_location), np.array(self._target_location)]
        #return loc_list
        return np.array(loc_list, dtype=object)


    def _get_info(self):
       return np.array([np.linalg.norm(self._agent_location - self._target_location, ord=1)])


    # Reset function
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Fix seed
        self.max_steps = 1000  # Reset timescale

        # Initial random position
        self._agent_location = self.np_random.uniform(low=0, high=self.size, size=2)

        # Make sure target is not at agent's position - I believe this might be redundant, or changed such that it does not start within epsilon distance
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(
                low=0, high=self.size, size=2,
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_fame()

        return observation, info


    def step(self, action):
        move_size = 1
        angle_moved = action
        x_move = move_size * np.cos(angle_moved)
        y_move = move_size * np.sin(angle_moved)
        move = np.array([x_move, y_move])

        # Make sure stays inside borders  - burde nok lave periodiske boundaries
        self._agent_location = np.clip(
            self._agent_location + move, a_min=0, a_max=self.size)

        # End of episode iff within epsilon of target
        epsilon = 1  # Catch target radius
        dist = self._get_info()
        terminated = dist <= epsilon
        if terminated:
            reward = 1000  # Get 1000 score for catching prey
        else:
            reward = -1  # Lose 1 score for each step taken
        observation = self._get_obs()
        info = dist

        # Time steps
        self.max_steps -= 1
        if self.max_steps <= 0:
            truncated = True
        else:
            truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    # Rendering
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):  # - VIRKER IKKE FØR FÅR INSTALLERET PYGAME.
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # Single grid size

        # Draw target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        if self.render_mode == "human":
            # Copy drawings from `canvas` to visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Fix framerates
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



"""
