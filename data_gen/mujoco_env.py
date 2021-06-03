"""
This is a class structure for mujoco environments.
Base functions inherited from gym.
Additional functions needed for trajectory optimization algorithms are included.
"""

import os

import mujoco_py
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import time as timer
import dm_control.mujoco as mujoco
from dm_control.rl import control
import trajopt.envs.tasks as tasks
from dm_control.suite import base
import collections

DEFAULT_SIZE = 500

class TouchTable(base.Task):

    def __init__(self):
        super(TouchTable, self).__init__()

    def initialize_episode(self, physics):
        super(TouchTable, self).initialize_episode(physics)
        

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        table_pos=physics.named.data.geom_xpos["tabletop"]
        wrist_pos=physics.named.data.geom_xpos["herb/wam_1//unnamed_geom_24"]
        return -np.sum(np.abs(table_pos-wrist_pos))


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    #@profile
    def __init__(self, model_path, frame_skip, has_robot=True):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco.Physics.from_xml_path(model_path)
        task=TouchTable()
        self.sim = control.Environment(self.model, task, time_limit=1000, control_timestep=0.01)
        self.data = self.sim.physics._data

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        if has_robot:
            self.init_qpos = np.concatenate((self.init_qpos, self.data.qpos.ravel().copy()[self.init_qpos.shape[0]:]))#self.data.qpos.ravel().copy()
        else:
            self.init_qpos=self.data.qpos.ravel().copy()
        self.init_qvel = np.zeros(self.data.qvel.ravel().copy().shape)
        
        if has_robot:
            self.robot_reset()
        observation=self.model.position()
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        bounds = self.model.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    def get_env_state(self):
        return np.concatenate((self.data.qpos.ravel(), self.data.qvel.ravel()))

    def set_env_state(self, state):
        self.sim.physics.set_state(state)

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    #@profile
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.model.nq,) and qvel.shape == (self.model.model.nv,)
        self.sim.physics.set_state(np.concatenate((qpos, qvel)))
        self.sim.physics.forward()

    @property
    def dt(self):
        return self.model.model.opt.timestep * self.frame_skip

    #@profile
    def do_simulation(self, ctrl, n_frames):
        ctrl[7]=0
        ctrl[10]=0
        ctrl[9]=0
        ctrl[12]=0
        ctrl[14]=0
        ctrl=np.concatenate((ctrl[:7], np.zeros(7), ctrl[7:], np.zeros(8)))
        for _ in range(n_frames):
            self.sim.step(ctrl)

    def mj_render(self):
        u=0
        
    # def _get_viewer(self):
    #     return None
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([
            state.qpos.flat, state.qvel.flat])

    # -----------------------------

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
        self.mujoco_render_frames = False

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640,480),
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            o = self.reset()
            d = False
            t = 0
            arrs = []
            t0 = timer.time()
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1,:,:])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + str(ep) + ".mp4"
            skvideo.io.vwrite( file_name, np.asarray(arrs))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f"% (t1-t0))
    
    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
