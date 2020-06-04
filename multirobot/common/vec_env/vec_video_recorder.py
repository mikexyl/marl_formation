import os

from baselines import logger
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder as BlRecoder
from gym.wrappers.monitoring import video_recorder


class VecVideoRecorder(BlRecoder):
    def __init__(self, venv, directory, record_video_trigger):
        """
        # Arguments
            venv: VecEnv to wrap
            directory: Where to save videos
            record_video_trigger:
                Function that defines when to start recording.
                The function takes the current number of step,
                and returns whether we should start recording or not.
            video_length: Length of recorded video
        """

        super(BlRecoder, self).__init__(venv)
        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory): os.mkdir(self.directory)

        self.epoch_id = 0
        self.cycle_id = 0

        self.recording = False
        self.recorded_frames = 0

    def start_video_recorder(self):
        self.close_video_recorder()

        base_path = os.path.join(self.directory,
                                 '{:04}-{:04}'.format(self.epoch_id, self.cycle_id))
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.venv,
            base_path=base_path,
            metadata={'epoch': self.epoch_id,
                      'cycle': self.cycle_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def step(self, actions, epoch=None, cycle=None):
        self.step_async(actions)
        return self.step_wait(epoch, cycle)

    def step_wait(self, epoch=None, cycle=None):
        obs, rews, dones, infos = self.venv.step_wait()

        self.epoch_id = epoch
        self.cycle_id = cycle

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if not self._video_enabled():
                logger.info("Saving video to ", self.video_recorder.path)
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos

    def _video_enabled(self):
        return self.record_video_trigger(self.cycle_id)
