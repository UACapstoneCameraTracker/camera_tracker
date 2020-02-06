import camera_tracker.utils as utils
from states import (
    DetectState,
    TrackState,
    App,
    AppEnv,
)

cap = utils.get_stream()
env = AppEnv(cap)
app = App(env)
app.run()
