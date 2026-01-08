# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import asyncio
import math
import time

import carb.eventdispatcher
import carb.events
import carb.settings
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import PhysxSchema, UsdPhysics

# TIMELINE / STAGE
USE_CUSTOM_TIMELINE_SETTINGS = False
USE_FIXED_TIME_STEPPING = False
PLAY_EVERY_FRAME = True
PLAY_DELAY_COMPENSATION = 0.0
SUBSAMPLE_RATE = 1
STAGE_FPS = 30.0

# PHYSX
USE_CUSTOM_PHYSX_FPS = False
PHYSX_FPS = 60.0
MIN_SIM_FPS = 30

# Simulations can also be enabled/disabled at runtime
DISABLE_SIMULATIONS = False

# APP / RENDER
LIMIT_APP_FPS = False
APP_FPS = 120

# Duration after which to clear subscribers and print the cached events
SUBSCRIBER_WALL_TIME_LIMIT_SEC = 0.5
PRINT_EVENTS = True


def on_timeline_event(event: omni.timeline.TimelineEventType):
    global wall_start_time
    global timeline_sub
    global timeline_events
    elapsed_wall_time = time.time() - wall_start_time

    # Cache only time advance events
    if event.type == omni.timeline.TimelineEventType.CURRENT_TIME_TICKED.value:
        event_name = omni.timeline.TimelineEventType(event.type).name
        event_payload = event.payload
        timeline_events.append((elapsed_wall_time, event_name, event_payload))

    # Clear subscriber and print cached events
    if elapsed_wall_time > SUBSCRIBER_WALL_TIME_LIMIT_SEC:
        if timeline_sub is not None:
            timeline_sub.unsubscribe()
            timeline_sub = None
        num_events = len(timeline_events)
        fps = num_events / SUBSCRIBER_WALL_TIME_LIMIT_SEC
        print(f"[timeline] captured {num_events} events with aprox {fps} FPS")
        if PRINT_EVENTS:
            for i, (wall_time, event_name, payload) in enumerate(timeline_events):
                print(f"\t[timeline][{i}]\ttime={wall_time:.4f};\tevent={event_name};\tpayload={payload}")


def on_physics_step(dt: float):
    global wall_start_time
    global physx_events
    global physx_sub
    elapsed_wall_time = time.time() - wall_start_time

    # Cache physics events
    physx_events.append((elapsed_wall_time, dt))

    # Clear subscriber and print cached events
    if elapsed_wall_time > SUBSCRIBER_WALL_TIME_LIMIT_SEC:
        # Physics unsubscription needs to be defered from the callback function
        # see: '[Error] [omni.physx.plugin] Subscription cannot be changed during the event call'
        async def clear_physx_sub_async():
            global physx_sub
            if physx_sub is not None:
                physx_sub.unsubscribe()
                physx_sub = None

        asyncio.ensure_future(clear_physx_sub_async())
        num_events = len(physx_events)
        fps = num_events / SUBSCRIBER_WALL_TIME_LIMIT_SEC
        print(f"[physics] captured {num_events} events with aprox {fps} FPS")
        if PRINT_EVENTS:
            for i, (wall_time, dt) in enumerate(physx_events):
                print(f"\t[physics][{i}]\ttime={wall_time:.4f};\tdt={dt};")


def on_stage_render_event(event: carb.eventdispatcher.Event):
    global wall_start_time
    global stage_render_sub
    global stage_render_events
    elapsed_wall_time = time.time() - wall_start_time

    event_name = event.event_name
    event_payload = event.payload
    stage_render_events.append((elapsed_wall_time, event_name, event_payload))

    if elapsed_wall_time > SUBSCRIBER_WALL_TIME_LIMIT_SEC:
        if stage_render_sub is not None:
            stage_render_sub.reset()
            stage_render_sub = None
        num_events = len(stage_render_events)
        fps = num_events / SUBSCRIBER_WALL_TIME_LIMIT_SEC
        print(f"[stage render] captured {num_events} events with aprox {fps} FPS")
        if PRINT_EVENTS:
            for i, (wall_time, event_name, payload) in enumerate(stage_render_events):
                print(f"\t[stage render][{i}]\ttime={wall_time:.4f};\tevent={event_name};\tpayload={payload}")


def on_app_update(event: carb.eventdispatcher.Event):
    global wall_start_time
    global app_sub
    global app_update_events
    elapsed_wall_time = time.time() - wall_start_time

    event_name = event.event_name
    event_payload = event.payload
    app_update_events.append((elapsed_wall_time, event_name, event_payload))

    if elapsed_wall_time > SUBSCRIBER_WALL_TIME_LIMIT_SEC:
        if app_sub is not None:
            app_sub.reset()
            app_sub = None
        num_events = len(app_update_events)
        fps = num_events / SUBSCRIBER_WALL_TIME_LIMIT_SEC
        print(f"[app] captured {num_events} events with aprox {fps} FPS")
        if PRINT_EVENTS:
            for i, (wall_time, event_name, payload) in enumerate(app_update_events):
                print(f"\t[app][{i}]\ttime={wall_time:.4f};\tevent={event_name};\tpayload={payload}")


stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()


if USE_CUSTOM_TIMELINE_SETTINGS:
    # Ideal to make simulation and animation synchronized.
    # Default: True in editor, False in standalone.
    # NOTE:
    # - It may limit the frame rate (see 'timeline.set_play_every_frame') such that the elapsed wall clock time matches the frame's delta time.
    # - If the app runs slower than this, animation playback may slow down (see 'CompensatePlayDelayInSecs').
    # - For performance benchmarks, turn this off or set a very high target in `timeline.set_target_framerate`
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", USE_FIXED_TIME_STEPPING)

    # This compensates for frames that require more computation time than the frame's fixed delta time, by temporarily speeding up playback.
    # The parameter represents the length of these "faster" playback periods, which means that it must be larger than the fixed frame time to take effect.
    # Default: 0.0
    # NOTE:
    # - only effective if `useFixedTimeStepping` is set to True
    # - setting a large value results in long fast playback after a huge lag spike
    carb.settings.get_settings().set("/app/player/CompensatePlayDelayInSecs", PLAY_DELAY_COMPENSATION)

    # If set to True, no frames are skipped and in every frame time advances by `1 / TimeCodesPerSecond`.
    # Default: False
    # NOTE:
    # - only effective if `useFixedTimeStepping` is set to True
    # - simulation is usually faster than real-time and processing is only limited by the frame rate of the runloop
    # - useful for recording
    # - same as `carb.settings.get_settings().set("/app/player/useFastMode", PLAY_EVERY_FRAME)`
    timeline.set_play_every_frame(PLAY_EVERY_FRAME)

    # Timeline sub-stepping, i.e. how many times updates are called (update events are dispatched) each frame.
    # Default: 1
    # NOTE: same as `carb.settings.get_settings().set("/app/player/timelineSubsampleRate", SUBSAMPLE_RATE)`
    timeline.set_ticks_per_frame(SUBSAMPLE_RATE)

    # Time codes per second for the stage
    # NOTE: same as `stage.SetTimeCodesPerSecond(STAGE_FPS)` and `carb.settings.get_settings().set("/app/stage/timeCodesPerSecond", STAGE_FPS)`
    timeline.set_time_codes_per_second(STAGE_FPS)


# Create a PhysX scene to set the physics time step
if USE_CUSTOM_PHYSX_FPS:
    physx_scene = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            break
    if physx_scene is None:
        physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

    # Time step for the physics simulation
    # Default: 60.0
    physx_scene.GetTimeStepsPerSecondAttr().Set(PHYSX_FPS)

    # Minimum simulation frequency to prevent clamping; if the frame rate drops below this,
    # physics steps are discarded to avoid app slowdown if the overall frame rate is too low.
    # Default: 30.0
    # NOTE: Matching `minFrameRate` with `TimeStepsPerSecond` ensures a single physics step per update.
    carb.settings.get_settings().set("/persistent/simulation/minFrameRate", MIN_SIM_FPS)


# Throttle Render/UI/Main thread update rate
if LIMIT_APP_FPS:
    # Enable rate limiting of the main run loop (UI, rendering, etc.)
    # Default: False
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", LIMIT_APP_FPS)

    # FPS limit of the main run loop (UI, rendering, etc.)
    # Default: 120
    # NOTE: disabled if `/app/player/useFixedTimeStepping` is False
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", int(APP_FPS))


# Simulations can be selectively disabled (or toggled at specific times)
if DISABLE_SIMULATIONS:
    carb.settings.get_settings().set("/app/player/playSimulations", False)


# Start the timeline
timeline.set_current_time(0)
timeline.set_end_time(SUBSCRIBER_WALL_TIME_LIMIT_SEC + 1)
timeline.set_looping(False)
timeline.play()
timeline.commit()
wall_start_time = time.time()

# Subscribe and cache various events for a limited duration
timeline_events = []
timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(on_timeline_event)
physx_events = []
physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)
stage_render_events = []
stage_render_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
    event_name=omni.usd.get_context().stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME, True),
    on_event=on_stage_render_event,
    observer_name="subscribers_and_events.on_stage_render_event",
)
app_update_events = []
app_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
    event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
    on_event=on_app_update,
    observer_name="subscribers_and_events.on_app_update",
)

# Run the application for a while to trigger the events, with a buffer to ensure subscribers have enough wall-clock time
num_app_updates = int(math.ceil(SUBSCRIBER_WALL_TIME_LIMIT_SEC * STAGE_FPS * 4))
for _ in range(num_app_updates):
    simulation_app.update()

print(f"Finished running the application for {num_app_updates} updates.")
print(f"Wall time: {time.time() - wall_start_time:.4f} seconds")
print(f"Number of timeline events: {len(timeline_events)}")
print(f"Number of physics events: {len(physx_events)}")
print(f"Number of stage render events: {len(stage_render_events)}")
print(f"Number of app update events: {len(app_update_events)}")

simulation_app.close()
