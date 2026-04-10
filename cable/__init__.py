"""
cable -- Cable routing infrastructure for the tendon-driven manipulator.

Sub-modules
-----------
cable.pulley       PulleyBase hierarchy + URDF origin parser
cable.routing      CableRoute, CableSpring, CableRig, spring_zigzag_points
cable.drake_plant  DrakeCablePlant (headless FK)

All public names are re-exported here for backward compatibility::

    from cable import DrakeCablePlant, CableRig, PulleyBase  # all work
"""

from cable.pulley import (                     # noqa: F401
    _parse_urdf_part_origins,
    PulleyBase,
    CableStartPointR,
    CableStartPointL,
    DrivePulley,
    IdlerL,
    IdlerR,
    BigPulley,
    CableEndPointL,
    CableEndPointR,
)
from cable.routing import (                    # noqa: F401
    CableRoute,
    FixedBodyPoint,
    CableSpring,
    CableRig,
    spring_zigzag_points,
)
from cable.drake_plant import DrakeCablePlant  # noqa: F401
