# Isaac Sim / Omniverse Kit Script Editor
# Creates a brand-new scene with physics + ground, then adds a dynamic ball with properties.
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
import omni.usd


def _ensure_physics_scene(stage, path="/World/PhysicsScene"):
    """Create a PhysicsScene with gravity if it doesn't exist."""
    if stage.GetPrimAtPath(path):
        return

    scene = UsdPhysics.Scene.Define(stage, path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Optional PhysX scene settings
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableCCDAttr(True)


def _create_ground_plane(stage, ground_path, size=50.0):
    """Create a ground plane with collision."""
    # Create mesh plane
    plane = UsdGeom.Mesh.Define(stage, ground_path)
    
    # Simple quad geometry
    half = size / 2.0
    plane.CreatePointsAttr([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])
    
    # Physics
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(plane.GetPrim())


def _bind_physx_material(stage, target_prim, mat_path, static_friction, dynamic_friction, restitution):
    """Create and bind a PhysX material to a prim."""
    mat_prim = stage.DefinePrim(mat_path, "Material")
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    physx_mat.CreateStaticFrictionAttr(static_friction)
    physx_mat.CreateDynamicFrictionAttr(dynamic_friction)
    physx_mat.CreateRestitutionAttr(restitution)
    
    # Bind material
    binding_api = UsdPhysics.MaterialBindingAPI.Apply(target_prim)
    binding_api.Bind(UsdPhysics.MaterialBindingAPI.DirectBinding, mat_prim.GetPath())


def main():
    stage = omni.usd.get_context().get_stage()

    # Stage conventions
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create /World if needed
    UsdGeom.Xform.Define(stage, "/World")

    # Physics
    _ensure_physics_scene(stage)

    # -----------------------------
    # Ground plane
    # -----------------------------
    ground_path = "/World/GroundPlane"
    _create_ground_plane(stage, ground_path, size=50.0)

    # Bind physics material to ground (friction/restitution)
    # NOTE: add_ground_plane usually creates collision already; material binding is optional but useful.
    _bind_physx_material(
        stage,
        stage.GetPrimAtPath(ground_path),
        "/World/GroundPhysMat",
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.05,
    )

    # -----------------------------
    # Ball
    # -----------------------------
    ball_path = "/World/Ball"
    ball = UsdGeom.Sphere.Define(stage, ball_path)
    ball.CreateRadiusAttr(0.12)
    UsdGeom.Xformable(ball.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.5))
    UsdGeom.Gprim(ball.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.1, 0.1)])

    # Collision + rigid body + mass
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(ball.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(ball.GetPrim()).CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(ball.GetPrim()).CreateMassAttr(0.5)

    # Ball physics material
    _bind_physx_material(
        stage,
        ball.GetPrim(),
        "/World/BallPhysMat",
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.7,
    )
    
    print("✓ Ball created at (0, 0, 1.5) with radius 0.12m, mass 0.5kg")
    print("✓ Ground plane created at z=0, size 50m x 50m")
    print("✓ Physics scene ready - press PLAY to simulate")


main()