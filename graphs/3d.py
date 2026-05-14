from pathlib import Path

import numpy as np
import pinocchio as pin

import matplotlib.pyplot as plt


URDF_PATH = Path(__file__).resolve().parents[1] / "robot" / "iiwa.urdf"
EE_FRAME_NAME = "iiwa_link_ee"


def build_robot_model(urdf_path=URDF_PATH):
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return model, data


def forward_kinematics(q, model=None, data=None, ee_frame_name=EE_FRAME_NAME):
    """
    Прямая кинематика 7DOF манипулятора KUKA iiwa.

    Args:
        q: массив из 7 углов суставов [q1, ..., q7] в радианах.
        model: модель Pinocchio. Если None, модель загружается из URDF.
        data: data-объект Pinocchio. Если None, создается из model.
        ee_frame_name: имя фрейма end-effector в URDF.

    Returns:
        T_ee: однородная матрица 4x4 перехода world -> end-effector.
    """
    if model is None:
        model, data = build_robot_model()
    elif data is None:
        data = model.createData()

    q = np.asarray(q, dtype=float)
    if q.shape != (model.nq,):
        raise ValueError(f"Expected q with shape ({model.nq},), got {q.shape}")

    frame_id = model.getFrameId(ee_frame_name)
    if frame_id >= len(model.frames):
        raise ValueError(f"Frame {ee_frame_name!r} not found in model")

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    ee_pose = data.oMf[frame_id]
    return ee_pose.homogeneous


def end_effector_pose(q, model=None, data=None, ee_frame_name=EE_FRAME_NAME):
    T_ee = forward_kinematics(q, model, data, ee_frame_name)
    position = T_ee[:3, 3]
    rotation = T_ee[:3, :3]
    return position, rotation


def set_axes_equal(ax, points):
    min_values = points.min(axis=0)
    max_values = points.max(axis=0)
    center = (min_values + max_values) / 2.0
    radius = (max_values - min_values).max() / 2.0

    if radius == 0.0:
        radius = 0.1

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def interpolate_positions(source_time, source_positions, query_time):
    return np.column_stack(
        [
            np.interp(query_time, source_time, source_positions[:, axis])
            for axis in range(3)
        ]
    )


if __name__ == "__main__":

    model, data = build_robot_model()

    lines = np.genfromtxt("graphs/Cubes.txt", delimiter=",")

    current = []
    current_time = []
    target = []
    target_time = []

    for l in lines:
        q = l[2:9]
        T_ee = forward_kinematics(q, model, data)
        position = T_ee[:3, 3]
        if l[0] == 1:
            target.append(position)
            target_time.append(l[1])
        elif l[0] == 2:
            current.append(position)
            current_time.append(l[1])


    current = np.array(current)
    current_time = np.array(current_time)
    target = np.array(target)
    target_time = np.array(target_time)

    if current.size == 0:
        raise ValueError("No current trajectory points found: expected rows with marker 2")
    if target.size == 0:
        raise ValueError("No target trajectory points found: expected rows with marker 1")

    valid_current = (
        (current_time >= target_time.min())
        & (current_time <= target_time.max())
    )
    current = current[valid_current]
    current_time = current_time[valid_current]

    target_on_current_time = interpolate_positions(target_time, target, current_time)
    error = current - target_on_current_time
    error_norm = np.linalg.norm(error, axis=1)
    time = current_time - current_time[0]

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        current[:, 0],
        current[:, 1],
        current[:, 2],
        color="tab:blue",
        linewidth=2,
        label="current",
    )

    ax.plot(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        color="tab:red",
        linewidth=2,
        label="target",
    )

    step = max(len(current) // 120, 1)
    for current_point, target_point in zip(
        current[::step],
        target_on_current_time[::step],
    ):
        ax.plot(
            [target_point[0], current_point[0]],
            [target_point[1], current_point[1]],
            [target_point[2], current_point[2]],
            color="tab:gray",
            alpha=0.25,
            linewidth=0.8,
        )

    ax.scatter(
        current[0, 0],
        current[0, 1],
        current[0, 2],
        color="tab:green",
        s=40,
        label="current start",
    )
    ax.scatter(
        current[-1, 0],
        current[-1, 1],
        current[-1, 2],
        color="tab:orange",
        s=40,
        label="current finish",
    )

    ax.set_title("Current vs target end-effector trajectory")
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")
    ax.set_zlabel("Z, m")
    ax.legend()
    set_axes_equal(ax, np.vstack((current, target)))

    plt.figure(1)
    plt.plot(time, error[:, 0], label="dx")
    plt.plot(time, error[:, 1], label="dy")
    plt.plot(time, error[:, 2], label="dz")
    plt.plot(time, error_norm, color="black", linewidth=2, label="|error|")
    plt.title("Position error")
    plt.xlabel("Time")
    plt.ylabel("Error, m")
    plt.grid(True)
    plt.legend()

    print(f"Mean position error: {error_norm.mean():.6f} m")
    print(f"Max position error: {error_norm.max():.6f} m")

    plt.show()
