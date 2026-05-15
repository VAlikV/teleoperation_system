from pathlib import Path

import numpy as np
import pinocchio as pin

import matplotlib.pyplot as plt


URDF_PATH = Path(__file__).resolve().parents[1] / "robot" / "iiwa.urdf"
EE_FRAME_NAME = "iiwa_link_ee"
TARGET_INDEX = 0
CURRENT_INDEX = 1
TORQUE_INDEX = 3


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


def end_effector_jacobian(q, model=None, data=None, ee_frame_name=EE_FRAME_NAME):
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
    return pin.computeFrameJacobian(
        model,
        data,
        q,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )


def estimate_end_effector_wrench(q, tau, model=None, data=None, ee_frame_name=EE_FRAME_NAME):
    """
    Оценивает wrench на end-effector из tau = J(q)^T F.

    Для 7DOF манипулятора система переопределена: J.T имеет размер 7x6.
    Поэтому используется least-squares решение, а первые 3 компоненты wrench
    в Pinocchio соответствуют линейной силе в world-aligned координатах.
    """
    if model is None:
        model, data = build_robot_model()
    elif data is None:
        data = model.createData()

    tau = np.asarray(tau, dtype=float)
    if tau.shape != (model.nv,):
        raise ValueError(f"Expected tau with shape ({model.nv},), got {tau.shape}")

    jacobian = end_effector_jacobian(q, model, data, ee_frame_name)
    wrench, *_ = np.linalg.lstsq(jacobian.T, tau, rcond=None)
    force = wrench[:3]
    return wrench, force


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


def interpolate_vectors(source_time, source_vectors, query_time):
    return np.column_stack(
        [
            np.interp(query_time, source_time, source_vectors[:, axis])
            for axis in range(source_vectors.shape[1])
        ]
    )


def estimate_forces(current_time, current_q, torque_time, torque, model, data):
    valid_torque = (
        (torque_time >= current_time.min())
        & (torque_time <= current_time.max())
    )
    torque_time = torque_time[valid_torque]
    torque = torque[valid_torque]

    if torque.size == 0:
        return torque_time, np.empty((0, 3))

    q_on_torque_time = interpolate_vectors(current_time, current_q, torque_time)
    forces = []
    for q, tau in zip(q_on_torque_time, torque):
        _, force = estimate_end_effector_wrench(q, tau, model, data)
        forces.append(force)

    return torque_time, np.array(forces)


if __name__ == "__main__":

    model, data = build_robot_model()

    lines = np.genfromtxt("graphs/Table.txt", delimiter=",")[1000:]

    print(len(lines))

    current = []
    current_q = []
    current_time = []
    target = []
    target_time = []
    torque = []
    torque_time = []

    for l in lines:
        index = int(l[0])
        values = l[2:9]
        if index == TARGET_INDEX:
            T_ee = forward_kinematics(values, model, data)
            position = T_ee[:3, 3]
            target.append(position)
            target_time.append(l[1])
        elif index == CURRENT_INDEX:
            T_ee = forward_kinematics(values, model, data)
            position = T_ee[:3, 3]
            current.append(position)
            current_q.append(values)
            current_time.append(l[1])
        elif index == TORQUE_INDEX:
            torque.append(values)
            torque_time.append(l[1])


    current = np.array(current)
    current_q = np.array(current_q)
    current_time = np.array(current_time)
    target = np.array(target)
    target_time = np.array(target_time)
    torque = np.array(torque)
    torque_time = np.array(torque_time)

    if current.size == 0:
        raise ValueError(f"No current trajectory points found: expected rows with marker {CURRENT_INDEX}")
    if target.size == 0:
        raise ValueError(f"No target trajectory points found: expected rows with marker {TARGET_INDEX}")

    valid_current = (
        (current_time >= target_time.min())
        & (current_time <= target_time.max())
    )
    current = current[valid_current]
    current_q = current_q[valid_current]
    current_time = current_time[valid_current]

    target_on_current_time = interpolate_positions(target_time, target, current_time)
    error = current - target_on_current_time
    error_norm = np.linalg.norm(error, axis=1)
    time = current_time - current_time[0]

    force_time = np.array([])
    forces = np.empty((0, 3))
    force_plot_time = np.array([])
    if torque.size:
        force_time, forces = estimate_forces(
            current_time,
            current_q,
            torque_time,
            torque,
            model,
            data,
        )
        force_norm = np.linalg.norm(forces, axis=1)
        visible_force = force_norm > 1e-12
        force_time = force_time[visible_force]
        forces = forces[visible_force]

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

    if forces.size:
        force_positions = interpolate_positions(current_time, current, force_time)
        force_norm = np.linalg.norm(forces, axis=1)
        force_plot_time = force_time - current_time[0]

        force_step = max(len(forces) // 40, 1)
        trajectory_span = np.ptp(np.vstack((current, target)), axis=0).max()
        max_arrow_length = max(trajectory_span * 0.12, 0.03)
        scale = max_arrow_length / force_norm.max()
        ax.quiver(
            force_positions[::force_step, 0],
            force_positions[::force_step, 1],
            force_positions[::force_step, 2],
            forces[::force_step, 0] * scale,
            forces[::force_step, 1] * scale,
            forces[::force_step, 2] * scale,
            color="tab:purple",
            linewidth=1.1,
            arrow_length_ratio=0.25,
            normalize=False,
            label="estimated force",
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

    if forces.size:
        plt.figure(2)
        plt.plot(force_plot_time, forces[:, 0], label="Fx")
        plt.plot(force_plot_time, forces[:, 1], label="Fy")
        plt.plot(force_plot_time, forces[:, 2], label="Fz")
        plt.plot(
            force_plot_time,
            np.linalg.norm(forces, axis=1),
            color="black",
            linewidth=2,
            label="|F|",
        )
        plt.title("Estimated end-effector force")
        plt.xlabel("Time")
        plt.ylabel("Force")
        plt.grid(True)
        plt.legend()
    else:
        print(f"No force vectors plotted: expected torque rows with marker {TORQUE_INDEX}")

    print(f"Mean position error: {error_norm.mean():.6f} m")
    print(f"Max position error: {error_norm.max():.6f} m")

    plt.show()
