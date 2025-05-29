import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

matplotlib.use('TkAgg')




class ModelPredictiveControl:
    def __init__(self):

        # Reference or set point the controller will achieve.
        self.initial_pose = np.array([0,4,0,0])

        self.reference = [10, 10, 0]  # x,y,yaw
        self.target_velocity = 15
        self.trajectory = None
        self.obstacles = np.array([])

        self.lookahead_distance = 100

        self.horizon = 20
        self.dt = 0.25
        self.L = 10.0  # wheelbase


        # COSTS
        self.dis_cost = 0.9
        self.dis_traj_cost = 0.6
        self.ang_cost = 0.2

        self.smooth_cost = 0.4
        self.travel_dist_cost = 0.01
        self.stopping_cost = 0.0
        self.velocity_cost = 0.8

        self.obs_cost = 0.2

        # OTHER
        self.obs_radius = 0.9
        self.options = {'FIG_SIZE': [16, 16], 'OBSTACLES': True} # simulator options
        self.simulation_start_frame = 0
        self.simulation_end_frame = 250

        # Static obstacles
        #self.obstacles = np.array([[5,2,1],  # x,y,size
        #                           [20,2.4,2]])


    def set_initial_position(self, x, y, yaw, v):
        """
        pose: np.array shape (4,)   in order: x,y,yaw,v
              x,y in meters,  yaw in rad and v in m/s
        """
        self.initial_pose = np.array([x, y, yaw, v])


    def set_range(self, start_frame, end_frame):
        self.simulation_start_frame = start_frame
        self.simulation_end_frame = end_frame


    def set_reference_trajectory(self, trajectory):
        """
        trajectory: np.array shape (N,3)   where axis 1 is x,y,yaw
        """
        assert trajectory.ndim == 2 and trajectory.shape[1] == 3
        self.trajectory = trajectory


    def set_reference_trajectory_matrix(self, matrices):
        """
        trajectory: iterable of numpy arrays shaped (4,4)
        """
        trajectory_array = np.zeros((len(matrices), 3))
        for i, matrix in enumerate(matrices):
            trajectory_array[i,:2] = matrix[:2,3]
            trajectory_array[i, 2] = ModelPredictiveControl.yaw_from_matrix(matrix)
        self.trajectory = trajectory_array

    def set_target_velocity(self, velocity):
        """
        velocity: float
        """
        assert isinstance(velocity, (float, int))
        self.target_velocity = velocity

    def set_lookahead_distance(self, distance):
        """
        distance: float
        """
        assert isinstance(distance, (float, int))
        self.lookahead_distance = distance

    def set_obstacles(self, obstacles):
        """
        obstacles: np.array shape (N,3)
        """
        assert obstacles.ndim==2 and obstacles.shape[1]==3
        self.obstacles = obstacles

    def _set_reference_from_trajectory(self, state):
        # return reference POINT
        if self.trajectory is not None:
            vehicle_position = state[:2]
            path_points = self.trajectory

            # find the point that is closest to the car
            distances = np.linalg.norm(self.trajectory[:, :2] - vehicle_position.reshape(1, -1), axis=1)
            closest_index = np.argmin(distances)

            # find the point that is lookahead_distance in front of car
            total_distance = 0.0
            target_index = closest_index

            for i in range(closest_index, len(path_points) - 1):
                segment_distance = np.linalg.norm(path_points[i + 1] - path_points[i])
                total_distance += segment_distance
                target_index = i + 1
                if total_distance >= self.lookahead_distance:
                    break

            # Return the target point
            self.closest  = path_points[closest_index]
            self.reference =  path_points[target_index]




    def _get_reference_trajectory(self, state):
        # return reference TRAJCETORY
        if self.trajectory is not None:
            vehicle_position = state[:2]
            path_points = self.trajectory

            # find the point that is closest to the car
            distances = np.linalg.norm(self.trajectory[:,:2] - vehicle_position.reshape(1, -1), axis=1)
            closest_index = np.argmin(distances)


            # calculate average trajectory spacing:
            closest_trajectory_point = path_points[closest_index]
            end_trajectory_point = path_points[len(path_points) - 1]
            average_trajectory_spacing = np.linalg.norm(end_trajectory_point - closest_trajectory_point) / (len(path_points) - closest_index)
            average_prediction_spacing = self.target_velocity * self.dt
            index_for_each_prediction_point = closest_index + np.arange(0, self.horizon)  * average_prediction_spacing / average_trajectory_spacing
            index_for_each_prediction_point = np.clip(index_for_each_prediction_point.astype(int), 0, len(path_points)-1)


            # Return the target trajectory
            return self.trajectory[index_for_each_prediction_point]

    @staticmethod
    def yaw_from_matrix(matrix):
        rot = matrix[:3,:3]
        yaw, _, _ = R.from_matrix(rot).as_euler('zyx')
        return yaw

    @staticmethod
    def _normalize_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi


    @staticmethod
    def _get_angle_diff(angle_1, angle_2):
        angle_diff_1 = (angle_1 - angle_2)[..., None] % (2 * np.pi)
        angle_diff_2 = 2 * np.pi - angle_diff_1
        angle_diff = np.amin(np.concatenate((angle_diff_1, angle_diff_2), axis=-1), axis=-1)
        return angle_diff


    def _plant_model(self, prev_state, dt, pedal, steering):
        x_prev = prev_state[0]
        y_prev = prev_state[1]
        psi_prev = prev_state[2]
        v_prev = prev_state[3]

        x_t = x_prev + v_prev * dt * np.cos(psi_prev)# + 0.5 * dt*dt * pedal * np.cos(psi_prev)
        y_t = y_prev + v_prev * dt * np.sin(psi_prev)# + 0.5 * dt*dt * pedal * np.sin(psi_prev)

        v_t = 0.97 * v_prev + pedal + 1.4 * dt

        psi_t = psi_prev + dt * v_prev / self.L * np.tan(steering)
        psi_t = self._normalize_angle(psi_t)


        return [x_t, y_t, psi_t, v_t]



    # def own_cost_function(self,u, *args):
    #     state = args[0]
    #     ref = args[1]
    #
    #     cost = 0.0
    #
    #     pos_weight, angle_weight, vel_weight = 0.9, 0.3, 0.05
    #     for t in range(self.horizon):
    #         pos_t = self.plant_model(state, self.dt, *u[2*t: 2*(t+1)])
    #         pos_t_array = np.array(pos_t)
    #         cost_pos = np.linalg.norm(pos_t_array[:2] - ref[:2])
    #         cost_angle = normalize_angle(pos_t_array[2] - ref[2]) ** 2
    #         cost_velocity = np.abs(pos_t_array[3] - 0.0)
    #         cost += pos_weight * cost_pos + angle_weight * cost_angle + cost_velocity * vel_weight
    #     return cost


    def _cost_function(self, u, *args):

        u = u.reshape(self.horizon, 2)

        state = np.array(args[0])  # current location, yaw and v   # shape (4,)
        ref = np.array(args[1])    # target point and direction   #shape (3,)
        ref_trajectory = np.array(args[2]) # shape (horizon, 3) # one reference for each predicted state

        # self.set_reference_from_trajectory(state=state[:2])

        state_history = np.array([state])  #shape (1,4) where 4 is [x,y,phi,v]

        for i in range(self.horizon):
            control = u[i]
            state = np.array(self._plant_model(state, self.dt, control[0], control[1]))
            state_history = np.concatenate((state_history, state[None, ...]))
        # state_history has shape  # (horizon + 1, 4).  4 is [x,y,phi,v]


        # target cost (to reference point)
        pos_diff = np.linalg.norm(ref[:2] - state_history[:, :2], axis=-1, ord=2)  # output shape (horizon, )
        cost = np.sum(pos_diff[1:]) * self.dis_cost

        # target cost (to reference trajectory)
        if self.dis_traj_cost > 0.0:
            pos_diff = np.linalg.norm(ref_trajectory[:,:2] - state_history[1:, :2], axis=-1, ord=2)  # output shape (horizon, )
            cost += np.sum(pos_diff[1:]) * self.dis_traj_cost


        angle_diff = self._get_angle_diff(state_history[1:, 2], ref[2])
        cost += np.sum(angle_diff) * self.ang_cost


        # smoothness cost (steering angle and pedal input)
        if self.smooth_cost > 0:
            cost += np.sum(np.diff(u, axis=0)) * self.smooth_cost

        # stopping cost
        #if self.stopping_cost > 0:
        #    cost += np.sum(np.abs(state_history[:, 3])) * self.stopping_cost

        # travel distance cost
        if self.travel_dist_cost > 0:
            cost += np.sum(
                np.linalg.norm(np.diff(state_history[:,2], axis=0), ord=2, axis=-1)) * self.travel_dist_cost

        # obstacle cost front axis
        if self.obs_cost > 0 and len(self.obstacles) > 0:
            positions_car_front = state_history[:, :2] + 0.2* self.L * np.column_stack((
                                                                        np.cos(state_history[:, 2]),  # x-component
                                                                        np.sin(state_history[:, 2])   # y-component
                                                                       ))

            dist = np.linalg.norm(self.obstacles[:, :2] - positions_car_front[1:, None, :], axis=-1, ord=2) - self.obstacles[:, 2]
            dist = np.clip(dist, a_min=0, a_max=None) + 1e-8
            cost += np.sum((1 / dist - 1 / self.obs_radius) * (dist < self.obs_radius)) * self.obs_cost


        if self.velocity_cost > 0:
            vel_diff = np.abs(self.target_velocity - state_history[:, 3])  # output shape (horizon, )
            cost += np.sum(vel_diff[1:]) * self.velocity_cost

        # # obstacle cost back axis
        # if self.obs_cost > 0 and len(self.obstacles) > 0:
        #     dist = np.linalg.norm(self.obstacles[:, :2] - state_history[1:, None, :2], axis=-1, ord=2) - self.obstacles[:, 2]
        #     dist = np.clip(dist, a_min=0, a_max=None) + 1e-8
        #     cost += np.sum((1 / dist - 1 / self.obs_radius) * (dist < self.obs_radius)) * self.obs_cost

        return cost

########################################################################################################################

    def simulate(self, plotting=True, run_func=None, rt_plotting=True):
        start = time.process_time()
        # Simulator Options
        options = self.options
        FIG_SIZE = options['FIG_SIZE']  # [Width, Height]
        OBSTACLES = options['OBSTACLES']

        mpc = self

        num_inputs = 2
        u = np.zeros(mpc.horizon * num_inputs)
        bounds = []

        # Set bounds for inputs bounded optimization.
        for i in range(mpc.horizon):
            bounds += [[-1, 1]]
            bounds += [[-0.8, 0.8]]


        # States over time
        state_i = mpc.initial_pose[np.newaxis, :]  # shape (4,) to (1,4)
        u_i = np.array([[0, 0]])  # throttle, steering
        ref_i = np.array([[0, 0, 0]])
        predict_info = [state_i]
        obs_info = [mpc.obstacles.copy()]

        sim_total = mpc.simulation_end_frame - mpc.simulation_start_frame

        for i in range(1, sim_total + 1):
            if run_func is not None:
                run_func(i + mpc.simulation_start_frame - 1)
            mpc._set_reference_from_trajectory(state=state_i[-1, :2])
            ref_trajectory = mpc._get_reference_trajectory(state_i[-1, :2])
            ref = mpc.reference  # for tracjectory
            closest = mpc.closest
            # mpc.obstacles[1, 1] -= 0.1  # move along y axis
            obs_info.append(mpc.obstacles.copy())

            u = np.delete(u, 0)
            u = np.delete(u, 0)
            u = np.append(u, u[-2])
            u = np.append(u, u[-2])
            start_time = time.time()

            # Non-linear optimization.
            u_solution = minimize(mpc._cost_function, u, (state_i[-1], ref, ref_trajectory),
                                  method='SLSQP',
                                  bounds=bounds,
                                  tol=1e-5)
            print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time, 5)))
            u = u_solution.x
            y = mpc._plant_model(state_i[-1], mpc.dt, u[0], u[1])


            predicted_state = np.array([y])
            for j in range(1, mpc.horizon):
                predicted = mpc._plant_model(predicted_state[-1], mpc.dt, u[2 * j], u[2 * j + 1])
                predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)

            predict_info += [predicted_state]
            state_i = np.append(state_i, np.array([y]), axis=0)
            u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)
            ref_i = np.append(ref_i, np.array([ref]), axis=0)

            if rt_plotting:
                #real time plotting
                plt.xlim(np.min(mpc.trajectory[:, 0]) - 20, np.max(mpc.trajectory[:, 0]) + 20)
                plt.ylim(np.min(mpc.trajectory[:, 1]) - 20, np.max(mpc.trajectory[:, 1]) + 20)

                plt.plot(mpc.trajectory[:, 0], mpc.trajectory[:, 1], 'g.') # reference path in green
                plt.plot(closest[0], closest[1], 'yx')  # reference (yellow)
                plt.plot(ref[0], ref[1], 'x', color='orange')  # goal(orange)
                plt.plot(predicted_state[:, 0], predicted_state[:, 1], '.', color=[0.8, 0.8, 1], zorder=10) # predicted path (light blue)
                if mpc.obstacles.ndim==2:
                    plt.scatter(mpc.obstacles[:,0], mpc.obstacles[:,1], 500*mpc.obstacles[:,2], color='red')  # obstacles (red)
                plt.quiver(y[0], y[1], 30 * np.cos(y[2]), 30 * np.sin(y[2]),   # current pose(dark blue)
                           angles='xy', scale_units='xy', scale=1,
                           color='blue', width=0.01, zorder=9)

                # plt.scatter(y[0], y[1], 300, color='blue')
                # plt.pause(0.01)
                plt.pause(6)
                plt.clf()



        ###################
        # SIMULATOR DISPLAY
        if plotting:
            # Total Figure
            fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
            gs = gridspec.GridSpec(8, 8)

            # Elevator plot settings.
            ax = fig.add_subplot(gs[:8, :8])

            #plt.xlim(-3, 30)
            #ax.set_ylim([-3, 30])
            plt.xlim(np.min(mpc.trajectory[:, 0]) - 20, np.max(mpc.trajectory[:, 0]) + 20)
            plt.ylim(np.min(mpc.trajectory[:, 1]) - 20, np.max(mpc.trajectory[:, 1]) + 20)
            plt.xticks(np.arange(0, 11, step=2))
            plt.yticks(np.arange(0, 11, step=2))
            plt.title('MPC 2D')

            # Time display.
            time_text = ax.text(6, 0.5, '', fontsize=15)

            # Main plot info.
            car_width = 1.0
            patch_car = mpatches.Rectangle((0, 0), car_width, 2.5, fc='k', fill=False)
            patch_goal = mpatches.Rectangle((0, 0), car_width, 2.5, fc='b',
                                            ls='dashdot', fill=False)

            ax.add_patch(patch_car)
            ax.add_patch(patch_goal)
            predict, = ax.plot([], [], 'r--', linewidth=1)

            # Car steering and throttle position.
            telem = [3, 14]
            patch_wheel = mpatches.Circle((telem[0] - 3, telem[1]), 2.2)
            ax.add_patch(patch_wheel)
            wheel_1, = ax.plot([], [], 'k', linewidth=3)
            wheel_2, = ax.plot([], [], 'k', linewidth=3)
            wheel_3, = ax.plot([], [], 'k', linewidth=3)
            throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1] - 2, telem[1] + 2],
                                        'b', linewidth=20, alpha=0.4)
            throttle, = ax.plot([], [], 'k', linewidth=20)
            brake_outline, = ax.plot([telem[0] + 3, telem[0] + 3], [telem[1] - 2, telem[1] + 2],
                                     'b', linewidth=20, alpha=0.2)
            brake, = ax.plot([], [], 'k', linewidth=20)
            throttle_text = ax.text(telem[0], telem[1] - 3, 'Forward', fontsize=15,
                                    horizontalalignment='center')
            brake_text = ax.text(telem[0] + 3, telem[1] - 3, 'Reverse', fontsize=15,
                                 horizontalalignment='center')

            # Trajectory
            x = mpc.trajectory[:, 0]
            y = mpc.trajectory[:, 1]
            angles = mpc.trajectory[:, 2]
            dx = np.cos(angles)
            dy = np.sin(angles)
            ax.quiver(x, y, dx, dy, angles, cmap='hsv', scale=20, width=0.001)

            patch_obs = []
            # Obstacles
            if OBSTACLES:
                for x_obs, y_obs, size_obs in mpc.obstacles:
                    patch_obs.append(mpatches.Circle((x_obs, y_obs), size_obs))
                    ax.add_patch(patch_obs[-1])

            # Shift xy, centered on rear of car to rear left corner of car.
            def car_patch_pos(x, y, psi):
                # return [x,y]
                x_new = x - np.sin(psi) * (car_width / 2)
                y_new = y + np.cos(psi) * (car_width / 2)
                return [x_new, y_new]

            def steering_wheel(wheel_angle):
                wheel_1.set_data([telem[0] - 3, telem[0] - 3 + np.cos(wheel_angle) * 2],
                                 [telem[1], telem[1] + np.sin(wheel_angle) * 2])
                wheel_2.set_data([telem[0] - 3, telem[0] - 3 - np.cos(wheel_angle) * 2],
                                 [telem[1], telem[1] - np.sin(wheel_angle) * 2])
                wheel_3.set_data([telem[0] - 3, telem[0] - 3 + np.sin(wheel_angle) * 2],
                                 [telem[1], telem[1] - np.cos(wheel_angle) * 2])

            def update_plot(time_step):
                # Car.
                patch_car.set_xy(car_patch_pos(state_i[time_step, 0], state_i[time_step, 1], state_i[time_step, 2]))
                patch_car.angle = np.rad2deg(state_i[time_step, 2]) - 90
                # Car wheels
                np.rad2deg(state_i[time_step, 2])
                steering_wheel(u_i[time_step, 1] * 2)
                throttle.set_data([telem[0], telem[0]],
                                  [telem[1] - 2, telem[1] - 2 + max(0, u_i[time_step, 0] / 5 * 4)])
                brake.set_data([telem[0] + 3, telem[0] + 3],
                               [telem[1] - 2, telem[1] - 2 + max(0, -u_i[time_step, 0] / 5 * 4)])

                # Goal.
                patch_goal.set_xy(car_patch_pos(ref_i[time_step][0], ref_i[time_step][1], ref_i[time_step][2]))
                patch_goal.angle = np.rad2deg(ref_i[time_step][2]) - 90

                # Obstacles
                for i, p_obs in enumerate(patch_obs):
                    p_obs.set_center((obs_info[time_step][i, 0], obs_info[time_step][i, 1]))

                # if (num <= 130):
                #    patch_goal.set_xy(car_patch_pos(ref_1[0],ref_1[1],ref_1[2]))
                #    patch_goal.angle = np.rad2deg(ref_1[2])-90

                # print(str(state_i[num,3]))
                predict.set_data(predict_info[time_step][:, 0], predict_info[time_step][:, 1])
                # Timer.
                # time_text.set_text(str(100-t[num]))

                return patch_car, time_text

            print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
            # Animation.
            car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1, len(state_i)), interval=100, repeat=True,
                                              blit=False)
            car_ani.save('mpc-video.mp4')

            plt.show()
        throttles = u_i[:,0]
        steerings = u_i[:,1]
        waypoints_per_frame = predict_info[1:]
        return steerings, throttles, waypoints_per_frame




def generate_complex_trajectory():
    """
    Generates a more complex trajectory within the (0,10) bounds.
    The trajectory includes curves, a loop, and direction changes.
    """

    # 1. Sinusoidal curve (S-curve)
    x1 = np.linspace(0, 12, 50)
    y1 = 2 * np.sin(2 * np.pi * x1 / 5) + x1

    # 2. Circular loop
    t = np.linspace(0, 2 * np.pi, 100)  # Parameter for the curve
    x2 = 6 + 4 * np.cos(t)
    y2 = 4 + 4 * np.sin(t)

    # 3. Zigzag pattern
    x3 = np.linspace(0, 10, 20) * 10
    y3 = 5 + 2 * np.sin(4 * np.pi * (x3 - 7) / 3) * 10

    # Combine all segments
    # x = np.concatenate([x1, x2, x3])
    # y = np.concatenate([y1, y2, y3])
    x, y = x1, y1

    phi = np.arctan2(np.diff(y), np.diff(x))
    phi = np.append(phi, phi[-1])  # Repeat the last angle

    return np.array(list(zip(x, y, phi)))

def make_straight_trajectory():
    x = np.linspace(0, 30, 50)
    y = x * 0 + 2

    phi = np.arctan2(np.diff(y), np.diff(x))
    phi = np.append(phi, phi[-1])  # Repeat the last angle

    return np.array(list(zip(x, y, phi)))



if __name__ == '__main__':
    MPC = ModelPredictiveControl()

    MPC.set_reference_trajectory(make_straight_trajectory())
    MPC.set_lookahead_distance(8.0)
    MPC.set_obstacles(np.array([[5, 2, 1],  # x,y,size
                                [20, 2.4, 4]]))

    MPC.simulate()
