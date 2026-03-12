import numpy as np
import copy
import sim
import goal
import rrt
import utils


########## TODO ##########

def plan_cost(plan):
    """
    Compute the cost of a plan as:
        EE XY path length  +  total displacement of non-target objects.

    Path length is approximated as sum of ||(vx, vy)|| * duration for each
    control segment (the EE Cartesian distance covered per segment).

    Non-target object disturbance is the total XY displacement experienced by
    every object except the target (object index 0) across the plan.

    args: plan: A list of rrt.Node representing the trajectory.
    returns: cost: Scalar float.
    """
    n_joints = sim.pandaNumDofs
    n_obj = (plan[0].state["stateVec"].shape[0] - n_joints) // 3

    # EE XY path length
    path_length = sum(
        np.sqrt(n.get_control()[0]**2 + n.get_control()[1]**2) * n.get_control()[3]
        for n in plan[1:]
    )

    # Non-target object disturbance: total XY displacement of objects 1..n_obj-1
    disturbance = 0.0
    for k in range(len(plan) - 1):
        sv0 = plan[k].state["stateVec"]
        sv1 = plan[k + 1].state["stateVec"]
        for i in range(1, n_obj):      # skip index 0 (target object)
            idx = n_joints + 3 * i
            disturbance += np.linalg.norm(sv1[idx:idx+2] - sv0[idx:idx+2])

    return path_length + disturbance


def simulate_controls(start_state, ctrls, pdef):
    """
    Re-simulate a list of controls forward from start_state.

    args: start_state: Initial state dict {"stateID", "stateVec"}.
               ctrls: List of control arrays (each shape (4,)).
                pdef: ProblemDefinition instance.
    returns: (plan, valid):
        plan  - list of rrt.Node (always at least length 1 containing start).
        valid - True iff every propagation step succeeded and every resulting
                state passed is_state_valid.
    """
    nodes = [rrt.Node(start_state)]
    state = start_state
    for ctrl in ctrls:
        next_state, ok = pdef.propagate(state, ctrl)
        if not ok or not pdef.is_state_valid(next_state):
            return nodes, False
        node = rrt.Node(next_state)
        node.set_control(ctrl)
        node.set_parent(nodes[-1])
        nodes.append(node)
        state = next_state
    return nodes, True


class TrajectoryOptimizer:
    """
    Stochastic trajectory optimizer using coordinate-wise random perturbations
    (a form of stochastic hill climbing / STOMP-style optimization).

    At each iteration one randomly chosen control is perturbed with Gaussian
    noise. The entire plan is re-simulated from scratch. The perturbation is
    accepted only if the new plan is (a) fully valid, (b) still satisfies the
    goal, and (c) has strictly lower cost than the current best.
    """

    def __init__(self, pdef, n_iter=500, sigma=0.05):
        """
        args:   pdef: ProblemDefinition instance.
              n_iter: Number of optimization iterations.
               sigma: Std-dev of Gaussian noise added to each control element.
        """
        self.pdef = pdef
        self.n_iter = n_iter
        self.sigma = sigma

    def optimize(self, plan):
        """
        Optimize a plan returned by KinodynamicRRT.solve().

        args: plan: Initial plan (list of rrt.Node), as returned by solve().
        returns:    Optimized plan (list of rrt.Node); returns input unchanged
                    if plan is None or has fewer than 2 nodes.
        """
        if plan is None or len(plan) < 2:
            return plan

        low = self.pdef.bounds_ctrl.low
        high = self.pdef.bounds_ctrl.high
        goal_obj = self.pdef.get_goal()
        start_state = plan[0].state

        best_plan = plan
        best_cost = plan_cost(plan)
        ctrls = [node.get_control().copy() for node in plan[1:]]

        print("Optimization start: cost=%.4f  nodes=%d" % (best_cost, len(plan)))

        for it in range(self.n_iter):
            i = np.random.randint(0, len(ctrls))
            new_ctrls = [c.copy() for c in ctrls]
            new_ctrls[i] = np.clip(
                new_ctrls[i] + np.random.normal(0, self.sigma, 4), low, high
            )

            new_plan, valid = simulate_controls(start_state, new_ctrls, self.pdef)
            if not valid:
                continue
            if goal_obj is not None and not goal_obj.is_satisfied(new_plan[-1].state):
                continue

            new_cost = plan_cost(new_plan)
            if new_cost < best_cost:
                best_plan = new_plan
                best_cost = new_cost
                ctrls = new_ctrls
                print("  iter %4d: cost improved → %.4f" % (it, best_cost))

        print("Optimization done:  cost=%.4f  nodes=%d" % (best_cost, len(best_plan)))
        return best_plan

##########################

