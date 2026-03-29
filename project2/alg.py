import numpy as np
import itertools as it
import scipy.spatial
import utils
import time


########## Task 1: Primitive Wrenches ##########

def primitive_wrenches(mesh, grasp, mu=0.2, n_edges=8):
    """
    Find the primitive wrenches for each contact of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
    returns:   W: The primitive wrenches.
                  Type: numpy.ndarray of shape (len(grasp) * n_edges, 6)
    """
    ########## TODO ##########
    W = np.zeros(shape=(len(grasp) * n_edges, 6))

    # Get contact points (centroids of the grasped triangles)
    con_pts = utils.get_centroid_of_triangles(mesh, grasp)
    cm = mesh.center_mass

    for i, tr_id in enumerate(grasp):
        # Outward face normal at the contact
        n = mesh.face_normals[tr_id]

        # Build two tangent vectors perpendicular to n
        v = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, v)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 = t2 / np.linalg.norm(t2)

        # Moment arm from center of mass to contact point
        r = con_pts[i] - cm

        for j in range(n_edges):
            theta = 2.0 * np.pi * j / n_edges
            # Primitive force: normal component 1, tangential component mu
            f = n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
            # Torque = r x f
            tau = np.cross(r, f)
            W[i * n_edges + j, :3] = f
            W[i * n_edges + j, 3:] = tau

    ##########################
    return W


########## Task 2: Grasp Quality Evaluation ##########

def eval_Q(mesh, grasp, mu=0.2, n_edges=8, lmbd=1.0):
    """
    Evaluate the L1 quality of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
            lmbd: The scale of torque magnitude.
                  (default: 1.0)
    returns:   Q: The L1 quality score of the given grasp.
    """
    ########## TODO ##########
    Q = -np.inf

    # Get primitive wrenches
    W = primitive_wrenches(mesh, grasp, mu=mu, n_edges=n_edges)

    # Scale the torque components by sqrt(lmbd)
    W[:, 3:6] *= np.sqrt(lmbd)

    # Compute the convex hull of the primitive wrenches
    try:
        hull = scipy.spatial.ConvexHull(W)
    except scipy.spatial.QhullError:
        return Q

    # The L1 quality is the minimum signed distance from the origin
    # to the hyperplanes of the convex hull.
    # hull.equations has rows [n1,...,n6, offset] where n·x + offset <= 0 inside.
    # Signed distance from origin = -offset
    offsets = hull.equations[:, -1]
    Q = np.min(-offsets)

    ##########################
    return Q


########## Task 3: Stable Grasp Sampling ##########

def sample_stable_grasp(mesh, thresh=0.0):
    """
    Sample a stable grasp such that its L1 quality is larger than a threshold.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
            thresh: The threshold for stable grasp.
                    (default: 0.0)
    returns: grasp: The stable grasp represented by the indices of triangles.
                    Type: list of int
                 Q: The L1 quality score of the sampled grasp, 
                    expected to be larger than thresh.
    """
    ########## TODO ##########
    grasp = None
    Q = -np.inf
    n_faces = len(mesh.faces)
    
    grasp_list = []
    Q_list = []

    while Q <= thresh:
        # Randomly sample 3 distinct face indices
        grasp = list(np.random.choice(n_faces, size=3, replace=False))
        Q = eval_Q(mesh, grasp, mu=1)
        grasp_list.append(grasp)
        Q_list.append(Q)

    ##########################
    return grasp, Q


########## Task 4: Grasp Optimization ##########

def find_neighbors(mesh, tr_id, eta=1):
    """
    Find the eta-order neighbor faces (triangles) of tr_id on the mesh model.
    args:       mesh: The object mesh model.
                      Type: trimesh.base.Trimesh
               tr_id: The index of the query face (triangle).
                      Type: int
                 eta: The maximum order of the neighbor faces:
                      Type: int
    returns: nbr_ids: The list of the indices of the neighbor faces.
                      Type: list of int
    """
    ########## TODO ##########
    nbr_ids = []

    # BFS up to eta hops using vertex-sharing adjacency
    current = {tr_id}
    visited = {tr_id}

    for _ in range(eta):
        next_level = set()
        for fid in current:
            # Get vertices of this face
            verts = mesh.faces[fid]
            for v in verts:
                # Get all faces incident to this vertex
                neighbors = mesh.vertex_faces[v]
                for nf in neighbors:
                    if nf == -1:
                        break
                    if nf not in visited:
                        next_level.add(nf)
                        visited.add(nf)
        current = next_level

    # Return all visited faces except tr_id itself
    visited.discard(tr_id)
    nbr_ids = list(visited)

    ##########################
    return nbr_ids

def local_optimal(mesh, grasp):
    """
    Find the optimal neighbor grasp of the given grasp.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
             grasp: The indices of the mesh triangles being contacted.
                    Type: list of int
    returns: G_opt: The optimal neighbor grasp with the highest quality.
                    Type: list of int
             Q_max: The L1 quality score of G_opt.
    """
    ########## TODO ##########
    G_opt = None
    Q_max = -np.inf

    # For each contact, find its neighbor faces (including itself)
    neighbor_lists = []
    for tr_id in grasp:
        nbrs = find_neighbors(mesh, tr_id, eta=1)
        nbrs.append(tr_id)  # include the current face
        neighbor_lists.append(nbrs)

    # Evaluate all combinations of neighbor faces
    for combo in it.product(*neighbor_lists):
        candidate = list(combo)
        Q = eval_Q(mesh, candidate)
        if Q > Q_max:
            Q_max = Q
            G_opt = candidate

    ##########################
    return G_opt, Q_max

def optimize_grasp(mesh, grasp):
    """
    Optimize the given grasp and return the trajectory.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
            grasp: The indices of the mesh triangles being contacted.
                   Type: list of int
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int)
    """
    traj = []
    ########## TODO ##########

    current_grasp = list(grasp)
    current_Q = eval_Q(mesh, current_grasp)
    traj.append(current_grasp)

    while True:
        G_opt, Q_max = local_optimal(mesh, current_grasp)
        if Q_max <= current_Q:
            break
        current_grasp = G_opt
        current_Q = Q_max
        traj.append(current_grasp)

    ##########################
    return traj


########## Task 5: Grasp Optimization with Reachability ##########

def optimize_reachable_grasp(mesh, r=0.5):
    """
    Sample a reachable grasp and optimize it.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
                r: The reachability measure. (default: 0.5)
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int) 
    """
    traj = []
    ########## TODO ##########

    def is_reachable(grasp, r):
        """Check if a grasp satisfies the reachability constraint."""
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        centroid = np.mean(con_pts, axis=0)
        avg_dist = np.mean(np.linalg.norm(con_pts - centroid, axis=1))
        return avg_dist < r

    # Sample a reachable stable grasp
    n_faces = len(mesh.faces)
    while True:
        grasp = list(np.random.choice(n_faces, size=3, replace=False))
        if is_reachable(grasp, r) and eval_Q(mesh, grasp) > 0.0:
            break

    current_grasp = list(grasp)
    current_Q = eval_Q(mesh, current_grasp)
    traj.append(current_grasp)

    # Optimize while respecting reachability
    while True:
        # Find neighbor faces for each contact (including itself)
        neighbor_lists = []
        for tr_id in current_grasp:
            nbrs = find_neighbors(mesh, tr_id, eta=1)
            nbrs.append(tr_id)
            neighbor_lists.append(nbrs)

        # Search over all neighbor combinations for best reachable grasp
        G_opt = None
        Q_max = -np.inf
        for combo in it.product(*neighbor_lists):
            candidate = list(combo)
            if not is_reachable(candidate, r):
                continue
            Q = eval_Q(mesh, candidate)
            if Q > Q_max:
                Q_max = Q
                G_opt = candidate

        if G_opt is None or Q_max <= current_Q:
            break
        current_grasp = G_opt
        current_Q = Q_max
        traj.append(current_grasp)

    ##########################
    return traj
