import time
import os
import local_packages.ExampleBasedRigging as ebr
import local_packages.tools3d_ as t3d

import landmarks.LICT_narrow_r as LICT_narrow
import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
import numba


# # # # # # # Example Based Rigging # # # # # # #           

# Get the landmarks which remain unaffected across different facial expressions.
# This will enable the best alignement between meshes with variable expressions. 
skull_landmaks_target = LICT_narrow.LM[13::]


print('Starting example based rigging for face ')  
start_ebr = time.time()

# Parametrs chosen according to the original paper - refer to it for more details
kappa = 0.1
theta = 2
n_iterations = 3
distribution = np.flip(np.logspace(0.1, 1, n_iterations, endpoint=True))
beta = t3d.normalise (distribution, 0.02, 0.09)
gamma = t3d.normalise (distribution, 80, 1000)


# # # # # # #        
objpath_training_poses = 'data/training_poses/'
objpath_target_neutral = 'data/training_poses/0.obj'
objpath_personalised_bs = 'personalised_blendshapes/' 

# # # # # # # # # # # # # # Create matrices and structures before the main loop
B_0, _, _, _ = t3d.Read(objpath_target_neutral, QuadMode = True)
A_BS_model, B_BS_model, A_0, faces, n, bs_names = ebr.reading_generic_bs('data/generic_blendShapes/')
n_vertices = A_0.shape[1]
tri = faces.T 
num_triangles = tri.shape[0]
S_training_poses, m = ebr.reading_training_data(objpath_training_poses)
# Allign all training poses to neutral pose using 'skull' landmarks      
for i in range (len(S_training_poses)):
    S_training_poses[i] = t3d.align_target_to_source(S_training_poses[i], faces, skull_landmaks_target, B_0, faces, skull_landmaks_target) 

Alpha_star = ebr.blend_shape_weights(A_0, B_0, A_BS_model, S_training_poses)
A_0 = A_0.T
B_0 = B_0.T

A_BS_model = ebr.columnise(A_BS_model)
A_BS_model = np.asarray(A_BS_model)

S_training_poses = ebr.columnise(S_training_poses) 
M_A_star_f = ebr.make_M_A_star_fast(tri, A_0, B_0, A_BS_model)
W_seed_f = ebr.make_W_seed_fast(tri, A_BS_model, kappa, theta)
M_S_minus_M_B_0_f, M_B_0_f, M_S_f = ebr.make_M_S_minus_M_B_0_fast(S_training_poses, B_0, tri)
A_sparse_recon = ebr.make_A_sparse_reconstruction(tri, n_vertices)
  
Alpha_optimum = Alpha_star.copy()

# Main loop
for opt_iteration in range(n_iterations):
    
    print('\nOptimization Step: ' + str(opt_iteration))
    print('Part A:')
    print('Calculating new triangle local frames...')
    start_temp = time.time()
    I = np.eye(3)
    A = sp.kron(Alpha_optimum, I)
    M_B = np.zeros((n*3, 2*num_triangles))
    M_B = ebr.lf_optimisation(num_triangles, A.A, M_S_minus_M_B_0_f, M_B, M_A_star_f, beta, gamma, W_seed_f, opt_iteration, n, m)
    print ("...done in ",(time.time() - start_temp), "sec") 
    print('\nReconstructing vertex positions of unknown blendshapes from M_B...')
    #RECONSTRUCTION
    start = time.time()
    reconstruction = [ebr.recon(M_B, A_sparse_recon, n_vertices, num_triangles, i) for i in range(n)]
    for f in reconstruction:
        idx = f[3]
        B_BS_model[idx][0, :] = f[0]
        B_BS_model[idx][1, :] = f[1]
        B_BS_model[idx][2, :] = f[2]
    
    print ("done in  ",(time.time() - start), "sec") 
    print('\nPart B:')
    print('Optimising blend shape weights...')
    # Step B - Hold blendshapes  constant and solve for optimum weights
    start_temp = time.time()
    B_BS_model = np.asarray(B_BS_model)
    B_All = B_BS_model.reshape(n, n_vertices*3)
    B_All = B_All.T
    for i in range(m): 
        Sj_minus_B0 = (S_training_poses[i]-B_0).T.reshape(n_vertices*3, 1)
        qp_P = 2 * (B_All.T @ B_All + gamma[opt_iteration] * np.identity(n))
        qp_q = -2 * (Sj_minus_B0.T @ B_All+ gamma[opt_iteration] * Alpha_star[i, :]).flatten()
        qp_lb = np.zeros(n)
        qp_ub = np.ones(n)
        Alpha_optimum_temp = solve_qp(P=qp_P, q=qp_q, lb=qp_lb, ub=qp_ub, solver="osqp")      
        Alpha_optimum[i, :] = Alpha_optimum_temp
    print ("...done in ",(time.time() - start_temp), "sec") 
    Exp1 = B_0 + B_BS_model[13].T
    t3d.ShowDeltaGrad(B_0.T,Exp1.T, faces)
end = time.time()
print(end-start)

# save generated bs
if not os.path.exists(objpath_personalised_bs):
    os.makedirs(objpath_personalised_bs)
i =0
for delta in B_BS_model:
    blend_shape = delta + B_0.T
    blend_shape = t3d.align_target_to_source(blend_shape, faces, skull_landmaks_target,B_0.T, faces, skull_landmaks_target)
    t3d.SaveObj(blend_shape, faces, objpath_target_neutral, save_destination = objpath_personalised_bs + bs_names[i] +'.obj' , CM=True)
    i = i+1

print (" All done in ",(time.time() - start_ebr), " sec") 

