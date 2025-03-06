from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation

robot, G = load_symmetric_system(robot_name='mini_cheetah')

bullet_client = configure_bullet_simulation(gui=True)         # Start pybullet simulation
robot.configure_bullet_simulation(bullet_client, world=None)  # Load robot in pybullet environment
# Get joint space position and velocity coordinates  (q_js, v_js) | q_js ∈ Qjs, dq_js ∈ TqQjs
_, v_js = robot.get_joint_space_state()

# Get the group representation on the space of joint space generalized velocity coordinates 
rep_TqQ_js = G.representations['TqQ_js']
for g in G.elements:
  # Transform the observations 
  breakpoint()
  g_v_js = rep_TqQ_js(g) @ v_js   # rep_TqQ_js(g) ∈ R^12x12