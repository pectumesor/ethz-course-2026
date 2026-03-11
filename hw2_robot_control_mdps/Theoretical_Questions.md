## Exercise 1

# Issues with IK, if Lemniscate (a = width) is increased?

The infinity sign becomes larger, thus some keypoints are at positions unreachable by the robot arm, thus IK does not converge.

# Influence of dt parameter for IK ?

Smaller means velocity integration will be smaller, thus smaller position changes. Converges slowly. Larger values may overshoot and diverge

# Pros and Cons of numerical IK solver vs Analytical Solver ? 

Pros: Can handle singularities with damping, easy to implement, works for almost any robot
Cons: Slower, approximate solution, usually finds one solution instead of all possible.

# Limits of my IK comapred to state-of-the art solvers ?

SotA Solvers such as Mink model the problem as a Quadratic Program, thus adding constraints such as joint limits easy. With our solver this is a limitation.

## Exercise 2

# Which issues arises if we keep increasing Kp ? 

The pull between keypoints get stronger, thus the robot accelerates and moves faster between points.

# Hows does Kd mitigate the effect of Kp ?

It damps the robot speed to prevent it from overshooting keypoints. The effect looks like a jagged, shaky behavior on the arm.

# In what scenarios is a non-zero K_I needed ?

K_I helps to compensate if gravity is too much. Thus in cases where the gravity together with the proportional pull forces would overshooting a downwards movement, it gets dampened and corrected by K_I.

