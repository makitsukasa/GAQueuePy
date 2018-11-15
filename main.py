import numpy as np
from jggsystem import JGGSystem
from problem.frontier.sphere import sphere
from problem.frontier.ellipsoid import ellipsoid

n = 20
system = JGGSystem(ellipsoid, n, 6 * n, n + 1, 6 * n)

system.step(33800)

print(system.get_best_evaluation_value());
