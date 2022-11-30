
from matplotlib import pyplot as plt

def update_epsilon(step):
    n_epsilon = 30
    pinit = 1
    pend = 0.15
    r = max((n_epsilon - step) / n_epsilon, 0)
    epsilon = (pinit - pend) * r + pend
    return epsilon


steps = range(0, 40)

epsilons = list(map(update_epsilon, steps))

plt.plot(steps, epsilons)
plt.ylabel('epsilon')
plt.xlabel('step')
plt.savefig('epsilon.pdf')