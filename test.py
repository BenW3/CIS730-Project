import matplotlib.pyplot as plt

roundplot  = [0,1,2,3,4,5,6,7]
rewardplot = [[12,13],[14,15],[16,17],[18,19],[20,21],[22,23],[24,25],[25,26]]
learner = []
for i in rewardplot:
    learner.append(i[0])
plt.plot(roundplot,learner, label = 'learner')
# plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.xlabel('round number')
plt.ylabel('reward')
plt.title("Model Reward Over Time")
# plt.legend("learner", "hand coded")
plt.legend()
# plt.show()
plt.savefig('learner_reward')
