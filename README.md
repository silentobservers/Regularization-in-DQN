# Regularization-in-DQN
Conference Paper: Regularization in DQN for Parameter-varying Control Learning Tasks

Absract: As an important technique of preventing overfitting, regularization is widely used in supervised learning. However, regularization has not been systematically studied in deep reinforcement learning (deep RL). In this paper, we study the generalization of deep Q-network (DQN), applying with mainstream regularization approaches, including l_1, l_2 and dropout. We pay attention on agent’s performance not only in original environments, but also in parameter-varying environments which are variational but the same task type. Furthermore, the dropout is modified to make it more adaptive to DQN. Then, a new dropout is proposed to speed up the optimization of DQN. Experiments show that regularization helps deep RL achieve better performance in both original and parameter-varying environments when the number of samples is insufficient.

Dependencies:
python 3.x; tensorflow 1.0+; gym; numpy and matplotlib.

Notice: gym environment should be modified to register new environmental ID. 

