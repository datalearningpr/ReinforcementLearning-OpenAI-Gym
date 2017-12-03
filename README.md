
# Reinforcement Learning - OpenAI Gym

Play OpenAI Gym games with different reinforcement learning methods.  

There are plenty of tutorials online introducing implementation of different RL methods, most of them are using TensorFlow. It will help your learning and understanding if you can reproduce them via different tools(Keras).  

**Hope this reporsitroy will help people who wants to reproduce the TensorFlow version of different RL methods via Keras.**

### Prerequisites

Keras,
TensorFlow,
Numpy,
Gym

## Different RL methods

* QLearning  
Given a state, choose an action, you should get the Q-value of the action you picked for this given state. Q-value can be considered as the assessment of how good the action is for this state, the higher the better.  
The nature of QLearning is using the AVERAGE of Q(s, a) to estimate the Q(s, a). If you collect all the data and then you can calculate the average, but it is possible for QLearning to update when you get the data each step. The idea is similar to dynamic programming, it is called Bellman equation:  
```
	Q(s, a) = r + gamma * (max(Q(s', a')))
```
By getting all the possible Q(s, a) of the game, the game can be solved.


* **DQN**  
QLearning will be difficult or impossible in some cases. Say there are millions of combinations of states and actions, or the state is continuous. Then we introduce the deep learning to solve this problem. Deep learning is good at mapping complex input to ideal output.  
How to train the NET? In DQN, we have 2 NETs, q_net and target_net. We use the Bellman equation to train the NET as well, target_net will get the max(Q(s', a')) and then the ideal Q(s, a) can be obtained via Bellman equation. Model can be trained.
DQN uses several tricks to accelerate the converge: 2 models, copy weights from q_net to target_net after certain steps. Store training sample and pick randomly to break the correlation of the samples.


* **Double DQN**
Double DQN is another trick to help the converge of training the NET. Make 2 NETs interact with each other. Since DQN already has 2 NETs, q_net and target_net. Just a small modification will make it work:  
use the q_net to get the action and target_net to get the max Q

* **Prioritized Experience Replay DQN  **
Another trick to help the converge of DQN. For the samples we collect. We assign weights differently to them based on difference between ideal Q(s, a) and estimated Q(s, a), e.g. TD error.
This trick also helps the games that samples are very imbalanced. say winning move is the final move and only happens in the last moment.


* **DuelingDQN**  
DuelingDQN separates the Q to be Q of S plus Q of A.
This improves the DQN in a situation that with certain state, no matter what action you choose, the result is no big difference.


* **Policy Gradient**  
QLearning/DQN outputs Value, and then based on the value, generates the action. Policy Gradient directly outputs the probability of each action, then picks the action based on the probability.  
However, training the NET becomes a problem since there is no loss function, there is no way you can know the "correct" probability and then get the loss.  
In order to train the NET, we use the reward as indicator, if an actions gives a good reward, we tend to increase the probability. Therefore, we have the loss function as:
```
	- log(P(s, a)) * V
```
V is the reward, -log(P(s, a)) is the negative log probability, the lower the probability, the higher the loss(the more surprised we feel about this action). V is the discounted reward of the whole series.
Then, we can train the Policy gradient network.


* **ActorCritic  **
As mentioned above, Policy cannot be updated each step the game is playing. Actor Critic helps with this case. It combines the policy gradient and function approximation.
Actor - policy gradient updates as what we described in Policy Gradient.
Critic - maps state to a value Q, updates via TD error, this TD error will feedback to Actor to act as V so that Actor can be trained each step rather than the end of a series of actions.


* **A3C  **
Since A3C are using gradient applying logic, it is best to use all tensorflow, Keras is not suitable here. Therefore, it is not implemented here, for tensorflow version, there are lots tutorials online.  
In A3C, multi-threading are used to make different agent to play their own games. The father agent only contains model, does not update or train by himself, all happened in child thread.  
Each child agent plays its own game, train their own model and get the gradients, when updates father model, it simply applies the gradient to the father model.


* **Deep Deterministic Policy Gradient(DDPG)  **
Policy gradient outputs distribution, DDPG goes a step further, directly outputs the action. In order to do that, the two NETs used are different and more complex.  
the actor NET has two nets, eval and target, like DQN. the critic NET the same.  
The critic NET structure is very different now, takes state and action as input, then output the Q, since the input action is output of actor, the training becomes, get the gradient of Q over action then get the gradient of action over state(weights of net), then we can have gradient the actor NET should move, so that actor NET can be train as well. critic NET training is still using TD error.


* **Proximal Policy Optimization(PPO)  **
PPO is based on ActorCritic, it helps with the difficulty of Policy Gradient in choosing the learning rate(learning step size).
The training of Actor is different, it still has 2 nets(new and old), when updating, we get the ratio of probability of new net and old net, then using clipping method to control the ratio not going too far below or beyond(maintain in an acceptable range). so that the model can have a good balance of being able to converge and converging fast enough.


* **Distributed Proximal Policy Optimization(DPPO)**
This version DDPO is simply use different thread playing the game and collecting data, training and updating still happen in one father thread of model. Simply making advantage of more agent collecting data rather than one agent.  
A3C can use the same logic as well.

## License

GPL

## Acknowledgments

Many thanks to different online tutorials of teaching RL methods, especially Mofan Zhou's GitHub RL projects.
