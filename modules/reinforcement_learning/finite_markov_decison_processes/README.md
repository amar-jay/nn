# Finite Markov Decision Processes

### Difference betweeen MDPs and Bandit Problems

MDPs are a classical formalization of sequential decision making, where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards. Thus MDPs involve delayed reward and the need to tradeoff immediate and delayed reward.

Whereas in bandit problems we estimated the value $q*(a)$ of each action $a$, in MDPs we estimate the value $q*(s,a)$ of each action $a$ in each state $s$, or we estimate the value $v*(s)$ of each state given optimal action selections.

---

The learner and decision maker is called the **agent**. everything outside the agent, is called the **environment**.

**_ AGENT - ENVIRONMENT IMAGE_**

#### Expected Return

In general, we seek to **maximize the expected return**, where the return, denoted $G_t$, is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

$$
 G_t =R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T
$$

where T - final time step, where each interaction is called an _episode_.

#### Discount Factor

Analogous to `time value of money` in finance, Expected return is discounted across time/state such that each return has a higher value than that of previous state. This concept is called _Discounting_.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$


where $\gamma$ is the discount factor, $0 < \gamma < 1$, which represents the rate at which future rewards are discounted. This equation can be simplified using the formula for an infinite geometric series:

$$ 
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = \frac{R_{t+1}}{1 - \gamma} 
$$

#### state-value function

The value of a state s under a policy , denoted $v_\pi(s)$, is the expected return when starting in s and following $\pi$ thereafter. For MDPs, we can define $v_\pi(s)$ formally by

$$ 
v(s) = \mathbb{E}[G_t | S_t = s] = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s\right] \quad \text{for all } s \in S 
$$



For the action-value function, we define it as the expected return starting from state $s$, taking the action $a$, and thereafter following policy $\pi$:

$$
q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a] = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a\right]
$$

where $\mathbb{E}[\cdot]$ denotes the expected value of a random variable given that the agent follows policy $\pi$, and $t$ is any time step.

##### Difference between state-value function and action-value function

| -              | **State-Value Function (V(s))**          | **Action-Value Function (Q(s, a))**                            |
| -------------- | ---------------------------------------- | -------------------------------------------------------------- |
| **Definition** | Expected return starting from state $s$. | Expected return starting from state $s$ and taking action $a$. |
| **Input**      | State $s$.                               | State $s$ and action $a$.                                      |
