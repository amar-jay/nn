# Multi Armed bandit

- Major resource: Richard S. Sutton & Andrew G. Barto

### Stategies

- **$\epsilon$-greedy strategy**: works for stationary reward distribution (on addition of another parameter it affects the rewards distribution significantly)

  ```python
  counts = zeros(no_of_arms) # counts is table of (action/arm, frequency)
  values = zeros(no_of_arms) # table of (action/arm, running average of value estimates)
  def action(epsilon):
    if random.random() < epsilon:
        # Exploration: random action
        return random.randint(len(values))
    else:
        # Exploitation: maximum known action
        return argmax(values)

  def update(action, reward):
      # Update running average of action-reward values
      n = counts[action] += 1
      values[action] = ((n-1)/n) * values[action] + (1/n) * reward
  ```

- **UCB**: Provides theoretical guarantees for regret minimization (this method chooses deter
  ministically but achieve exploration by subtly favoring at each step the actions that have so far received fewer samples)

  ```python
  counts = linspace(0, no_of_arms-1, no_of_arms) # counts is table of (action/arm, frequency) - initial exploration
  values = zeros(no_of_arms) # table of (action/arm, running average of value estimates)

  def action(c): # c- exploration parameter
    # UCB selection
    ucb_values = values + c * sqrt(
        log(sum(counts)) / counts
    )
    return np.argmax(ucb_values)

  def update(action, reward):
      # Update running average of action-reward values
      n = counts[action] += 1
      values[action] = ((n-1)/n) * values[action] + (1/n) * reward
  ```

- **Gradient based algoriths (softmax distribution):** Instead of estimating the value of actions, these algorithms focus on learning action preferences. Uses a softmax distribution to convert preferences into probabilities, favoring more preferred actions while still exploring less-preferred ones.

  ```python
  preferences = zeros(no_of_arms) # H(a) - a table of probabilities of each action
  baseline = 0  # Average reward
  alpha = 0 # step size

  def action(c): # c- exploration parameter
    probabilities = softmax(preferences)

    # make random choice based on softmax preference distribution
    return random.choice(n_of_arms, p=probabilities)

  def update(action, reward):
    probabilities = softmax(preferences)

    # Update baseline (average reward)
    baseline += alpha * (reward - baseline)


    # Update preferences- increase the odds of the action taken and reduce all others binomially
    for a in range(self.n_arms):
        if a == action:
            # Selected action
            self.preferences[a] += self.alpha * (reward - self.baseline) * (1 - probabilities[a])
        else:
            # Non-selected actions
            self.preferences[a] -= self.alpha * (reward - self.baseline) * probabilities[a]

    # Update running average of action-reward values
    n = counts[action] += 1
    values[action] = ((n-1)/n) * values[action] + (1/n) * reward
  ```

- **Optimistic Initialization**: Starts the action-value estimates with high (optimistic) values, encouraging exploration of all actions initially.
  Actions that haven’t been explored will seem more promising due to their optimistic estimates, driving early exploration. Even greedy methods (which typically don’t explore) are forced to explore initially. The optimism may lead to inefficiency if the initialization is poorly calibrated

  ```python
  initial_value = 5.0 # set an inital value
  counts = ones(no_of_arms) * initial_value # counts is table of (action/arm, frequency) - initial exploration
  values = zeros(no_of_arms) # table of (action/arm, running average of value estimates)

  def action(c): # c- exploration parameter

    # take pure greedy action or can implement a combination of epsilon gradient and optimistic initialization- your choice
    return np.argmax(ucb_values)

  def update(action, reward):
      # Update running average of action-reward values
      n = counts[action] += 1
      values[action] = ((n-1)/n) * values[action] + (1/n) * reward
  ```

## Summary

| Method                           | **Approach** | **Exploration Mechanism**    | **Theoretical Foundation** | **Adaptation to Changes**        | **Additional Considerations** |
| -------------------------------- | ------------- | ---------------------------- | -------------------------- | -------------------------------- | ----------------------------- |
| **ε-Greedy**                     | ✅       | Fixed exploration rate       | Limited                    | Poor in later stages             | Can be inefficient            |
| **Upper Confidence Bound (UCB)** | Deterministic | Decreases with more samples  | Theoretically well-founded | Good in stationary environments  | Optimal regret bound          |
| **Gradient Bandit**              | ❌         | Learns relative preferences  | Empirical                  | Good for non-stationary problems | Adapts to reward scaling      |
| **Optimistic Initialization**    | ✅       | Natural exploration early on | Simple but effective       | May need tuning                  | Effective in early stages     |

## Comparison
 A parameter study of the various bandit algorithms presented in the RL book. Each point is the average reward obtained over 1000 steps with a particular algorithm at a particular setting of its parameter.
![image](https://github.com/user-attachments/assets/31bd0eb9-7cdc-4214-9d24-a505dfb12fe2)


