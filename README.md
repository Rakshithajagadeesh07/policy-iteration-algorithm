# POLICY ITERATION ALGORITHM

## AIM
Implement policy iteration algorithm to find optimal policy by iteratively maximizing the value function.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
# Step 1:
Import required libraries.
# Step 2:
Load the frozen lake environment.
# Step 3:
Define the value evaluation, value improvement and value iteration functions.
# Step 4: 
Run the functions and display the results.

## POLICY IMPROVEMENT FUNCTION
##### Name : Rakshitha J
##### Register Number : 212223240135
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: np.argmax(Q[s])
    return new_pi
```

## POLICY ITERATION FUNCTION
##### Name : Rakshitha J
##### Register Number : 212223240135
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = np.zeros(len(P), dtype=int)
    while True:
        pi_func = lambda s: pi[s]
        V = policy_evaluation(pi_func, P, gamma, theta)
        policy_stable = True
        for s in range(len(P)):
            old_action = pi[s]
            Q = np.zeros(len(P[s]))
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[a] += prob * (reward + gamma * V[next_state] * (not done))
            pi[s] = np.argmax(Q)
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break

    return V, lambda s: pi[s]
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

<img width="707" height="281" alt="image" src="https://github.com/user-attachments/assets/567a1296-e4f3-440c-8710-c4b9c090920e" />

### 2. Policy, Value function and success rate for the Improved Policy

<img width="713" height="358" alt="image" src="https://github.com/user-attachments/assets/b80fa494-1565-4690-939f-d31938d25b01" />

### 3. Policy, Value function and success rate after policy iteration

<img width="687" height="347" alt="image" src="https://github.com/user-attachments/assets/8b1c9139-636d-4852-a4fa-7e10c9a5bc11" />

## RESULT:
Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
