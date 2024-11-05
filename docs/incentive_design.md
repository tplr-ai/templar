# Incentive Design

## Introduction

This document provides a detailed explanation of the incentive mechanism employed in the protocol, focusing on how miners and validators interact to collaboratively train a model while ensuring honest participation. The protocol is designed to encourage miners to contribute genuine model updates that improve the overall model performance, with validators evaluating these contributions to assign rewards accordingly.

## Overview of the Protocol

The protocol involves two main participants:

- **Miners**: Nodes responsible for training the model on assigned data subsets and uploading their updates.
- **Validators**: Nodes that evaluate the miners' contributions by comparing the uploaded updates with locally computed gradients.

The coordination between miners and validators ensures that only beneficial updates are integrated into the model, and participants are incentivized to act honestly.

## Miners

### Operations

1. **Model Synchronization**:
   - Miners start by synchronizing their model with the latest global state.
   - They download **state slices** from other miners, ensuring their model parameters are up-to-date.

2. **Data Acquisition**:
   - Each miner receives a specific subset of the dataset (pages) for the current window.
   - The data assignment is deterministic, based on a seed derived from the window number and the miner's UID.

3. **Training**:
   - Miners train their local model on the assigned data, performing gradient updates.
   - The training is conducted for a specific number of steps, determined by the batch size and sequence length.

4. **Delta Computation and Upload**:
   - After training, miners compute the **delta** (the difference between the updated and initial model parameters).
   - These deltas are compressed and uploaded to a designated S3 bucket associated with the miner.

5. **Window Progression**:
   - Miners proceed to the next window and repeat the process, ensuring continuous contribution to model training.

### Formal Definitions

- **Model Parameters**: $\theta^t$ at window $t$.
- **Updated Parameters**: $\theta^{t+1}$ after training.
- **Delta**: $\delta^t = \theta^{t+1} - \theta^t$.

## Validators

### Operations

1. **Model Synchronization**:
   - Validators synchronize their model to match the state at the beginning of the evaluation window.
   - They download and apply **state slices** to ensure consistency.

2. **Delta Acquisition**:
   - Validators download the deltas uploaded by miners for the evaluation window.

3. **Local Gradient Computation**:
   - For each miner, the validator computes the local gradient $\hat{g}_i$ on the same data subset the miner was assigned.

4. **Scoring**:
   - Validators calculate the **cosine similarity** between each miner's delta $\delta_i$ and the validator's local gradient $\hat{g}_i$.
   - This similarity score reflects how well the miner's update aligns with the true gradient.

5. **Reward Assignment**:
   - Based on the similarity scores, validators assign weights (rewards) to miners.
   - These weights are normalized and set on the chain to influence the global model updates.

### Formal Definitions

- **Local Gradient**: $\hat{g}_i$, the gradient computed by the validator for miner $i$.
- **Miner's Delta**: $\delta_i$, uploaded by miner $i$.
- **Cosine Similarity**:

$$
s_i = \frac{\delta_i \cdot \hat{g}_i}{|\delta_i| |\hat{g}_i|}
$$

- **Assigned Weight**: $w_i$, proportional to $s_i$.

## Incentive Mechanism

### Objective

The incentive mechanism aims to:

- **Encourage Honest Participation**: Miners are motivated to perform genuine training and provide truthful updates.
- **Promote Model Improvement**: Only updates that positively contribute to the model are rewarded.
- **Discourage Malicious Behavior**: Malicious or random updates yield low or negative rewards, making dishonest behavior unprofitable.

### Detailed Explanation

#### Cosine Similarity Scoring

For each miner $i$:

1. **Compute Cosine Similarity**:

$$
s_i = \frac{\delta_i \cdot \hat{g}_i}{|\delta_i| |\hat{g}_i|}
$$

   - Measures the alignment between the miner's update and the true gradient.

2. **Interpretation of $s_i$**:
   - **$s_i > 0$**: Miner’s update is in the same general direction as the true gradient, contributing positively.
   - **$s_i \approx 0$**: Miner’s update is orthogonal to the true gradient, offering little to no benefit.
   - **$s_i < 0$**: Miner’s update opposes the true gradient, potentially harmful.

#### Weight Assignment

1. **Initial Weight Calculation**:

    Assign initial weights proportional to the similarity scores:

$$
w_i\prime = \max(s_i, 0)
$$

2. **Normalization**:

    Normalize the weights to ensure they sum up to 1:

$$
w_i = \frac{w_i'}{\sum_j w_j'}
$$

   This ensures the distribution of rewards is fair and proportional to positive contributions.

#### Reward Distribution

- **Total Reward Pool**: Determined by network parameters and available tokens.
- **Individual Reward**:

$$
R_i = R_{\text{total}} \times w_i
$$

- Miners receive rewards based on their normalized weights.

### Formal Guarantees

1. **Alignment Incentive**:
   - Miners maximize rewards by aligning their updates with the true gradient.
   - Honest training naturally leads to higher cosine similarity scores.

2. **Robustness Against Malicious Behavior**:
   - Malicious updates yield low or negative similarity scores.
   - Negative scores are set to zero in weight assignment, nullifying rewards for harmful contributions.

3. **Fair Reward Distribution**:
   - Normalization ensures that rewards are proportionally distributed among positive contributors.
   - Miners contributing more effectively to the model receive higher rewards.

4. **Convergence Assurance**:
   - By aggregating updates that align with the true gradients, the model is guaranteed to improve or converge under standard optimization assumptions.

5. **Data Subset Specialization**:
   - Miners focus on specific data subsets, promoting specialization and efficient coverage of the entire dataset.

6. **Sybil Resistance**:
   - Rewards are tied to the quality of contributions, not the number of identities.
   - Multiple identities with low-quality updates do not gain an advantage.

## Formal Analysis

### Miner Utility Maximization

Each miner seeks to maximize their expected reward \( R_i \):

$$
\max_{\delta_i} \quad R_i = R_{\text{total}} \times \frac{\max(s_i, 0)}{\sum_j \max(s_j, 0)}
$$

Subject to:

- **Update Constraint**: $\delta_i = \theta^{t+1}_i - \theta^t$
- **Training Dynamics**: $\theta^{t+1}_i = \theta^t - \eta \hat{g}_i$ (using learning rate $\eta$)

The miner's optimal strategy is to set $\( \delta_i \)$ proportional to $\( -\hat{g}_i \)$, aligning with the negative gradient descent direction.

### Validator Consistency

Validators ensure that:

- The evaluation is done fairly using consistent data subsets.
- The local gradients $\( \hat{g}_i \)$ are computed accurately.

### Security Considerations

1. **Data Integrity**:
   - Data subsets are determined by deterministic functions, preventing miners from choosing favorable data.

2. **Parameter Confidentiality**:
   - Only parameter slices are shared, and the indices are not revealed in advance, reducing the risk of targeted attacks.

3. **Resistance to Free Riders**:
   - Miners not contributing meaningful updates do not receive rewards.
   - Validators' scoring mechanism filters out non-beneficial contributions.

## Conclusion

The protocol's incentive mechanism effectively encourages miners to contribute authentic, high-quality updates to the global model. By tying rewards to the cosine similarity between miners' updates and validators' local gradients, the system ensures that only beneficial contributions are rewarded. Formal guarantees provide robustness against malicious actors and promote the overall improvement of the model through collaborative effort.

The careful design of data assignment, update evaluation, and reward distribution creates a self-regulating ecosystem where honest participation is the most profitable strategy for miners, aligning individual incentives with the collective goal of training an effective model.
