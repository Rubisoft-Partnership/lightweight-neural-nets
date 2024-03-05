# lightweight-neural-nets

Neural network models for embedded devices


## Model-ff

Model-ff uses Forward-Forward algorithm to train a neural network.

### Definitions

$\theta \in \R: \text{threshold}$

$f \in\N: \text{features}, l\in\N: \text{classes}$

$\mathcal{L} = \{l_1, ..., l_l\}: \text{set of labels}$

$u_0\in\N: \text{hidden layer size}, u_1\in\N: \text{output layer size}$

$x\in\R^f: \text{sample}$

$y\in\R^l: \text{label}$

$\eta:\R^f\times\R^l\rightarrow\R^f: \text{embed function}$

$W_0\in\R^{f\times u_0}: \text{weight matrix from input to hidden layer}$

$W_1\in\R^{u_0\times u_1}: \text{weight matrix from hidden to output layer}$

$a_0:\R^{u_0}\rightarrow\R^{u_0}: \text{hidden activation function}$

$a_1:\R^{u_1}\rightarrow\R^{u_1}: \text{output activation function}$

### Forward-Forward

Input to hidden layer:

$x_{pos} :=\eta(x,y)$

$x_{neg} :=\eta(x,\bar{y}), \bar{y}\in\mathcal{L}|\bar{y}\neq y$

$h_0^{pos} := W_0x_{pos}$

$h_0^{neg} := W_0x_{neg}$

$z_0^{pos} := a_0(h_0^{pos})$

$z_0^{neg} := a_0(h_0^{neg})$

Hidden to output layer:

$h_1^{pos} := W_1z_0^{pos}$

$h_1^{neg} := W_1z_0^{neg}$

$z_1^{pos} := a_1(h_1^{pos})$

$z_1^{neg} := a_1(h_1^{neg})$


$G_{pos} = \sum_{i=0}^{i\le u_1}z_{1,i}^{pos}$

$G_{neg} = \sum_{i=0}^{i\le u_1}z_{1,i}^{neg}$

$L_{FF}=\zeta(\theta - G_{pos}) + \zeta(G_{neg}-\theta)$


Weights are updated using the following formula for every Tinn using ReLu:
