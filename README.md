# lightweight-neural-nets

Neural network models for embedded devices

## Model-ff

Model-ff uses Forward-Forward algorithm to train a neural network.

It is composed of one or more `FFCell` objects. Each cell is composed of an input layer and an output layer and can be trained independently.

In a `FFNet` object, the cells are connected in a chain. The output of a cell is the input of the next cell.

In the training phase, for every cell, the forward and backward passes are executed according to the following steps:

### Definitionsr more `FFCell` objects. Each cell is a layer in the network

$\theta \in \mathbb{R}: \text{threshold}$

$\alpha \in \mathbb{R}: \text{learning rate}$

$f \in\mathbb{N}: \text{features}, l\in\mathbb{N}: \text{classes}$

$\mathcal{L} = \{l_1, ..., l_l\}: \text{set of labels}$

$o\in\mathbb{N}: \text{output layer size}$

$x\in\mathbb{R}^f: \text{sample}$

$y\in\mathbb{R}^l: \text{label}$

$\eta:\mathbb{R}^f\times\mathbb{R}^l\rightarrow\mathbb{R}^f: \text{embed function}$

$W\in\mathbb{R}^{f\times o}: \text{weight matrix from input to output layer}$

$a:\mathbb{R}^{f}\rightarrow\mathbb{R}^{o}: \text{activation function (ReLu)}$

$G(z):\mathbb{R}^{o}\rightarrow\mathbb{R}=\sum\limits_{i=1}^{i\le o}(z_i)^2$

$L(z, \bar{z}):\mathbb{R}^{o \times o}\rightarrow\mathbb{R}=\zeta\bigl(\theta - G(z)\bigr) + \zeta\bigl(G(\bar{z})-\theta)$

### Forward pass

Label embedding for the positie and negative samples:

$x :=\eta(x',y)$

$\bar x :=\eta(x,\bar{y}), \bar{y}\in\mathcal{L}|\bar{y}\neq y$

Scalar product of the input and the weight vector:

$h := Wx$

$\bar{h} := W\bar x$

Compute the activations:

$z := a(h)$

$\bar{z} := a(\bar{h})$

### Backward pass

All weights are udated using this formula:

$$
w_{i,j}=w_{i, j}-\alpha\frac{\delta L}{\delta w_{i, j}}=
w_{i, j}-\alpha\Bigl(\frac{\delta L_{pos}}{\delta w_{i, j}} +
\frac{\delta L_{neg}}{\delta w_{i, j}}\Bigr)
$$

Gradient of the positive pass:

$$
\frac{\delta L_{pos}}{\delta w_{i, j}} =
\frac{\delta L_{pos}}{\delta G(z)}
\frac{\delta G(z)}{\delta z_{j}} 
\frac{\delta z_{i}}{\delta w_{i,j}}
$$

Gradient of the loss with respect to the goodness for the positive pass:

$$
\frac{\delta L_{pos}}{\delta G(z)} =
-\sigma\bigl(\theta - G(z)\bigr)
$$

Gradient of the goodness with respect to the j-th activation:

$$
\frac{\delta G(z)}{\delta z_{j}} = 2z_{j}
$$

Gradient of the j-th activation with respect to the weight <i,j>:

$$
\frac{\delta z_{j}}{\delta w_{i,j}} =
\frac{\delta a\bigl(\sum\limits_iw_{i,j} x_i\bigr)}{\delta w_{i,j}} =
\begin{cases}
    x_i & \text{if} \space h_j \ge 0 \\
    0 & \text{if} \space h_j \lt 0
\end{cases}
$$

Putting all the pieces together for the positive pass:

$$
\frac{\delta L_{pos}}{\delta w_{i, j}} =
\begin{cases}
    -\sigma\bigl(\theta - G(z)\bigr)2z_{j}x_i & \text{if} \space h_j \ge 0 \\
    0 & \text{if} \space h_j \lt 0  \\
\end{cases} =
\begin{cases}
    -\sigma\bigl(\theta - G(z)\bigr)2z_{j}x_i & \text{if} \space z_j \ge 0 \\
    0 & \text{if} \space z_j \lt 0  \\
\end{cases} =
-\sigma\bigl(\theta - G(z)\bigr)2z_{j}z_i
$$

> Note that $z_j$ is never smaller than zero as it's the result of ReLu activation.

Similarly for the negative pass we have:

$$
\frac{\delta L_{neg}}{\delta w_{i,j}}=
\frac{\delta L_{neg}}{\delta G(\bar{z})}
\frac{\delta G(\bar{z})}{\delta \bar z_{j}}
\frac{\delta \bar z_{i}}{\delta w_{i,j}}
$$

This is the only piece that differs form the positive:

$$
\frac{\delta L_{neg}}{\delta G(\bar{z})}=
\sigma\bigl(G(\bar{z}) - \theta\bigr)
$$

Gradient for the negative pass:

$$
\frac{\delta L_{neg}}{\delta w_{i,j}}=
\begin{cases}
    \sigma\bigl(G(\bar z)-\theta\bigr)2\bar z_j\bar x_i & \text{if} \space \bar z_j \ge 0\\
    0 & \text{if}\space \bar z_j\lt 0
\end{cases}
=\sigma\bigl(G(\bar{z})-\theta\bigr)2z_{j}\bar{z}_i
$$

Finally we get:

$$
w_{i, j} = w_{i, j} -\alpha\Bigl(-\sigma\bigl(\theta - G(z)\bigr)2z_{j}x_i + \sigma\bigl(G(\bar{z}) - \theta\bigr)2\bar z_{j}\bar{x}_i\Bigr)
$$
