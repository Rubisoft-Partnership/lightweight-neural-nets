# lightweight-neural-nets

Neural network models for embedded devices


## Model-ff

Model-ff uses Forward-Forward algorithm to train a neural network.

### Definitions

$\theta \in \R: \text{threshold}$

$\alpha \in \R: \text{learning rate}$

$f \in\N: \text{features}, l\in\N: \text{classes}$

$\mathcal{L} = \{l_1, ..., l_l\}: \text{set of labels}$

$o\in\N: \text{output layer size}$

$x\in\R^f: \text{sample}$

$y\in\R^l: \text{label}$

$\eta:\R^f\times\R^l\rightarrow\R^f: \text{embed function}$

$W\in\R^{f\times u_0}: \text{weight matrix from input to output layer}$

$a:\R^{f}\rightarrow\R^{o}: \text{activation function (ReLu)}$

$G(z):\R^{o}\rightarrow\R=\sum\limits_{i=1}^{i\le o}(z)^2$

$L(z, \bar{z}):\R^{o \times o}\rightarrow\R=\zeta\bigl(\theta - G(z)\bigr) + \zeta\bigl(G(\bar{z})-\theta)$

### Forward pass

Input to output layer:

$x :=\eta(x',y)$

$\bar x :=\eta(x,\bar{y}), \bar{y}\in\mathcal{L}|\bar{y}\neq y$

$h := Wx$

$\bar{h} := W\bar x$

$z := a(h)$

$\bar{z} := a(\bar{h})$

### Backward pass

Updating weights:

$
w_{i, j} =
w_{i, j} -\alpha\frac{\delta L}{\delta w_{i, j}} =
w_{i, j} -
\alpha\Bigl(\frac{\delta L_{pos}}{\delta w_{i, j}} + 
\frac{\delta L_{neg}}{\delta w_{i, j}}\Bigr)
$

$
\frac{\delta L_{pos}}{\delta w_{i, j}} = 
\frac{\delta L_{pos}}{\delta G(z)}
\frac{\delta G(z)}{\delta z_{j}} 
\frac{\delta z_{i}}{\delta w_{i,j}}
$

$
\frac{\delta L_{pos}}{\delta G(z)} =
-\sigma\bigl(\theta - G(z)\bigr)
$

$
\frac{\delta G(z)}{\delta z_{j}} =
2z_{j}
$

Using ReLu activation function:

$
\frac{\delta z_{j}}{\delta w_{i,j}} =
\frac{\delta a\bigl(\sum\limits_iw_{i,j} x_i\bigr)}{\delta w_{i,j}} =
\begin{cases}
    x_i & \text{if} \space h_j \ge 0 \\
    0 & \text{if} \space h_j \lt 0 \\
\end{cases}
$

$
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
$

> Note that $z_j$ is never smaller than zero as it's the result of ReLu activation.

Similarly for the negative pass we have:

$
\frac{\delta L_{neg}}{\delta w_{i, j}} = 
\frac{\delta L_{neg}}{\delta G(\bar{z})}
\frac{\delta G(\bar{z})}{\delta \bar{z}_{j}}
\frac{\delta \bar{z}_{i}}{\delta w_{i,j}}
$

$
\frac{\delta L_{neg}}{\delta G(\bar{z})} =
\sigma\bigl(G(\bar{z}) - \theta\bigr)
$

$
\frac{\delta L_{neg}}{\delta w_{i, j}} =
\begin{cases}
    \sigma\bigl(G(\bar{z}) - \theta\bigr)2\bar{z}_{j}\bar{x}_i & \text{if} \space \bar{z}_j \ge 0 \\
    0 & \text{if} \space \bar{z}_j \lt 0  \\
\end{cases} =
\sigma\bigl(G(\bar{z}) - \theta\bigr)2z_{j}\bar{z}_i
$

Finally:

$$
w_{i, j} = w_{i, j} -\alpha\Bigl(-\sigma\bigl(\theta - G(z)\bigr)2z_{j}x_i + \sigma\bigl(G(\bar{z}) - \theta\bigr)2\bar z_{j}\bar{x}_i\Bigr)
$$