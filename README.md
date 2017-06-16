
# Introduction
A little experiment to assess lineal and polynomial decision regions in logistic regression. The code is written in Julia 0.5 and plots are made with [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl)

A 2D dataset of points is randomly generated and logistic regression is used to distinguish the blue point from the red ones.
The blue points are determined to be those having both features *x1* and *x2* over 0.5, so the expected decision boundary should be a square with a vertex on (0.5, 0.5).

Logistic regression produces a decision boundary that is linear, which may or may not be adequate to this problem depending on the randomly generated data. As more points are closer to the center of the area (0.5, 0.5) more harder it would be for the model to correctly classify those points.

The experiment here is to apply polynomial expansion to the feature space in order to have non-linear decision boundaries. The main idea is to produce a higher dimensional feature space in which logistic regression is applied.

The idea of using products of features in order to generate more features and obtain better models is discussed in Andrew Ng's course on Machine Learning. Here I use that idea to expand a feature set to an arbitrary polynomial degree.

# Polynomial expansion
The goal is to generate the monomials of the polynomial ignoring the coefficients, because the coefficients will be represented by the weights of the model. This combinatory problem can be tackled in the following way:

Given the feature set *\[x1, x2\]*, the linear model applied to it has the form *w1\*x1 + w2\*x2 + b*. The quadratic expansion of that feature space would be *\[x1^2, x1x2, x2^2\]*. By concatenating the original space with the new one, a linear model can be applied to this higher dimensional feature space: *w1\*x1 + w2\*x2 + w3\*x1^2 + w4\*x1x2 + w5\*x2^2 + b*. This model represents a non-linear decision boundary in the original 2D space.

The expansion to the nth degree is performed by combining each feature with the expansion to the (n-1)th degree of itself and the following features. The combination consist of multiplying the feature by each column in the expansion. For example:

    expand([x1,x2], 2) = 
    = x1 * expand([x1,x2], 1) ++ x2 * expand([x2], 1) =
    = x1 * [x1,x2] ++ x2 * [x2] =
    = [x1^2, x1x2, x2^2]

where *++* means concatenation and _*_ means combination.

# Model application and plotting
Logistic regression is fitted by gradient descent, using both a fixed learning rate and maximum iteration count. No regularization is being applied. The cost function used is cross entropy. The same hyper-parameters are applied to the three models evaluated in this experiment: linear fit, quadratic fit, and cubic fit. The accuracy of each model is also reported.

Decision boundaries are plotted by sweeping through the 2D space, applying the corresponding expansion to each point, and obtaining the prediction from the model.

