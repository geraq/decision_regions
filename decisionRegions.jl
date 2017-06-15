using PyPlot

M = 100
X = rand(M, 2)
y = (X[:,1] .> 0.5) & (X[:,2] .> 0.5)
y = map(b -> if b 1.0 else 0.0 end, y)
plt[:hold](true)
plt[:ion]() # otherwise waits for each image to be closed before going on.


function plotData(X::Matrix{Float64}, y::Vector{Float64})
    plot(X[y .== 1, 1], X[y .== 1, 2], "b*")
    plot(X[y .== 0, 1], X[y .== 0, 2], "r*")
end

function logisticRegression(X::Matrix{Float64}, y::Vector{Float64}, alpha::Float64, maxIter::Int)
    (nExamples, nAttrs) = size(X) # M x N
    X2 = [ones(nExamples, 1) X] # M x (N+1)
    W = rand(nAttrs + 1, 1) # N+1 x 1       
    for i in 1:maxIter        
        O = apply(W, X) # M x 1
        #gradient = X2' * ((O .* (1 - O)) .* (O - y)) # (N+1) x M * (M x 1 .* M x 1) #MSE
        gradient = X2' * (O - y) # (N+1) x M * (M x 1 .* M x 1) #Cross-entropy
        W = W - alpha * gradient
    end
    return W
end

function apply(W::Matrix{Float64}, X::Matrix{Float64})
    (nExamples, nAttrs) = size(X) # M x N
    X2 = [ones(nExamples, 1) X] # M x (N+1)
    O = X2 * W # M x (N+1) * (N+1) x 1 --> M x 1
    return 1 ./ (1 + exp(-O)) # M x 1
end

function plotPerceptron(W::Matrix{Float64})
    f(x) = -(W[1] + W[2] * x) / W[3]
    x = 0:0.01:1
    y = map(f, x)
    plot(x, y, "g-")
end

function plotRegions(f::Function)
    c1 = Matrix{Float64}(0,2)
    c2 = Matrix{Float64}(0,2)
    for x1 in 0:0.02:1
        for x2 in 0:0.02:1
            label = f([x1, x2])
            if label > 0.5
                c1 = [c1 ; [x1 x2]]
            else
                c2 = [c2 ; [x1 x2]]
            end
        end
    end
    plot(c1[:, 1], c1[:, 2], "b.", ms=2)
    plot(c2[:, 1], c2[:, 2], "r.", ms=2)
end

#quadratic expansion, given [x1,x2], this produces [x1, x2, x1^2, x1*x2, x^2]#
function expand(X::Matrix{Float64})
    (nExamples, nAttrs) = size(X)
    X2 = zeros(nExamples, sum(1:nAttrs))    
    i = 1
    for x1 in 1:nAttrs
        for x2 in x1:nAttrs
            X2[:, i] = X[:, x1] .* X[:, x2]
            i += 1
        end
    end
    return [X X2]
end

#=A generalization of the previous function. Polynomial expansion to the nth degree (no coefficients)
Each column distributes over the expansion of itself and the remaining columns, to the (n-1)th degree.
i.e., e([x1, x2], 2) = [x1*e([x1, x2], 1) x2*e([x2], 1)] =
[x1 * [x1, x2] x2*x2] = [x1^2 x1*x2 x2^2]
=#
function expand(X::Matrix{Float64}, n::Int)
    if (n <= 1)
        return X
    else        
        (nExamples, nAttrs) = size(X)
        Xnew = Matrix{Float64}(nExamples, 0)
        for i in 1:nAttrs
            X2 = expand(X[:, i:end], n - 1)
            for j in 1:size(X2, 2)
                Xnew = [Xnew (X[:, i] .* X2[:, j])]
            end
        end
        return Xnew
    end
end

#concatenates the results of the expansions from 1 up to N.#
function expandSet(X::Matrix{Float64}, N::Int)
    (nExamples, nAttrs) = size(X)
    sets = map(n -> expand(X, n), 1:N)
    return foldl((R, D) -> [R D], Matrix{Float64}(nExamples, 0), sets)
end

alpha = 0.1
maxIter = 200

W = logisticRegression(X, y, alpha, maxIter)
O = apply(W, X)
P = map(x -> x > 0.5 ? 1 : 0, O)
acc = mean(P .== y)
println("linear fit accuracy = $(acc)")
#plotPerceptron(W)
plotRegions(x -> apply(W, x')[1])
plotData(X, y)
#plt[:show]()

X2 = expand(X)
W2 = logisticRegression(X2, y, alpha, maxIter)
O2 = apply(W2, X2)
P2 = map(x -> x > 0.5 ? 1 : 0, O2)
acc2 = mean(P2 .== y)
println("quadratic fit accuracy = $(acc2)")
figure()
#plotRegions(x -> apply(W2, [x[1] x[2] x[1].^2 x[1].*x[2] x[2].^2])[1])
plotRegions(x -> apply(W2, expandSet(x', 2))[1])
plotData(X, y)
#plt[:show]()

X3 = expandSet(X, 3)
W3 = logisticRegression(X3, y, alpha, maxIter)
O3 = apply(W3, X3)
P3 = map(x -> x > 0.5 ? 1 : 0, O3)
acc3 = mean(P3 .== y)
println("cubic fit accuracy = $(acc3)")
figure()
#plotRegions(x -> apply(W3, [x[1] x[2] x[1].^2 x[1].*x[2] x[2].^2 x[1].^3 x[1].^2 .* x[2] x[1].*x[2].^2 x[2].^3])[1])
plotRegions(x -> apply(W3, expandSet(x',3))[1])
plotData(X, y)
plt[:show]()
readline()

