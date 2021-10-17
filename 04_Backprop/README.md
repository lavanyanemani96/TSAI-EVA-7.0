# Session 4.0 Backpropagation 

## Part I 
In the excel sheet we create the simple network with two inputs and two outputs with a single hidden layer. After writing all the equations for backpropagation we create in principle a neural-net with simple excelsheet. The sheet has three tabs; Equations, Backpropagation table, Loss curves. 

![Simple perceptron model](./images/simple_perceptron_model.png)

This is the simple network we want to train using only the excel sheet. We use σ sigmoid activation function and L2 loss. The learning rate is represented as η. 

The inputs are : i1, i2 

h1 = w1 x i1 + w2 x i2
h2 = w3 x i1 + w4 x i2

out_h1 = σ(h1)
out_h2 = σ(h2)

o1 = w5 x out_h1 + w6 x out_h2
o2 =  w7 x out_h1 + w8 x out_h2

The outputs are : out_o1, out_o2
out_o1, out_o2 = σ(o1), σ(o2)

The targets are : t1, t2
E1 = 1/2 x (t1 - out_o1)^2
E2= 1/2 x (t2 - out_o2)^2

The total loss : E
E = E1+E2


To perform backpropagation we need to know the partial derivatives of the total error with respect to the weights of the network represented by w. The full derivation is shown in the excel sheet. 

∂E/∂w5 = (out_o1 - t1) x (out_o1(1-out_o1)) x (out_h1)

∂E/∂w6 = (out_o1 - t1) x (out_o1(1-out_o1)) x (out_h2)

∂E/∂w7 = (out_o2 - t2) x (out_o2(1-out_o2)) x (out_h1)

∂E/∂w8 = (out_o2 - t2) x (out_o2(1-out_o2)) x (out_h2)

∂E/∂w1 = ((out_o1-t1) x (out_o1(1-out_o1)) x (w5)) x (h1(1-h1)) x (i1) + ((out_o2-t2) x (out_o2(1-out_o2)) x (w7)) x (h1(1-h1)) x (i1)

∂E/∂w2 = ((out_o1-t1) x (out_o1(1-out_o1)) x (w5)) x (h1(1-h1)) x (i2) + ((out_o2-t2) x (out_o2(1-out_o2)) x (w7)) x (h1(1-h1)) x  (i2)

∂E/∂w3 = ((out_o1-t1) x (out_o1(1-out_o1)) x (w6)) x (h2(1-h2)) x (i1) + ((out_o2-t2) x (out_o2(1-out_o2)) x (w8)) x (h2(1-h2)) x  (i1)

∂E/∂w4 = ((out_o1-t1) x (out_o1(1-out_o1)) x (w6)) x (h2(1-h2)) x (i2) + ((out_o2-t2) x (out_o2(1-out_o2)) x (w8)) x (h2(1-h2)) x (i2)

After each step, the weights are updated as: 
w = w - η x ∂E/∂w

![Backpropagation table screenshot](./images/Backprop_screenshot.png)

We can see that for a larger learning rate η we converge to a lower total loss quicker.

![Loss curve for different η](./images/Loss_curve.png)