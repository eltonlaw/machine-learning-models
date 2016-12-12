Properties of 'feedforward.py' implementation
- StratifiedKFold cross validation
- Activation function: Hyperbolic tangent(tanh)
- Weights initialized uniformly over [-sqrt(6)/(sqrt((layer[i] length) + (layer[i+1] length))] as described in "Understanding the difficulty of training deep feedforward neural networks"(Bengio,Glorot,2010)
- Biases initialized at 0
- Cross entropy loss function
- One-hot encoded output

feedforward_v2.py
- Biases initialized at 0.1 so that the ReLU's start off active


