#Notes

Properties of 'backprop.py' implementation
- StratifiedKFold: Prevents overfitting, more accurate, uses data more effectively. 
- Activation function: Hyperbolic tangent(tanh)
- Weights initialized uniformly over [-sqrt(6)/(sqrt((layer[i] length) + (layer[i+1] length))] as described in "Understanding the difficulty of training deep feedforward neural networks"(Bengio,Glorot,2010)
- Biases initialized at 0
- Find a different cost function
- Output is the probability of being each number

#References

Q Learning
- http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf
- http://mnemstudio.org/path-finding-q-learning-tutorial.htm

Optical Character Recognition
- https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=qsJmtkMAAAAJ&citation_for_view=qsJmtkMAAAAJ:9yKSN-GCB0IC
