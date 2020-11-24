# End-to-End-Supervised-Learning-Project
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.
=How supervised learning algorithms work==

Given a set of <math>N</math> training examples of the form <math>\{(x_1, y_1), ..., (x_N,\; y_N)\}</math> such that <math>x_i</math> is the [[feature vector]] of the i-th example and <math>y_i</math> is its label (i.e., class), a learning algorithm seeks a function <math>g: X \to Y</math>, where <math>X</math> is the input space and
<math>Y</math> is the output space.  The function <math>g</math> is an element of some space of possible functions <math>G</math>, usually called the ''hypothesis space''.  It is sometimes convenient to
represent <math>g</math> using a scoring function <math>f: X \times Y \to \mathbb{R}</math> such that <math>g</math> is defined as returning the <math>y</math> value that gives the highest score: <math>g(x) = \underset{y}{\arg\max} \; f(x,y)</math>.  Let <math>F</math> denote the space of scoring functions.
