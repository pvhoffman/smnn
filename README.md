Softmax Neural Network implementation in C++ vectorized using the Armadillo C++ linear algebra library.

Gives P(y=j|x;t) where t is the theta parameter

Implementation from this document:
http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression

Softmax is prefered over binary classification for mutually exclusive classes and is also prefered for use in autoencoders.

Hidden units will activate on logistical regression.

The predict member function returns the probability for all cases, e.g. P(y=j|x;t) for all possible 'quantized' y

Usage example:
    smnn::layerdesc_t ind;
    ind.push_back(2);
    ind.push_back(2);
    ind.push_back(2);

    smnn::SMNeuralNet nn(ind, 0.2);

    arma::mat X = arma::zeros(5000, 2);
    arma::mat y = arma::zeros(5000, 1);

    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a ^ b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i) = c;
    }

    for(int i = 0; i < 1000; i++){
        double J = nn.train(X.t(), y);
        if(i && (i%10) == 0)
            std::cout << "Cost at iteration " << i << " is " << J << std::endl;
    }


    X = arma::zeros(1, 2);
    unsigned a = ( rand() & 1 ) ? 1 : 0;
    unsigned b = ( rand() & 1 ) ? 1 : 0;
    unsigned c = ( a | b ) ? 1 : 0;

    X(0, 0) = a;
    X(0, 1) = b;

    std::cout << "Probabilities for " << a << " xor " << b << " being " << c << " out of possible (0 or 1) is:" << std::endl << nn.predict(X.t());
 
