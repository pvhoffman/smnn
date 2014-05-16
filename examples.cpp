#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "smneuralnet.h"
#include "higgs.h"

void higgs_signal_prediction()
{
    const char* training_samples_filename = "training-x.out";
    const char* labels_filename = "training-y.out";
    const char* data_filename = "test-x.out"; 
    const char* output_filename = "higgs-predict.out";

    smnn::higgs::train(training_samples_filename, labels_filename);
    smnn::higgs::predict(data_filename, output_filename);
}

int main (int argc, char const* argv[])
{
    higgs_signal_prediction();
#if 0
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
#endif
    return 0;
}

