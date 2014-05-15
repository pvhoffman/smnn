#ifndef SMNEURALNET_H_

#define SMNEURALNET_H_

#include <armadillo>
#include <vector>
#include <stdexcept>

namespace smnn {

typedef std::vector<unsigned int> layerdesc_t;

class SMNeuralNet
{
private:
protected:
    // 'learning rate' - not really but we'll call it that anyway
    double _lambda;

    // thetas - one per two layers, one linear transformation to another.
    std::vector<arma::mat> _thetas;
    
    // test for output layer - hidden layers use regularized logistic regression activation, e.g. 1 / 1 + e^-z
    bool is_output_layer(const unsigned int& n) const;

    arma::mat activate(const arma::mat& z) const;

    //
public:
    // ctor
    SMNeuralNet(const layerdesc_t& layerdesc, const double& lambda );

    //dtor
    virtual ~SMNeuralNet();

    // train on X - X should be in R NxM where N = features, M = training examples 
    // the number of classes, for which we wish to calculate P(y=j|x;theta), is implied 
    // by the size of the last theta matrix
    virtual double train(const arma::mat& X, const arma::mat& y);

    // predict returns a probability vector for each case
    virtual arma::mat predict(const arma::mat& X);

}; // end class SMNeuarlNet

class SMException : public std::runtime_error
{
private:
    std::string _what;
public:
    SMException(const char* what) : std::runtime_error(what), _what(what) {}
    virtual const char* what() {return _what.c_str();};
};




} // end namespace smnn 

#endif /* SMNEURALNET_H_ */

