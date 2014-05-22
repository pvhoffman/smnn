#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "higgs.h"
#include "smneuralnet.h"



namespace smnn { namespace higgs {

//---------------------------------------------------------------------------------------
// static variables and function declerations
static const double lambda = 0.00185;
//static const double lambda = 0.3785;

static const unsigned int highest_poly_order = 3;

static const unsigned int hidden_layer_count_0 = 2;
static const unsigned int hidden_layer_units_0 = 8;

static const unsigned int hidden_layer_count_1 = 3;
static const unsigned int hidden_layer_units_1 = 3;

static const char* higgs_filename = "c:\\temp\\higgs\\higgs-params";//hidden layer 3

static unsigned int training_iterations = 2000;

static arma::mat polynomialize(const arma::mat& X, const unsigned int& toorder);
//---------------------------------------------------------------------------------------
void train(const char* training_samples_filename, const char* labels_filename)
{
    arma::mat X;
    arma::mat y;
    smnn::layerdesc_t ind;

    std::cout << "Loading training samples." << std::endl;
    X.load(training_samples_filename/*, arma::raw_ascii*/);

    std::cout << "Loading samples labels." << std::endl;
    y.load(labels_filename/*, arma::raw_ascii*/);


    std::cout << "Mean-normalizing test samples." << std::endl;
    for(unsigned int i = 0; i < X.n_cols; i++){
        const arma::mat cx = X.col(i);
        const arma::mat cy = smnn::SMNeuralNet::normalize(cx);
        X.col(i) = cy;
    }

    std::cout << "Polynomializing test samples" << std::endl;
    arma::mat Xp = polynomialize(X, highest_poly_order);

    std::cout << "Input layer contains " << Xp.n_cols << " units." << std::endl;

    // input layer
    ind.push_back(Xp.n_cols);

    for(unsigned int i = 0; i < hidden_layer_count_0; i++){
        ind.push_back(hidden_layer_units_0);
    }

    for(unsigned int i = 0; i < hidden_layer_count_1; i++){
        ind.push_back(hidden_layer_units_1);
    }

   // outputlayer
    ind.push_back(2);

    // contruct the network
    smnn::SMNeuralNet nn(ind, lambda);

    std::cout << "Training network...."  << std::endl;
    
    double lastJ = 0.0;
    unsigned int mcount = 0;

    for(unsigned int i = 0; i < training_iterations; i++){
        double J = nn.train(Xp.t(), y);

        std::cout << "Cost at iteration " << (i+1) << " is " << J << std::endl;

        if(J > lastJ && i > 0){
                std::cout << "Warning:  Function not minimizing." << std::endl;

                if(mcount > 20){
                        std::cout << "Giving up.  Function not minimizing." << std::endl;
                }
        }

        lastJ = J;
    }

    std::cout << "Training network...."  << std::endl;
    
    std::cout << "Saving network parameters."  << std::endl;
    nn.save(higgs_filename);

    std::cout << "Training complete." << std::endl;

}
//---------------------------------------------------------------------------------------
void predict(const char* data_filename, const char* output_filename)
{
    arma::mat X;
    smnn::layerdesc_t ind;
    const unsigned int base_event_id = 350000;

    std::cout << "Loading test samples." << std::endl;
    X.load(data_filename/*, arma::raw_ascii*/);


    std::cout << "Mean-normalizing test samples." << std::endl;
    for(unsigned int i = 0; i < X.n_cols; i++){
        const arma::mat cx = X.col(i);
        const arma::mat cy = smnn::SMNeuralNet::normalize(cx);
        X.col(i) = cy;
    }

    std::cout << "Polynomializing test samples" << std::endl;
    arma::mat Xp = polynomialize(X, highest_poly_order);

    std::cout << "Input layer contains " << Xp.n_cols << " units." << std::endl;
   // input layer
    ind.push_back(Xp.n_cols);

    for(unsigned int i = 0; i < hidden_layer_count_0; i++){
        ind.push_back(hidden_layer_units_0);
    }

    for(unsigned int i = 0; i < hidden_layer_count_1; i++){
        ind.push_back(hidden_layer_units_1);
    }

    // outputlayer
    ind.push_back(2);

    // contruct the network
    smnn::SMNeuralNet nn(ind, lambda);

    // load the trained network parameters
    std::cout << "Loading trained network parameters." << std::endl;
    nn.load(higgs_filename);

    //predict
    std::cout << "Predicting results." << std::endl;
    const arma::mat ps = nn.predict(Xp.t());

    FILE* fpout = fopen(output_filename, "wt");

    if(!fpout){ 
        std::string what("Cannot create file ");
        what.append(output_filename);
        throw std::runtime_error(what);
    }

    std::cout << "Writing results to disk." << std::endl;

    fputs("PS, PB, EventId, Class\n", fpout);

    for(unsigned int i = 0, j = ps.n_cols; i < j; i++){
        const double p_of_b_given_x = ps(0, i);
        const double p_of_s_given_x = ps(1, i);

        const char klass = (p_of_b_given_x > p_of_s_given_x) ? 'b' : 's';
        const unsigned int eventid = base_event_id + i;

        fprintf(fpout, "%f, %f, %d, %c\n", p_of_s_given_x, p_of_b_given_x, eventid, klass);
        
    }

    fclose(fpout);


    std::cout << "Prediction complete." << std::endl;
}
//---------------------------------------------------------------------------------------
static arma::mat polynomialize(const arma::mat& X, const unsigned int& toorder)
{
        if(!toorder) return X;

        arma::mat Xp = arma::zeros(X.n_rows, X.n_cols * toorder);

        for(unsigned int i = 0, k = 0 ; i < X.n_cols; i++){
            for(unsigned int j = 0; j < toorder; j++){
                Xp.col(k++) = arma::pow(X.col(i),(j + 1));
            }
        }
        return Xp;
}
//---------------------------------------------------------------------------------------
}} //namespace smnn , namespace higgs 
