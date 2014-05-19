#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "higgs.h"
#include "smneuralnet.h"



namespace smnn { namespace higgs {

//---------------------------------------------------------------------------------------
// static variables and function declerations
//static const double lambda = 0.00185;
static const double lambda = 0.3785;

static const unsigned int hidden_layer_count_i = 3;
static const unsigned int hidden_layer_units_i = 13;

static const unsigned int hidden_layer_count_0 = 10;
static const unsigned int hidden_layer_units_0 = 11;

static const unsigned int hidden_layer_count_1 = 10;
static const unsigned int hidden_layer_units_1 = 7;

static const unsigned int hidden_layer_count_2 = 10;
static const unsigned int hidden_layer_units_2 = 5;

static const unsigned int hidden_layer_count_3 = 10;
static const unsigned int hidden_layer_units_3 = 3;

static unsigned int get_hidden_unit_size(const unsigned int& n);
static const char* higgs_filename = "c:\\temp\\higgs\\higgs-params";//hidden layer 3

static unsigned int training_iterations = 2000;
//---------------------------------------------------------------------------------------
void train(const char* training_samples_filename, const char* labels_filename)
{
    arma::mat X;
    arma::mat y;
    smnn::layerdesc_t ind;

    std::cout << "Loading training samples." << std::endl;
    X.load(training_samples_filename/*, arma::raw_ascii*/);

#if 0
    for(unsigned int i = 0, j = X.n_cols; i < j; i++){
        const arma::mat cx = X.col(i);
        const double mn = cx.min();//X.col(i).min();
        const arma::mat ex = smnn::SMNeuralNet::normalize(cx);
        X.col(i) = ex;
        if(mn < -998.0 && mn > -1000.0){
            std::cout << "Normalizing column " << (i+1) << std::endl;
            const arma::mat ex = smnn::SMNeuralNet::normalize(cx);
            X.col(i) = ex;
        } else {
            std::cout << "Standardizing column " << (i+1) << std::endl;
            const arma::mat ex = smnn::SMNeuralNet::standardize(cx);
            X.col(i) = ex;
        }
    }
#endif

    std::cout << "Loading samples labels." << std::endl;
    y.load(labels_filename/*, arma::raw_ascii*/);


    // input layer
    ind.push_back(X.n_cols);

    for(unsigned int i = 0; i < hidden_layer_count_i; i++){
        ind.push_back(hidden_layer_units_i);
    }

    for(unsigned int i = 0; i < hidden_layer_count_0; i++){
        ind.push_back(hidden_layer_units_0);
    }

    for(unsigned int i = 0; i < hidden_layer_count_1; i++){
        ind.push_back(hidden_layer_units_1);
    }

    for(unsigned int i = 0; i < hidden_layer_count_2; i++){
        ind.push_back(hidden_layer_units_2);
    }

    for(unsigned int i = 0; i < hidden_layer_count_3; i++){
        ind.push_back(hidden_layer_units_3);
    }
   // outputlayer
    ind.push_back(2);

    // contruct the network
    smnn::SMNeuralNet nn(ind, lambda);

    std::cout << "Training network...."  << std::endl;
    
    double lastJ = 0.0;
    unsigned int mcount = 0;

    for(unsigned int i = 0; i < training_iterations; i++){
        double J = nn.train(X.t(), y);

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


    for(unsigned int i = 0, j = X.n_cols; i < j; i++){
        const arma::mat cx = X.col(i);
        const double mn = cx.min();//X.col(i).min();
        const arma::mat ex = smnn::SMNeuralNet::normalize(cx);
        X.col(i) = ex;
#if 0
        if(mn < -998.0 && mn > -1000.0){
            std::cout << "Normalizing column " << (i+1) << std::endl;
            const arma::mat ex = smnn::SMNeuralNet::normalize(cx);
            X.col(i) = ex;
        } else {
            std::cout << "Standardizing column " << (i+1) << std::endl;
            const arma::mat ex = smnn::SMNeuralNet::standardize(cx);
            X.col(i) = ex;
        }
#endif
    }

    // input layer
    ind.push_back(X.n_cols);

    for(unsigned int i = 0; i < hidden_layer_count_i; i++){
        ind.push_back(hidden_layer_units_i);
    }

    for(unsigned int i = 0; i < hidden_layer_count_0; i++){
        ind.push_back(hidden_layer_units_0);
    }

    for(unsigned int i = 0; i < hidden_layer_count_1; i++){
        ind.push_back(hidden_layer_units_1);
    }

    for(unsigned int i = 0; i < hidden_layer_count_2; i++){
        ind.push_back(hidden_layer_units_2);
    }
    for(unsigned int i = 0; i < hidden_layer_count_3; i++){
        ind.push_back(hidden_layer_units_3);
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
    const arma::mat ps = nn.predict(X.t());

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
static unsigned int get_hidden_unit_size(const unsigned int& n)
{
    return 3;
/*
    const double features = n;
    const double count    = n * (1.0 / 3.7);
    const unsigned res    = count;
    return res;
*/
}
//---------------------------------------------------------------------------------------
}} //namespace smnn , namespace higgs 
