#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "higgs.h"
#include "smneuralnet.h"



namespace smnn { namespace higgs {

//---------------------------------------------------------------------------------------
// static variables and function declerations
//static const double lambda = 0.00185;
static const double lambda = 0.185;
static const char* higgs_filename = "c:\\temp\\higgs\\higgs-params";

static unsigned int get_hidden_unit_size(const unsigned int& n);

static unsigned int training_iterations = 2000;
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

    unsigned hidden_layer_units = get_hidden_unit_size(X.n_cols);
    std::cout << "Hidden layer contains " << hidden_layer_units << " units." << std::endl;

    // input layer
    ind.push_back(X.n_cols);

    // hidden layer 1
    ind.push_back(7);

    // hidden layer 2
    ind.push_back(7);

    // hidden layer 3
    ind.push_back(7);

    // outputlayer
    ind.push_back(2);

    // contruct the network
    smnn::SMNeuralNet nn(ind, lambda);

    std::cout << "Training network...."  << std::endl;
    
    double lastJ = 0.0;
    unsigned int mcount = 0;

    for(unsigned int i = 0; i < training_iterations; i++){
        double J = nn.train(X.t(), y);

        if(J > lastJ && i > 0){
                std::cout << "Warning:  Function not minimizing." << std::endl;

                if(mcount > 20){
                        std::cout << "Giving up.  Function not minimizing." << std::endl;
                        break;
                }

                mcount = mcount + 1;
        }

        lastJ = J;

        std::cout << "Cost of network at iteration " << (i+1) << " is " << J << std::endl;
    }

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

    unsigned hidden_layer_units = get_hidden_unit_size(X.n_cols);
    std::cout << "Hidden layer contains " << hidden_layer_units << " units." << std::endl;

    // input layer
    ind.push_back(X.n_cols);

    // hidden layer 1
    ind.push_back(7);

    // hidden layer 2
    ind.push_back(7);

    // hidden layer 3
    ind.push_back(7);

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
