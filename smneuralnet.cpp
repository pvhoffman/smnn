#include "stdafx.h"
#include "smneuralnet.h"

namespace smnn {
//---------------------------------------------------------------------------------------
// static functions and variables

//---------------------------------------------------------------------------------------
SMNeuralNet::SMNeuralNet(const layerdesc_t& layerdesc, const double& lambda ) : _lambda(0)
{
    const int n = layerdesc.size();
    if(n < 2){ 
        throw SMException("Softmax neural network must have more than 1 layer.");
    }

    for(int i = 0, j = 1; j < n; i++, j++){
        // plus 1 for bias units
        const unsigned int a = layerdesc[i] + 1;
        const unsigned int b = layerdesc[j];

        // theta will be in (b)x(a)
        _thetas.push_back( arma::randu<arma::mat>(b, a) ); 
    }
}
//---------------------------------------------------------------------------------------
SMNeuralNet::~SMNeuralNet()
{
}
//---------------------------------------------------------------------------------------
arma::mat SMNeuralNet::predict(const arma::mat& X) const
{
   std::vector<arma::mat> as;

    // X is our first activation to use when 
    // calculating the error deltas from the
    // input layer + 1
    as.push_back(X);

    // forward propagate
    for(unsigned int i = 0, j = _thetas.size(); i < j; i++){
        const arma::mat& theta = _thetas[i];
        arma::mat z            =  (theta * arma::join_cols(arma::ones(1, as[i].n_cols), as[i]));

        if( is_output_layer(i) ){
            //softmax 
            // stop the exponential from growing too large without affecting
            // the calculation, because of the nature of division, by
            // subtracting some const, in this case that const is the max value for each column
            arma::mat max_z = arma::max(z);
            z.each_row() -= max_z;

            arma::mat a = arma::exp(z);
            arma::mat t = arma::sum(a);

            a.each_row() /= t;

            as.push_back( a );

        } else {
            // logistic 
            as.push_back( activate(z) );
        }
    }

    return as.back();
}
//---------------------------------------------------------------------------------------
double SMNeuralNet::train(const arma::mat& X, const arma::mat& y)
{
    const double classes  = _thetas[_thetas.size() - 1].n_rows;// - 1;
    const double examples = X.n_cols;
    double cost = 0.0;


    arma::mat gs = arma::zeros<arma::mat>( (unsigned int)classes, (unsigned int) examples );
    //arma::mat ys = arma::zeros<arma::mat>( (unsigned int)classes, (unsigned int) examples );
    // the columns of the first row gs should contain 1 for those examples in the first class and zero otherwise
    // the columns of the second row gs should contan 1 for those examples in the second class and zero otherwise
    // etc
    // con't seem to find a sensable way to do this with armadillo
    for( unsigned int i = 0, j = (unsigned int)classes; i < j; i++){
        for(unsigned int k = 0, l = (unsigned int)examples; k < l; k++){
            if( i == y(k) ){
                gs(i,k) = 1;
            }
        }
    }

    // keep the activation and pre-activation values for
    // gradient/delta calculation
    std::vector<arma::mat> zs;
    std::vector<arma::mat> as;

    // X is our first activation to use when 
    // calculating the error deltas from the
    // input layer + 1
    as.push_back(X);

    // forward propagate
    for(unsigned int i = 0, j = _thetas.size(); i < j; i++){
        const arma::mat& theta = _thetas[i];
        zs.push_back( (theta * arma::join_cols(arma::ones(1, as[i].n_cols), as[i])) );
        if( is_output_layer(i) ){
            //softmax 
            // stop the exponential from growing too large without affecting
            // the calculation, because of the nature of division, by
            // subtracting some const, in this case that const is the max value for each column
            arma::mat max_z = arma::max(zs[i]);
            zs[i].each_row() -= max_z;

            arma::mat a = arma::exp(zs[i]);
            arma::mat t = arma::sum(a);

            a.each_row() /= t;

            as.push_back( a );

        } else {
            // logistic 
            as.push_back( activate(zs[i]) );
        }
    }

    // calculate the cost
    const arma::mat prediction = as.back();//.rows(1, as.back().n_rows - 1);;
    const arma::mat v1 = arma::reshape(gs, gs.n_rows * gs.n_cols, 1);//arma::reshape(gs, 1, gs.n_rows * gs.n_cols);//arma::vectorise(gs, 0);
    const arma::mat v2 = arma::reshape( arma::log(prediction), prediction.n_cols * prediction.n_rows, 1);//arma::reshape( arma::log(prediction), 1, prediction.n_cols * prediction.n_rows);//arma::vectorise( arma::log(prediction), 1 );
    const arma::mat v3 = v1.t() * v2;
    const arma::mat v4 = arma::sum(arma::square(arma::reshape(_thetas.back(), _thetas.back().n_rows * _thetas.back().n_cols, 1 )));

    //std::cout << prediction << std::endl;

    cost = (-1.0 / examples) * v3(0,0);

    // add in the weight decay
    cost = cost + (_lambda / 2.0) * v4(0,0);



    arma::mat delta = (gs - prediction);
    arma::mat gradient;

    bool special = false;

    for(std::vector<arma::mat>::reverse_iterator tc = _thetas.rbegin()
            , te = _thetas.rend()
            , ac = as.rbegin()
            , ae = as.rend()
            , zc = zs.rbegin()
            , ze = zs.rend()

            ; tc != te
            ; tc++){


        if(ac != ae) ac++;
        if(zc != ze) zc++;

        const arma::mat a  = arma::join_cols(arma::ones(1, ac->n_cols), *ac);
        arma::mat& theta   = *tc;

        if(tc == _thetas.rbegin()){
            special = true;
            gradient = ((-1.0 / examples) * ((delta * a.t()) + (_lambda * theta)));
        } else {
            if(special){
                // for the special case the error deltas must be 'converted' to logistic regression
                // piece of cake an has nots of room for optimizations
                const arma::mat& z1 = zs.back();
                const arma::mat  a1 = activate(z1);
                const arma::mat  d1 = (a1 - gs);
                const arma::mat  a2 = arma::join_cols(arma::ones(1, a1.n_cols), a1);

                gradient = (1.0 / examples) * (d1 * a2.t());
            } else {
                gradient = (1.0 / examples) * (delta * a.t());
            }
            const arma::mat reg = arma::join_cols( arma::zeros(1, theta.n_cols), ((_lambda / examples) * theta.rows(1, theta.n_rows - 1)) );
            gradient = gradient + reg;
        }


        // calulcate the next delta
        if(zc != ze){
            if(special){
                const arma::mat& z1 = zs.back();
                const arma::mat  a1 = activate(z1);
                const arma::mat  d1 = (a1 - gs);

                const arma::mat t1 = theta.t() * d1;
                const arma::mat t2 = t1.rows(1, t1.n_rows-1);
                const arma::mat& z = *zc;

                const arma::mat g1 = activate(z);
                const arma::mat g2 = 1.0 - g1;
                const arma::mat g3 = g1 % g2;
                delta = t2 % g3;

                special = false;
            } else {
                const arma::mat t1 = theta.t() * delta;
                const arma::mat t2 = t1.rows(1, t1.n_rows-1);

                const arma::mat& z = *zc;
                const arma::mat g1 = activate(z);
                const arma::mat g2 = 1.0 - g1;
                const arma::mat g3 = g1 % g2;
                delta = t2 % g3;
            }
        }

        theta = (theta - gradient);

        //std::cout << gradient << std::endl;

    }
    return cost;
}
//---------------------------------------------------------------------------------------
bool SMNeuralNet::is_output_layer(const unsigned int& n) const
{
    if(n == _thetas.size() - 1)
        return true;
    return false;
}
//---------------------------------------------------------------------------------------
arma::mat SMNeuralNet::activate(const arma::mat& z) const
{
    const arma::mat a = (-z);
    const arma::mat b = arma::exp(a);
    const arma::mat c = 1.0 + b;
    const arma::mat d = 1.0 / c;
    return d;
}
//---------------------------------------------------------------------------------------
void SMNeuralNet::save(const char* fileName) 
{

    for(unsigned int i = 0, j = _thetas.size(); i < j; i++){
        char suffix[] = {'-', ('0' + i), '\x0'};
        std::string outs(fileName);

        outs.append(suffix);
        if(!_thetas[i].save(outs.c_str(), arma::raw_ascii)){
                std::string what("Cannot save network parameter to ");
                what.append(outs);
                throw SMException(what.c_str());
        }
    }
}
//---------------------------------------------------------------------------------------
void SMNeuralNet::load(const char* fileName) 
{
    WIN32_FIND_DATA find_data;

    char drive[_MAX_DRIVE];
    char path[_MAX_DIR];
    char name[_MAX_FNAME];
    char extn[_MAX_EXT];

    _splitpath(fileName, drive, path, name, extn);

    std::string ins(fileName);
    ins.append("-?");

    HANDLE find_handle = ::FindFirstFile(ins.c_str(), &find_data);

    if(!find_handle || INVALID_HANDLE_VALUE == find_handle) {
        std::string what("Cannot find ");
        what.append(fileName);
        throw SMException(what.c_str());
    }


    do {
        ins.assign(drive);
        ins.append(path);
        ins.append(find_data.cFileName);

        const unsigned int n = strlen(find_data.cFileName) - 1;
        const unsigned int i = find_data.cFileName[n] - '0';
        if(!_thetas[i].load(ins.c_str(), arma::raw_ascii)){
                std::string what("Cannot load parameter from ");
                what.append(ins);

                throw SMException(what.c_str());
        }
    } while(::FindNextFile(find_handle, &find_data));

    ::FindClose(find_handle);

}
//---------------------------------------------------------------------------------------
arma::mat SMNeuralNet::mean_normalize(const arma::mat& cx)
{
    const arma::mat m = arma::mean(cx);
    const double md = m(0,0);

    //const double mx = cx.max();
    //const double mn = cx.min();
    //const double nn = (mx - mn) - 1.0;
    const double nn = cx.n_rows - 1;

    const arma::mat res = (cx - md) / nn;

    return res;
}
//---------------------------------------------------------------------------------------
arma::mat SMNeuralNet::standardize(const arma::mat& cx)
{
    const arma::mat m = arma::mean(cx);
    const double md = m(0,0);


    const arma::mat s = arma::stddev(cx);
    const double ms = s(0,0);

    const arma::mat res = (cx - md) / ms;

    return res;
}
//---------------------------------------------------------------------------------------
arma::mat SMNeuralNet::normalize(const arma::mat& cx)
{
    /* mx = Max of X(<all rows>,column)*/
    /* mn = Min of X(<all rows>,column)*/
    /* X' = (X - mn) / (mx / mn) */

    const double a = 0.0;
    const double b = 200.0;

    const double mx = cx.max();
    const double mn = cx.min();
    const double mz = mx - mn;

    const arma::mat t1 = (cx - mn);
    const arma::mat t2 = t1 * (b - a);
    const arma::mat t3 = t2 / mz;
    const arma::mat t4 = t3 + a;

    //arma::mat res = ((cx - mn) / mz);

    return t4;
}
//---------------------------------------------------------------------------------------
// static functions and variables
//---------------------------------------------------------------------------------------
} // end namespace smnn 

