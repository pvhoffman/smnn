#ifndef HIGGS_H_
#define HIGGS_H_

namespace smnn { namespace higgs {

void train(const char* training_samples_filename, const char* labels_filename);
void predict(const char* data_filename, const char* output_filename);

}} //namespace smnn , namespace higgs 


#endif /* HIGGS_H_ */

