

%module cussp 
%include "std_vector.i"
%{
#include "../cuda/cussp.h"
%}
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%include "std_string.i"

%include "../cuda/cussp.h"



