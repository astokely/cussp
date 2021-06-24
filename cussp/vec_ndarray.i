%include numpy.i
%include <std_vector.i>
#include <vector>

%fragment("VecToNDArray", "header") {
template<typename T>
PyArrayObject* VecToNDArray(
    std::vector<T>& vec, 
    int typenum
){
   if( !vec.empty() ){

       size_t nRows = vec.size();
       npy_intp dims[1] = {nRows};

       PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(
            1, 
            dims, 
            typenum
        );
       T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

       copy(vec.begin(),vec.end(),vec_array_pointer);
       return vec_array;
    } 
    else {
        npy_intp dims[1] = {0};
        return (PyArrayObject*) PyArray_ZEROS(1, dims, typenum, 0);
   }
}
}


%fragment("Py_SequenceToVec", "header") {
template<typename T>
void Py_SequenceToVec(
    PyObject* obj, 
    std::vector<T>& out
) {
    PyArrayObject* arr = (PyArrayObject*) obj;
    if (
        PyArray_Check(arr) && 
        PyArray_ISCARRAY_RO(arr) 
        && PyArray_NDIM(arr) == 1
    ) {
        int type = PyArray_TYPE(arr);
        int length = PyArray_SIZE(arr);
        double* data = (double*) PyArray_DATA(arr);
        if (type == NPY_DOUBLE) {
            out.resize(length);
            double* doubleData = (double*) data;
            for (int i = 0; i < length; i++)
                out[i] = doubleData[i];
            Py_DECREF(arr);
        }
        if (type == NPY_FLOAT) {
            out.resize(length);
            float* floatData = (float*) data;
            for (int i = 0; i < length; i++)
                out[i] = floatData[i];
            Py_DECREF(arr);
        }
        if (type == NPY_INT32) {
            out.resize(length);
            int* intData = (int*) data;
            for (int i = 0; i < length; i++)
                out[i] = intData[i];
            Py_DECREF(arr);
        }
        if (type == NPY_INT64) {
            out.resize(length);
            long long* longData = (long long*) data;
            for (int i = 0; i < length; i++)
                out[i] = longData[i];
            Py_DECREF(arr);
        }
    }
}
}

%typemap(in, fragment="Py_SequenceToVec") 
    std::vector<double> & 
    (std::vector<double> dvec) {
    Py_SequenceToVec($input, dvec);
    $1 = &dvec;
}

%typemap(in, fragment="Py_SequenceToVec") 
    std::vector<double>  
    (std::vector<double> dvec) {
    Py_SequenceToVec($input, dvec);
    $1 = &dvec;
}

%typemap(in, fragment="Py_SequenceToVec") 
    std::vector<int> & 
    (std::vector<int> ivec) {
    Py_SequenceToVec($input, ivec);
    $1 = &ivec;
}

%typemap(out, fragment="VecToNDArray") std::vector<double>& {
    $result = (PyObject*) VecToNDArray(*$1, NPY_DOUBLE);
}

%typemap(out, fragment="VecToNDArray") std::vector<int>& {
    $result = (PyObject*) VecToNDArray(*$1, NPY_INT);
}

