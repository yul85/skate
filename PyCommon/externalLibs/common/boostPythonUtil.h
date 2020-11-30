#ifndef _BPUTIL_H_
#define _BPUTIL_H_

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#define XD(v)	((extract<double>)(v))
#define XC(v)	((extract<char*>)(v))
#define XS(v)	((extract<string>)(v))
#define XF(v)	((extract<float>)(v))
#define XI(v)	((extract<int>)(v))
#define XB(v)	((extract<bool>)(v))

//void printSO3(const object& SO3);

#endif // _BPUTIL_H_
