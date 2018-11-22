#include "stdafx.h"

#include "../../externalLibs/common/boostPythonUtil.h"
#include "csMath.h"
#include "EulerAngles.h"
#include "../../externalLibs/common/VPUtil.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

static bool np_initialized = false;

BOOST_PYTHON_MODULE(csMath)
{
	def("R2zxy_s", R2zxy_s);
	def("R2xyz_s", R2xyz_s);
	def("R2zxy_r", R2zxy_r);
	def("R2xyz_r", R2xyz_r);
	def("exp", exp_py);
	def("log", log_py);
	def("slerp", slerp_py);
	def("cross", cross_py);
}

void pyR_2_H(const object& pyR, HMatrix& H)
{
	H[0][0] = (float)XD(pyR[0][0]); H[0][1] = (float)XD(pyR[0][1]); H[0][2] = (float)XD(pyR[0][2]); H[0][3] = 0.; 
	H[1][0] = (float)XD(pyR[1][0]); H[1][1] = (float)XD(pyR[1][1]); H[1][2] = (float)XD(pyR[1][2]); H[1][3] = 0.;
	H[2][0] = (float)XD(pyR[2][0]); H[2][1] = (float)XD(pyR[2][1]); H[2][2] = (float)XD(pyR[2][2]); H[2][3] = 0.;
	H[3][0] = 0.;					H[3][1] = 0.;					H[3][2] = 0.;					H[3][3] = 1.; 
}

object R2euler(const object& pyR, int order)
{
    if (!np_initialized)
    {
        np::initialize();
        np_initialized = true;
    }
///* EulerSample.c - Read angles as quantum mechanics, write as aerospace */
//#include <stdio.h>
//#include "EulerAngles.h"
//void main(void)
//{
//    EulerAngles outAngs, inAngs = {0,0,0,EulOrdXYXr};
//    HMatrix R;
//    printf("Phi Theta Psi (radians): ");
//    scanf("%f %f %f",&inAngs.x,&inAngs.y,&inAngs.z);
//    Eul_ToHMatrix(inAngs, R);
//    outAngs = Eul_FromHMatrix(R, EulOrdXYZs);
//    printf(" Roll   Pitch  Yaw	  (radians)\n");
//    printf("%6.3f %6.3f %6.3f\n", outAngs.x, outAngs.y, outAngs.z);

	static np::ndarray O = np::array(bp::make_tuple(0.,0.,0.));
	static HMatrix H;

    //np::ndarray O = np::array(bp::make_tuple(0., 0., 0.));
 
	pyR_2_H(pyR, H);
	EulerAngles angs  = Eul_FromHMatrix(H, order);
 
	object pyV = O.copy();
	pyV[0] = angs.x;
	pyV[1] = angs.y;
	pyV[2] = angs.z;
 	
	return pyV; 
}

object exp_py( const object& axis_angle_vec )
{
    if (!np_initialized)
    {
        np::initialize();
        np_initialized = true;
    }

	static np::ndarray I = np::array( bp::make_tuple(bp::make_tuple(1.,0.,0.), bp::make_tuple(0.,1.,0.), bp::make_tuple(0.,0.,1.)) );
	static Vec3 vAxis;
	//np::ndarray I = np::array( bp::make_tuple(
        //bp::make_tuple(1.,0.,0.), 
        //bp::make_tuple(0.,1.,0.), 
        //bp::make_tuple(0.,0.,1.)) );

	object pyR = I.copy();
	pyVec3_2_Vec3(axis_angle_vec, vAxis);
	SE3_2_pySO3(Exp((Axis)vAxis), pyR);


//    R[0,0] = c + (1.0-c)*x*x;R[0,1] = (1.0-c)*x*y - s*z;R[0,2] = (1-c)*x*z + s*y;
//    R[1,0] = (1.0-c)*x*y + s*z;R[1,1] = c + (1.0-c)*y*y;R[1,2] = (1.0-c)*y*z - s*x;
//    R[2,0] = (1.0-c)*z*x - s*y;R[2,1] = (1.0-c)*z*y + s*x;R[2,2] = c + (1.0-c)*z*z;
//    return R
	
	return pyR;
}

object log_py( const object& rotation_mat )
{
    if (!np_initialized)
    {
        np::initialize();
        np_initialized = true;
    }

	//static numeric::array O(bp::make_tuple(0.,0.,0.));
	static SE3 T;
	static se3 d;


	pySO3_2_SE3(rotation_mat, T);
	d = Log(T);
	np::ndarray pyV = np::array(bp::make_tuple(0.,0.,0.));
//	object pyV;
//  make_pyVec3(pyV);
	pyV[0] = d[0];
	pyV[1] = d[1];
	pyV[2] = d[2];
	return pyV;
}

object slerp_py( const object& R1, const object& R2, scalar t )
{
	//static numeric::array I( bp::make_tuple(bp::make_tuple(1.,0.,0.), bp::make_tuple(0.,1.,0.), bp::make_tuple(0.,0.,1.)) );
	static SE3 T1;
	static SE3 T2;
	static SE3 T_slerp;

    if (!np_initialized)
    {
        np::initialize();
        np_initialized = true;
    }

	pySO3_2_SE3(R1, T1);
	pySO3_2_SE3(R2, T2);
	T_slerp = slerp(T1, T2, t);

    object pyR;
    make_pySO3(pyR);
	SE3_2_pySO3(T_slerp, pyR);

	return pyR;
}

object cross_py( const object& vec1, const object& vec2 )
{
    if (!np_initialized)
    {
        np::initialize();
        np_initialized = true;
    }
	np::ndarray O = np::array(bp::make_tuple(0.,0.,0.));
	O[0] = vec1[1]*vec2[2] - vec1[2] * vec2[1];
	O[1] = vec1[2]*vec2[0] - vec1[0] * vec2[2];
	O[2] = vec1[0]*vec2[1] - vec1[1] * vec2[0];

	return O;
}
