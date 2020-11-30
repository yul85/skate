#include "EulerAngles.h"
#include <VP/vphysics.h>

object R2euler(const object& pyR, int order);

/////////////////////////////////////
// expose to python

// static axes
object R2zxy_s(const object& pyR) { return R2euler(pyR, EulOrdZXYs); }
object R2xyz_s(const object& pyR) { return R2euler(pyR, EulOrdXYZs); }

// rotating axes
object R2zxy_r(const object& pyR) { return R2euler(pyR, EulOrdZXYr); }
object R2xyz_r(const object& pyR) { return R2euler(pyR, EulOrdXYZr); }

object exp_py(const object& axis_angle_vec);
object log_py(const object& rotation_mat);
object slerp_py(const object& R1, const object& R2, scalar t);
object cross_py(const object& vec1, const object& vec2);
