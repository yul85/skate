#include <iostream>
#include <sstream>
#include <string>
#include "stdafx.h"
#include "../../../PyCommon/externalLibs/common/boostPythonUtil.h"
#include "QuadProg++.hh"

bp::list qp(const object &G_np, const object &g0_np, const object &CE_np, const object &ce0_np, const object &CI_np, const object &ci0_np)
{
    quadprogpp::Matrix<double> G, CE, CI;
    quadprogpp::Vector<double> g0, ce0, ci0, x;

    // todo: Convert numpy -> quadprogpp Matrix or vector structure

    int num_var, num_eq, num_ineq;

    num_var = XI(G_np.attr("shape")[1]);
    num_eq = XI(CE_np.attr("shape")[0]);
//    int num_eq1 = XI(CE_np.attr("shape")[1]);
    num_ineq = XI(CI_np.attr("shape")[0]);

//    printf("CE size is %d, %d . \n", num_eq, num_ineq);

    G.resize(num_var, num_var);

    for(int i=0; i < num_var ; i++)
    {
		for(int j=0; j< num_var ; j++)
		{
			G[i][j] = XD(G_np[i][j]);
		}
	}

    g0.resize(num_var);

    for(int i=0; i < num_var ; i++)
			g0[i] = XD(g0_np[i]);

    CE.resize(num_var, num_eq);
    ce0.resize(num_eq);
//    printf("11111111111111\n");
    if (num_eq != 0)
    {
        for(int i=0; i < num_var ; i++)
        {
            for(int j=0; j< num_eq; j++)
            {
                CE[i][j] = XD(CE_np[j][i]);
            }
        }

        for(int i=0; i < num_eq ; i++)
            ce0[i] = XD(ce0_np[i]);
    }

    CI.resize(num_var, num_ineq);
    ci0.resize(num_ineq);
//    printf("2222222222222222222\n");
    if (num_ineq != 0)
    {
        for(int i=0; i < num_var ; i++)
        {
            for(int j=0; j< num_ineq ; j++)
            {
                CI[i][j] = XD(CI_np[j][i]);
            }
        }

        for(int i=0; i < num_ineq ; i++)
	        ci0[i] = XD(ci0_np[i]);
    }

    x.resize(num_var);

//    printf("333333333333333333\n");
    solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    bp::list ls;
	for(int i=0; i<num_var; i++)
		ls.append(x[i]);
	return ls;
}

BOOST_PYTHON_MODULE(QuadProg)
{
	def("qp", qp);
}
