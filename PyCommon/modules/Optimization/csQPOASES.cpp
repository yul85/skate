#include <qpOASES.hpp>
#include "stdafx.h"
#include "../../../PyCommon/externalLibs/common/boostPythonUtil.h"
#include <string>


int example(void);
bp::list qp(const object &H, const object &g, const object &A, const object &lb, const object &ub, const object &lbA, const object &ubA, int nWSR, bool printObjVal, std::string printLevel);

BOOST_PYTHON_MODULE(csQPOASES)
{
	//numeric::array::set_module_and_type("numpy", "ndarray");
	def("example", example);
	def("qp", qp);
}

bp::list qp(const object &H, const object &g, const object &A, const object &lb, const object &ub, const object &lbA, const object &ubA, int nWSR, bool printObjVal, std::string printLevel)
{
	USING_NAMESPACE_QPOASES


	int nV = XI(H.attr("shape")[1]);
	int nC = XI(A.attr("shape")[0]);

	real_t H_qp[nV*nV];
	real_t A_qp[nC*nV];
	real_t g_qp[nV];
	real_t lb_qp[nV];
	real_t ub_qp[nV];
	real_t lbA_qp[nC];
	real_t ubA_qp[nC];

	bool Blb = true;
	bool Bub = true;
	bool BlbA = true;
	bool BubA = true;
	if(lb.is_none())
		Blb = false;
	if(ub.is_none())
		Bub = false;
	if(lbA.is_none())
		BlbA = false;
	if(ubA.is_none())
		BubA = false;

	for(int i=0; i < nV ; i++){
		for(int j=0; j< nV ; j++){
			H_qp[i*nV + j] = XD(H[i][j]);
		}
	}

	for(int i=0; i<nV; i++){
		g_qp[i] = XD(g[i]);
		if(Blb) lb_qp[nV] = XD(lb[i]);
		if(Bub) ub_qp[nV] = XD(ub[i]);
	}

	for(int i=0; i< nC; i++){
		for(int j=0; j<nV; j++){
			A_qp[i*nV + j] = XD(A[i][j]);
		}
	}

	for(int i=0; i<nC; i++){
		if(BlbA) lbA_qp[nV] = XD(lbA[i]);
		if(BubA) ubA_qp[nV] = XD(ubA[i]);
	}

	QProblem qp(nV, nC);

	Options options;
	options.printLevel = PL_NONE;
	if(!printLevel.compare("LOW"))
		options.printLevel = PL_LOW;
	else if(!printLevel.compare("MEDIUM"))
		options.printLevel = PL_MEDIUM;
	else if(!printLevel.compare("HIGH"))
		options.printLevel = PL_HIGH;

	qp.setOptions(options);


	qp.init(H_qp, g_qp, A_qp, (Blb ? lb_qp : NULL), (Bub ? ub_qp : NULL), (BlbA ? lbA_qp : NULL), (BubA ? ubA_qp : NULL), nWSR);

	real_t xOpt[nV];
	qp.getPrimalSolution(xOpt);

	//qp.printOptions();

	if(printObjVal)
		printf("objVal : %e\n", qp.getObjVal());
	bp::list ls;
	for(int i=0; i<nV; i++)
		ls.append(xOpt[i]);
	return ls;
}


/** Example for qpOASES main function using the QProblem class. */
int example( )
{
	USING_NAMESPACE_QPOASES

	/* Setup data of first QP. */
	real_t H[2*2] = { 1.0, 0.0, 0.0, 0.5 };
	real_t A[1*2] = { 1.0, 1.0 };
	real_t g[2] = { 1.5, 1.0 };
	real_t lb[2] = { 0.5, -2.0 };
	real_t ub[2] = { 5.0, 2.0 };
	real_t lbA[1] = { -1.0 };
	real_t ubA[1] = { 2.0 };

	/* Setup data of second QP. */
	real_t g_new[2] = { 1.0, 1.5 };
	real_t lb_new[2] = { 0.0, -1.0 };
	real_t ub_new[2] = { 5.0, -0.5 };
	real_t lbA_new[1] = { -2.0 };
	real_t ubA_new[1] = { 1.0 };


	/* Setting up QProblem object. */
	QProblem example( 2,1 );

	Options options;
	example.setOptions( options );

	/* Solve first QP. */
	int_t nWSR = 10;
	example.init( H,g,A,lb,ub,lbA,ubA, nWSR );

	/* Get and print solution of first QP. */
	real_t xOpt[2];
	real_t yOpt[2+1];
	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf( "\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", 
			xOpt[0],xOpt[1],yOpt[0],yOpt[1],yOpt[2],example.getObjVal() );
	
	/* Solve second QP. */
	nWSR = 10;
	example.hotstart( g_new,lb_new,ub_new,lbA_new,ubA_new, nWSR );

	/* Get and print solution of second QP. */
	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf( "\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", 
			xOpt[0],xOpt[1],yOpt[0],yOpt[1],yOpt[2],example.getObjVal() );

	example.printOptions();
	/*example.printProperties();*/

	return 0;
}

