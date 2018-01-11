#include <algorithm>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <map>
#include <fstream>
#include "lbfgs.h"
#include "Tensor.cpp"

Tensor * T;
int Jn, Kn, In, L;
float lambda;
// you need to decide how to define In, Jn, Kn, L, and how to get T.get(i,j,k)
static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n, // the number of the parameters
    const lbfgsfloatval_t step
    )
{
    int i = 0, j = 0, k = 0;
    int l = 0;
    lbfgsfloatval_t fx = 0.0;
    lbfgsfloatval_t * tmp = lbfgs_malloc(In * Jn * Kn);
    int Joffset = In * L;
    int Koffset = (In + Jn) * L;


    // compute for the sum of l minus y_{ijk}
    for(i = 0; i < In; i++) {
        for(j = 0; j < Jn;  j++) {
            for(k = 0; k < Kn;  k++) {
                // be careful to the 0 of tensor
                // may be you need to put a if here to check\
                whether the position of (i, j, k) in tensor is 0
                if(!T->contain(i, j, k)) {
                    continue;
                }
//                printf("for i(%d), j(%d), k(%d) the value is %d\n", i, j, k, T->contain(i, j, k));
	        int Ti = i * L;
	        int Tj = Joffset + j * L;
	        int Tk = Koffset + k * L;
                lbfgsfloatval_t yhat = 0.0;
                lbfgsfloatval_t u, w, v, y;
                for(l = 0; l < L; l++) {
                    u = x[Ti+l]; w = x[Tj+l]; v = x[Tk+l];
                    yhat += u * w * v;
                } 
                //ToDo define T
                y = T->get(i, j, k);
 //               printf("y get from T is %f\n", y);
  //              printf("yhat calculate is %f\n", yhat);
                yhat -= y;
                fx += yhat * yhat;
   //             printf("yhat calculate is %f\n", yhat);
                tmp[i * In + j * Jn + k] = yhat;
            }
    //        printf("the fx is %f\n", fx);
        }
    }

    //regularization
    for(int nn = 0; nn < n; nn++) {
        fx += lambda * x[nn] * x[nn];
    }

    // compute for the g
    // Todo to init g into 0
    for(int gg = 0; gg < n; gg++) {
        g[gg] = 2 * lambda * x[gg];
    }
    for(i = 0; i < In;  i++) {
	int Ti = i * L;
        for(j = 0; j < Jn;  j++) {
	    int Tj = Joffset + j * L;
            for(k = 0; k < Kn;  k++) {
                if(!T->contain(i, j, k)) {
                    continue;
                }
	        int Tk = Koffset + k * L;
                for(l = 0; l < L; l++) {
                    g[Ti+l] += 2*tmp[i*In+j*Jn+k] * x[Tj+l] * x[Tk+l];
                }
            }
        }
    }
    for(j = 0; j < Jn;  j++) {
        int Tj = Joffset + j * L;
        for(i = 0; i < In;  i++) {
            int Ti = i * L;
            for(k = 0; k < Kn;  k++) {
                if(!T->contain(i, j, k)) {
                    continue;
                }
                int Tk = Koffset + k * L;
                for(l = 0; l < L; l++) {
                    g[Tj+l] += 2*tmp[i*In+j*Jn+k] * x[Ti+l] * x[Tk+l];
                }
            }
        }
    }
    for(k = 0; k < Kn;  k++) {
        int Tk = Koffset + k * L;
        for(i = 0; i < In;  i++) {
            int Ti = i * L;
            for(j = 0; j < Jn;  j++) {
                if(!T->contain(i, j, k)) {
                    continue;
                }
                int Tj = Joffset + j * L;
                for(l = 0; l < L; l++) {
                    g[Tk+l] += 2*tmp[i*In+j*Jn+k] * x[Ti+l] * x[Tj+l];
                }
            }
        }
    }

    return fx;
    //int i;
    //lbfgsfloatval_t fx = 0.0;
    //for (i = 0;i < n;i++ ) {
    //    lbfgsfloatval_t t1 = x[i] * x[i];
    //    g[i] = 2 * x[i];
    //    fx += t1;
    //}
    //return fx;
}
static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("fx: %f", fx);
    for(int i = 0; i < n; i++) {
        printf(", x[%d] = %f", i, x[i]);
    }
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}



int main(int argc, char * argv[]) 
{
  int i, ret = 0;
  Tensor tt("data.txt");
  std::cout << tt << std::endl;
  T = &tt;
  In = T->In;
  Jn = T->Jn;
  Kn = T->Kn;
  std::cout << "the argc is " << argc << "printed argc" << argv[0] << std::endl;
  L = atoi(argv[2]);
  int NW = (In + Jn + Kn) * L;
  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(NW);
  lbfgs_parameter_t param;
  if(x == NULL) {
    printf("Error: memory error.\n");
    return 1;
  }
  lambda = atof(argv[3]);

  for(i = 0; i < NW; i++) {
    x[i] = atof(argv[1]);
  }
  lbfgs_parameter_init(&param);
  std::cout << tt << std::endl;
  ret = lbfgs(NW, x, &fx, evaluate, progress, NULL, &param);
  printf("L-BFGS optimization terminated with status code=%d\n", ret);
  lbfgs_free(x);
  return 0;
}
