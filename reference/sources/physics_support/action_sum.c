/* E2 on edges and E4 averaged over the eight orientations of each cell.
 * Direct translation of the R249 corrected discrete action. */
#include <stddef.h>

static double dot4(const double *x, const double *y) {
    double value = 0.;
    for (int j = 0; j < 4; ++j) value += x[j]*y[j];
    return value;
}

static void derivative_link(const double *core, const double *neighbor,
                            double scalar, const double *momentum,
                            double *dcore, double *dneighbor) {
    double product = dot4(neighbor, momentum)/(scalar*scalar);
    for (int j = 0; j < 4; ++j) {
        dcore[j] -= momentum[j]+neighbor[j]*product;
        dneighbor[j] += momentum[j]/scalar-core[j]*product;
    }
}

int bst_action_compensated(int n, double h, double alpha, double beta, const double *field, double *gradient, long double *out) {
    const ptrdiff_t stride[3] = {(ptrdiff_t)4*n*n, 4*n, 4};
    const double volume = h*h*h;
    long double e2 = 0., e4 = 0.;
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            for (int z = 0; z < n; ++z) {
                const ptrdiff_t center = x*stride[0]+y*stride[1]+z*stride[2];
                const int coord[3] = {x,y,z};
                const double *core = field+center;
                for (int a = 0; a < 3; ++a) {
                    if (coord[a]+1 == n) continue;
                    const ptrdiff_t i = center+stride[a];
                    const double *neighbor = field+i;
                    const double scalar = dot4(core,neighbor);
                    if (scalar <= 0.) return 1;
                    double tangent[4], momentum[4];
                    for (int j = 0; j < 4; ++j) {
                        tangent[j] = (neighbor[j]/scalar-core[j])/h;
                        momentum[j] = alpha*volume*tangent[j]/h;
                    }
                    e2 += .5*volume*dot4(tangent,tangent);
                    derivative_link(core,neighbor,scalar,momentum,gradient+center,gradient+i);
                }
                for (int bits = 0; bits < 8; ++bits) {
                    ptrdiff_t index[3];
                    int valid = 1;
                    for (int a = 0; a < 3; ++a) {
                        int sign = bits & (1<<a) ? -1 : 1;
                        if (coord[a]+sign < 0 || coord[a]+sign >= n) valid = 0;
                        index[a] = center+sign*stride[a];
                    }
                    if (!valid) continue;
                    double d[3][4], norm[3], scalar[3], momentum[3][4] = {{0.}};
                    for (int a = 0; a < 3; ++a) {
                        const double *neighbor = field+index[a];
                        scalar[a] = dot4(core,neighbor);
                        if (scalar[a] <= 0.) return 1;
                        for (int j = 0; j < 4; ++j) d[a][j] = (neighbor[j]/scalar[a]-core[j])/h;
                        norm[a] = dot4(d[a],d[a]);
                    }
                    for (int a = 0; a < 3; ++a) {
                        for (int b = a+1; b < 3; ++b) {
                            const double p = dot4(d[a],d[b]);
                            const double weight = beta*volume/(8.*h);
                            e4 += volume/16.*(norm[a]*norm[b]-p*p);
                            for (int j = 0; j < 4; ++j) {
                                momentum[a][j] += weight*(d[a][j]*norm[b]-d[b][j]*p);
                                momentum[b][j] += weight*(d[b][j]*norm[a]-d[a][j]*p);
                            }
                        }
                    }
                    for (int a = 0; a < 3; ++a) {
                        derivative_link(core,field+index[a],scalar[a],momentum[a],
                                        gradient+center,gradient+index[a]);
                    }
                }
            }
        }
    }
    out[0] = alpha*e2+beta*e4; out[1] = e2; out[2] = e4;
    return 0;
}
