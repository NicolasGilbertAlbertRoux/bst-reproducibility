// Fixed-box R443 molecular dynamics with a smooth EATM trimer term.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Vec = std::array<double, 3>;

struct Pair {
    double length, energy;
    std::array<double, 6> a;

    std::array<double, 2> evaluate(double r) const {
        const double x = r/length;
        const double x2 = x*x;
        const double repulsive = a[0]*std::exp(a[1]*x)/x;
        const double attractive = a[2]*std::exp(a[3]*x);
        const double numerator = repulsive+attractive+a[4];
        const double correction = a[5]*x2*x2*x2;
        const double den = 1.+correction;
        const double u = numerator/den;
        const double derivative = (repulsive*(a[1]-1./x)+a[3]*attractive-u*6.*correction/x)/den;
        return {energy*u,-energy*derivative/length};
    }
};

struct Trimer {
    double bohr_per_length, hartree_per_energy, c9, alpha, low, high;
    std::array<double,5> a;
    double overlap_range=2.45;
    std::array<double,4> overlap_amplitudes{};
    double probe_correction() const {
        const Vec sides{1.13,1.13,1.13};
        const auto result=evaluate(sides);
        Trimer baseline=*this;
        baseline.overlap_amplitudes.fill(0.);
        return result.first-baseline.evaluate(sides).first;
    }

    static std::array<double,2> switch_off(double r,double low,double high) {
        if (r<=low) return {1.,0.};
        if (r>=high) return {0.,0.};
        const double t = (r-low)/(high-low);
        return {1.-t*t*t*(10.-15.*t+6.*t*t),-30.*t*t*(1.-t)*(1.-t)/(high-low)};
    }

    std::pair<double,Vec> evaluate(const Vec &r) const {
        const double x = r[0], y = r[1], z = r[2];
        const double w = x*y*z;
        const double A = x*x+y*y-z*z, B = x*x+z*z-y*y, C = y*y+z*z-x*x;
        const double p = A*B*C;
        const double f = 1.+.375*p/(w*w);
        const Vec dp = {2.*x*(B*C+A*C-A*B),2.*y*(B*C-A*C+A*B),2.*z*(-B*C+A*C+A*B)};
        const double g2 = std::pow(w,2./3.)*bohr_per_length*bohr_per_length;
        double poly=a[4], derivative=0.;
        for (int k=3;k>=0;--k) {
            derivative=derivative*g2+poly;
            poly=poly*g2+a[k];
        }
        const double e=std::exp(-alpha*bohr_per_length*(x+y+z));
        const double inv=1./(w*bohr_per_length*bohr_per_length*bohr_per_length);
        const double dispersion=c9*inv*inv*inv;
        const double u=dispersion+e*poly;
        std::array<std::array<double,2>,3> taper;
        for (int k=0;k<3;++k) {
            const auto far=switch_off(r[k],low,high);
            const auto near=switch_off(r[k],.63,.73);
            taper[k]={far[0]*(1.-near[0]),far[1]*(1.-near[0])-far[0]*near[1]};
        }
        const double S=taper[0][0]*taper[1][0]*taper[2][0];
        Vec grad;
        for (int k=0;k<3;++k) {
            const double df=.375*(dp[k]-2.*p/r[k])/(w*w);
            const double du=-3.*dispersion/r[k]+e*(derivative*(2./3.)*g2/r[k]-poly*alpha*bohr_per_length);
            const double dS=taper[k][1]*taper[(k+1)%3][0]*taper[(k+2)%3][0];
            grad[k]=hartree_per_energy*(S*(df*u+f*du)+dS*f*u);
        }
        double extra=0.;
        if (*std::max_element(r.begin(),r.end())<overlap_range) {
            Vec cut, derivative_cut;
            for (int j=0;j<3;++j) {
                const double scale=r[j]/overlap_range;
                const double gate=1.-std::pow(scale,18);
                cut[j]=gate*gate*gate;
                derivative_cut[j]=-54.*gate*gate*std::pow(scale,17)/overlap_range;
            }
            const double cutoff=cut[0]*cut[1]*cut[2];
            const double mean=(x+y+z)/3.;
            constexpr std::array<double,4> widths{.23,.41,.12,.28}, sites{1.04,1.34,1.14,1.60};
            for (int p=0;p<4;++p) {
                const double displacement=mean-sites[p];
                const double curve=overlap_amplitudes[p]*std::exp(-std::pow(displacement/widths[p],2));
                extra+=cutoff*curve;
                const double radial=-cutoff*curve*(2./3.)*displacement/(widths[p]*widths[p]);
                for (int j=0;j<3;++j) grad[j]+=radial+curve*derivative_cut[j]*cut[(j+1)%3]*cut[(j+2)%3];
            }
        }
        return {hartree_per_energy*S*f*u+extra,grad};
    }
};

struct Model {
    const int cells;
    const int count;
    const double volume, side, temperature, dt, cutoff, skin = .4;
    const Pair law;
    const std::array<double, 2> cut_value;
    const Trimer trimer;
    const bool three_body;
    std::vector<Vec> x, v, force, start, unwrapped, origins;
    std::vector<std::pair<int,int>> pairs;
    std::vector<std::array<int,3>> triples;
    std::array<double,5> stress{};
    double potential = 0., virial = 0., min_separation = 10.;
    std::mt19937_64 rng;

    Model(int cells_, double density, double t, const Pair &law_, double dt_, double cut, uint64_t seed,
          const Trimer &trimer_, bool three_body_)
        : cells(cells_), count(4*cells*cells*cells), volume(count/density), side(std::cbrt(volume)),
          temperature(t), dt(dt_), cutoff(cut), law(law_), cut_value(law.evaluate(cut)), trimer(trimer_),
          three_body(three_body_),
          x(count), v(count), force(count), start(count), unwrapped(count), origins(count), rng(seed) {
        if (side <= 2.*(cutoff+skin) || (three_body && side<=3.*trimer.high))
            throw std::runtime_error("Box too short for consistent minimum-image triangles");
        const std::array<Vec,4> sites{{{0.,0.,0.},{0.,.5,.5},{.5,0.,.5},{.5,.5,0.}}};
        const double unit = side/cells;
        int idx=0;
        for (int i=0; i<cells; ++i)
            for (int j=0; j<cells; ++j)
                for (int k=0; k<cells; ++k)
                    for (const auto &s:sites) x[idx++]={unit*(i+s[0]),unit*(j+s[1]),unit*(k+s[2])};
        std::normal_distribution<double> normal;
        Vec mean{};
        for (auto &q:v)
            for (int d=0; d<3; ++d) { q[d]=normal(rng); mean[d]+=q[d]/count; }
        for (auto &q:v)
            for (int d=0; d<3; ++d) q[d]-=mean[d];
        unwrapped=x;
        rescale();
        rebuild();
        evaluate();
    }

    Vec distance(const Vec &a,const Vec &b) const {
        Vec d;
        for (int k=0; k<3; ++k) {
            d[k]=a[k]-b[k];
            if (d[k]>side*.5) d[k]-=side;
            if (d[k]<-side*.5) d[k]+=side;
        }
        return d;
    }

    static double square(const Vec &d) {
        return d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
    }

    void rebuild() {
        const double radius2=(cutoff+skin)*(cutoff+skin);
        pairs.clear();
        for (int i=0; i<count; ++i)
            for (int j=i+1; j<count; ++j)
                if (square(distance(x[i],x[j]))<radius2) pairs.emplace_back(i,j);
        if (three_body) {
            triples.clear();
            const double r2 = (trimer.high+skin)*(trimer.high+skin);
            std::vector<std::vector<int>> nearby(count);
            for (const auto &[i,j]:pairs)
                if (square(distance(x[i],x[j]))<r2) nearby[i].push_back(j);
            for (int i=0;i<count;++i) {
                const auto &row=nearby[i];
                for (std::size_t a=0;a<row.size();++a)
                    for (std::size_t b=a+1;b<row.size();++b)
                        if (square(distance(x[row[a]],x[row[b]]))<r2)
                            triples.push_back({i,row[a],row[b]});
            }
        }
        start=x;
    }

    void evaluate() {
        potential=virial=0.;
        stress.fill(0.);
        std::fill(force.begin(),force.end(),Vec{});
        const double rc2=cutoff*cutoff;
        for (const auto &[i,j]:pairs) {
            const Vec d=distance(x[i],x[j]);
            const double r2=square(d);
            if (r2>=rc2) continue;
            const double r=std::sqrt(r2);
            if (r<.5) throw std::runtime_error("Overlap in SAAP dynamics");
            min_separation=std::min(r,min_separation);
            const auto [u,f] = law.evaluate(r);
            const auto [uc,fc] = cut_value;
            const double factor=(f-fc)/r;
            potential+=u-uc+fc*(r-cutoff);
            virial+=factor*r2;
            for (int k=0; k<3; ++k) {
                force[i][k]+=factor*d[k];
                force[j][k]-=factor*d[k];
            }
            tensor(stress,d,factor);
        }
        const double rt2=trimer.high*trimer.high;
        for (const auto &[i,j,k]:triples) {
            const std::array<Vec,3> d{distance(x[i],x[j]),distance(x[i],x[k]),distance(x[j],x[k])};
            const Vec r2{square(d[0]),square(d[1]),square(d[2])};
            if (*std::max_element(r2.begin(),r2.end())>=rt2) continue;
            const Vec r{std::sqrt(r2[0]),std::sqrt(r2[1]),std::sqrt(r2[2])};
            const auto [value,grad] = trimer.evaluate(r);
            potential+=value;
            const std::array<int,3> start{i,i,j},end{j,k,k};
            for (int b=0;b<3;++b) {
                const double f=-grad[b]/r[b];
                virial+=f*r2[b];
                tensor(stress,d[b],f);
                for (int axis=0;axis<3;++axis) {
                    force[start[b]][axis]+=f*d[b][axis];
                    force[end[b]][axis]-=f*d[b][axis];
                }
            }
        }
    }

    static void tensor(std::array<double,5> &s,const Vec &d,double w) {
        s[0]+=w*d[0]*d[1];
        s[1]+=w*d[0]*d[2];
        s[2]+=w*d[1]*d[2];
        s[3]+=w*(d[0]*d[0]-d[1]*d[1])*.5;
        s[4]+=w*(d[0]*d[0]+d[1]*d[1]-2.*d[2]*d[2])/(2.*std::sqrt(3.));
    }

    double kinetic() const {
        double k=0.;
        for (const auto &q:v) k+=square(q);
        return .5*k;
    }

    void rescale() {
        const double a=std::sqrt(1.5*(count-1)*temperature/kinetic());
        for (auto &q:v) for (double &d:q) d*=a;
    }

    void step() {
        double displacement2=0.;
        for (int i=0; i<count; ++i) {
            for (int k=0; k<3; ++k) {
                v[i][k]+=.5*dt*force[i][k];
                const double dx=dt*v[i][k];
                x[i][k]+=dx;
                unwrapped[i][k]+=dx;
                if (x[i][k]>=side) x[i][k]-=side;
                if (x[i][k]<0.) x[i][k]+=side;
            }
            displacement2=std::max(displacement2,square(distance(x[i],start[i])));
        }
        if (displacement2>=.25*skin*skin) rebuild();
        evaluate();
        for (int i=0; i<count; ++i)
            for (int k=0; k<3; ++k) v[i][k]+=.5*dt*force[i][k];
    }

    void record(std::ostream &out, double t) const {
        const double k=kinetic();
        auto w=stress;
        for (const auto &q:v) tensor(w,q,1.);
        out<<t<<','<<2.*k/(3.*(count-1))<<','<<(2.*k+virial)/(3.*volume)
           <<','<<potential/count<<','<<(potential+k)/count;
        for (double z:w) out<<','<<z;
        double msd=0.;
        for (int i=0;i<count;++i)
            for (int k=0;k<3;++k)
                msd+=std::pow(unwrapped[i][k]-origins[i][k],2);
        Vec cm{};
        for (const Vec &velocity:v)
            for (int k=0;k<3;++k)
                cm[k]+=velocity[k]/count;
        out<<','<<msd/count<<','<<std::sqrt(square(cm))<<'\n';
    }
};

int main(int argc,char **argv) {
    try {
        if (argc!=12) throw std::runtime_error("output cells density T coefficients dt rc seed steps equil three_body");
        const int cells=std::stoi(argv[2]), steps=std::stoi(argv[9]), equil=std::stoi(argv[10]);
        const double dt=std::stod(argv[6]);
        Pair law;
        std::ifstream input(argv[5]);
        input >> law.length >> law.energy;
        for (double &x:law.a) input >> x;
        Trimer three;
        input >> three.bohr_per_length >> three.hartree_per_energy >> three.c9 >> three.alpha >> three.low >> three.high;
        for (double &x:three.a) input >> x;
        input >> three.overlap_range;
        for (double &v : three.overlap_amplitudes) input >> v;
        if (!input || law.length<=0. || law.energy<=0.) throw std::runtime_error("Invalid SAAP parameters");
        for (double x:{.8,1.,1.3,2.5}) {
            const double expected = -(law.evaluate(x+1e-5)[0]-law.evaluate(x-1e-5)[0])/2e-5;
            if (std::abs(expected-law.evaluate(x)[1])>1e-6*(1.+std::abs(expected)))
                throw std::runtime_error("SAAP derivative control");
        }
        for (const Vec &r:{Vec{.94,1.13,1.16},Vec{1.13,1.13,1.13},Vec{1.1,1.43,three.overlap_range},
                           Vec{1.1,1.43,three.low+.15},Vec{three.high-.03,three.high-.25,.97}}) {
            const auto [U,g]=three.evaluate(r);
            for (int k=0;k<3;++k) {
                Vec a=r,b=r;
                a[k]+=1e-5;
                b[k]-=1e-5;
                const double d=(three.evaluate(a).first-three.evaluate(b).first)/2e-5;
                if (std::abs(g[k]-d)>2e-7*(1.+std::abs(d))) throw std::runtime_error("EATM derivative control");
            }
        }
        Model m(cells,std::stod(argv[3]),std::stod(argv[4]),law,
                dt,std::stod(argv[7]),std::stoull(argv[8]),three,std::stoi(argv[11])!=0);
        std::ofstream out(argv[1]);
        if (!out) throw std::runtime_error("Cannot open the trajectory output");
        // Velocity rescaling is restricted to equilibration. The production
        // stress correlations are measured on autonomous NVE trajectories.
        for (int i=0; i<equil; ++i) {
            m.step();
            if ((i+1)%50==0) m.rescale();
        }
        m.origins=m.unwrapped;
        out<<std::setprecision(13)<<"t,T,P,U,E,xy,xz,yz,xx_yy,xx_yy_zz,msd,centre_v\n";
        const int stride=std::max(1,int(std::lround(.2/dt)));
        std::ofstream pv(std::string(argv[1])+".pv",std::ios::binary);
 const int stride_pv=std::max(1,int(std::lround(1./dt)));
for (int i=0; i<=steps; ++i) {
            if (i%stride==0) m.record(out,i*dt);
 if (i%stride_pv==0) {for(const auto &x:m.unwrapped) pv.write(reinterpret_cast<const char *>(x.data()),sizeof(double)*3); for(const auto &v:m.v) pv.write(reinterpret_cast<const char *>(v.data()),sizeof(double)*3);}
            if (i<steps) m.step();
        }
        if (!out) throw std::runtime_error("Cannot write the trajectory output");
        std::cout<<m.count<<" particles, "<<steps*dt<<" time units, minimum separation "<<m.min_separation<<", overlap "
                 <<std::setprecision(17)<<three.probe_correction()<<'\n';
    } catch (const std::exception &ex) {
        std::cerr<<ex.what()<<'\n';
        return 1;
    }
}
