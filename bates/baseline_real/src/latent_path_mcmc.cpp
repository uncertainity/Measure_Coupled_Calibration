#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <iomanip>
#include <limits>
#include <Eigen/Cholesky>
#include "qe/params.hpp"
#include "qe/path.hpp"
#include "qe/latent_path_mcmc.hpp"
#include "qe/particle_filters.hpp"


using namespace std;
using namespace Eigen;


namespace qe {

namespace {

constexpr int MAX_JUMP_COUNT = 10;
constexpr double MIN_VARIANCE = 1e-12;

double logSumExp(const vector<double>& values) {
    const double max_value = *max_element(values.begin(), values.end());
    double scaled_sum = 0.0;
    for (double value : values) scaled_sum += exp(value - max_value);
    return max_value + log(scaled_sum);
}

double logPoissonWeight(double poisson_mean, int jump_count) {
    if (poisson_mean == 0.0)
        return jump_count == 0 ? 0.0 : -numeric_limits<double>::infinity();
    return -poisson_mean
         + jump_count * log(poisson_mean)
         - lgamma(static_cast<double>(jump_count) + 1.0);
}

} // namespace

double getPathLikelihood(const BatesPParams& P, const PPath& ppath,const vector<double>& vProxy,double dt){
    const size_t n = ppath.returns.size();
    if (n == 0) throw invalid_argument("Path likelihood requires at least one return.");
    if (vProxy.size() < n + 1)
        throw invalid_argument("Variance proxy path must have one more value than returns.");

    const double jump_intensity = max(P.JumpIntensityP, 0.0);
    const double jump_variance = P.JumpVolatility * P.JumpVolatility;
    const double jump_compensator = expm1(P.JumpMeanP + 0.5 * jump_variance);
    const double poisson_mean = jump_intensity * dt;
    double logLikelihood = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double vp = max(vProxy[i], 1e-8);
        const double observed_return = ppath.returns[i];
        const double observed_variance_change = vProxy[i + 1] - vProxy[i];
        const double base_return_mean =
            (P.mu - 0.5 * vp - jump_intensity * jump_compensator) * dt;
        const double variance_change_mean = P.kappaP * (P.thetaP - vp) * dt;
        const double variance_change_variance =
            max(P.xi * P.xi * vp * dt, MIN_VARIANCE);
        const double covariance = P.rho * P.xi * vp * dt;

        vector<double> component_log_densities;
        component_log_densities.reserve(MAX_JUMP_COUNT + 1);
        for (int jump_count = 0; jump_count <= MAX_JUMP_COUNT; ++jump_count) {
            const double log_poisson_weight = logPoissonWeight(poisson_mean, jump_count);
            if (!isfinite(log_poisson_weight)) continue;

            const double return_mean = base_return_mean + jump_count * P.JumpMeanP;
            const double return_variance =
                max(vp * dt + jump_count * jump_variance, MIN_VARIANCE);
            const double determinant =
                max(return_variance * variance_change_variance
                  - covariance * covariance, MIN_VARIANCE * MIN_VARIANCE);
            const double return_residual = observed_return - return_mean;
            const double variance_residual =
                observed_variance_change - variance_change_mean;
            const double quadratic = (
                variance_change_variance * return_residual * return_residual
                - 2.0 * covariance * return_residual * variance_residual
                + return_variance * variance_residual * variance_residual
            ) / determinant;
            const double log_bivariate_normal =
                -log(2.0 * M_PI) - 0.5 * log(determinant) - 0.5 * quadratic;
            component_log_densities.push_back(
                log_poisson_weight + log_bivariate_normal);
        }
        logLikelihood += logSumExp(component_log_densities);
    }
    return logLikelihood;
}

// [TODO] - are these constrainsts correct
BatesPParams xToParamsMCMC(VectorXd& x){
    BatesPParams P;
    //[TODO] -  was earlier exp
    P.mu = x[0];
    P.kappaP = exp(x[1]);
    P.thetaP = exp(x[2]);
    P.xi = exp(x[3]);
    P.rho = tanh(x[4]);
    P.JumpIntensityP = exp(x[5]);
    P.JumpMeanP = x[6];
    P.JumpVolatility = exp(x[7]);
    return P;
}

// [TODO] - make sure this is aligned with the above function
VectorXd ParamstoxMCMC(BatesPParams& P){
    VectorXd x;
    x[0] = P.mu;
    x[1] = log(P.kappaP);
    x[2] = log(P.thetaP);
    x[3] = log(P.xi);
    x[4] = atanh(P.rho);
    x[5] = log(P.JumpIntensityP);
    x[6] = P.JumpMeanP;
    x[7] = log(P.JumpVolatility);

    return x;
}

static inline double log_norm_prior(double x,double m, double s){
    double z = (m - x)/s;
    return -0.5 * log(2*M_PI) - log(s) - 0.5 * z*z; 

}

double logPrior(VectorXd x,double vbar){
    double lp = 0.0;
    lp += log_norm_prior(x[0], 0.0, 1.0);
    lp += log_norm_prior(x[1], std::log(1.0), 1.0);
    lp += log_norm_prior(x[2], std::log(vbar), 1.0);
    lp += log_norm_prior(x[3], std::log(0.3), 1.0);
    lp += log_norm_prior(x[4], 0.0, 1.5);

    // [TODO] - if these are fine
    // Bates jump params
    lp += log_norm_prior(x[5], std::log(0.5), 1.0);    // log(JumpIntensity)
    lp += log_norm_prior(x[6], -0.05, 0.25);           // JumpMean
    lp += log_norm_prior(x[7], std::log(0.2), 1.0);    // log(JumpVolatility)
    return lp;
}

double logPosterior(VectorXd x,double vbar, const PPath& ppath,const vector<double>& vProxy,double dt){
    double lp = logPrior(x,vbar);
    BatesPParams P = xToParamsMCMC(x);
    double ll = getPathLikelihood(P,ppath,vProxy,dt);
    return ll + lp;  
}

double logPosterior(VectorXd x,double vbar, const PPath& ppath,double v0_actual,double dt,int N){
    double lp = logPrior(x,vbar);
    BatesPParams P = xToParamsMCMC(x);
    //double ll = getPathLikelihood(P,ppath);
    filterValues f = ParticleFilter(P,v0_actual,ppath,dt,N);
    double ll = f.likelihood;
    return ll + lp;  
}


VectorXd randn_vec(mt19937& rng, int d){
    static thread_local normal_distribution<double>nd(0.0,1.0);
    VectorXd z(d);
    for (int i = 0; i < d; i++) z[i] = nd(rng);
    return z;
}

// The metropolis needs to retur a chain

vector <VectorXd> AdaptiveMetropolis(const PPath& ppath, const vector<double>& vProxy,const VectorXd& x0, int n_iters, double dt,int adapt_start = 1000, int adapt_every = 50,double jitter = 1e-5){
    const int d = x0.size();
    mt19937 rng(123);
    double vbar = 0.0;
    if (vProxy.empty()) throw invalid_argument("Variance proxy path cannot be empty.");
    for (double v : vProxy) vbar += v;

    vbar = vbar / static_cast<double>(vProxy.size());
    double s = 2.0/sqrt((double)d);
    MatrixXd C = MatrixXd::Identity(d,d);
    VectorXd x = x0;
    VectorXd mean = x;
    MatrixXd Sigma = MatrixXd::Identity(d,d);
    vector<VectorXd>chain;
    chain.reserve(n_iters);
    uniform_real_distribution<double> unif(0.0, 1.0);
    double logp = logPosterior(x,vbar,ppath,vProxy,dt);
    int accepted = 0;
    for(int it = 1; it < n_iters;it++){
        LLT<MatrixXd>llt(Sigma);
        if (llt.info() != Eigen::Success) {
            Sigma = Eigen::MatrixXd::Identity(d,d); // or reset to I
            llt.compute(Sigma);
        }
        MatrixXd L = llt.matrixL();
        VectorXd z = randn_vec(rng,d);
        VectorXd xnew = x + s*(L*z);
        double logpnew = logPosterior(xnew,vbar,ppath,vProxy,dt);
        double alpha_uniform = unif(rng);
        if (log(alpha_uniform) < (logpnew - logp)){
            x = xnew;
            logp = logpnew;
            accepted ++;
        }
        chain.push_back(x);
        if (it >= adapt_start){
            VectorXd x_it = x;
            VectorXd delta = x_it - mean;
            mean = mean + delta/(it - adapt_start + 1);
            VectorXd delta2 = x_it - mean; //we redeclare delta because of some precision issue
            C += delta*delta2.transpose();

            if((it % adapt_every) == 0){
                MatrixXd empCov = C/max(1,it - adapt_start);
                Sigma = empCov + jitter * MatrixXd::Identity(d,d);
            }

        }
    }
    double acRate = (double)accepted/(double)n_iters;
    cout<<"Acceptance Rate:"<<acRate<<endl;
    return chain;
}


vector <VectorXd> AdaptiveMetropolis(const PPath& ppath, const VectorXd& x0, int n_iters, double v0_actual,double vbar_prior,double dt,int num_particles,int adapt_start = 1000, int adapt_every = 50,double jitter = 1e-5){
    const int d = x0.size();
    mt19937 rng(123);
    if (!(vbar_prior > 0.0) || !isfinite(vbar_prior)) {
        throw invalid_argument("PMCMC variance prior anchor must be positive and finite.");
    }
    double vbar = vbar_prior;
    double s = 2.0/sqrt((double)d);
    MatrixXd C = MatrixXd::Identity(d,d);
    VectorXd x = x0;
    VectorXd mean = x;
    MatrixXd Sigma = MatrixXd::Identity(d,d);
    vector<VectorXd>chain;
    chain.reserve(n_iters);
    uniform_real_distribution<double> unif(0.0, 1.0);
    double logp = logPosterior(x,vbar,ppath,v0_actual,dt,num_particles);
    int accepted = 0;
    for(int it = 1; it < n_iters;it++){
        if(it % 100 == 0) cout<<it<<" iters of MCMC is done."<<endl;
        LLT<MatrixXd>llt(Sigma);
        if (llt.info() != Eigen::Success) {
            Sigma = Eigen::MatrixXd::Identity(d,d); // or reset to I
            llt.compute(Sigma);
        }
        MatrixXd L = llt.matrixL();
        VectorXd z = randn_vec(rng,d);
        VectorXd xnew = x + s*(L*z);
        double logpnew = logPosterior(xnew,vbar,ppath,v0_actual,dt,num_particles);
        double alpha_uniform = unif(rng);
        if (log(alpha_uniform) < (logpnew - logp)){
            x = xnew;
            logp = logpnew;
            accepted ++;
        }
        chain.push_back(x);
        if (it >= adapt_start){
            VectorXd x_it = x;
            VectorXd delta = x_it - mean;
            mean = mean + delta/(it - adapt_start + 1);
            VectorXd delta2 = x_it - mean; //we redeclare delta because of some precision issue
            C += delta*delta2.transpose();

            if((it % adapt_every) == 0){
                MatrixXd empCov = C/max(1,it - adapt_start);
                Sigma = empCov + jitter * MatrixXd::Identity(d,d);
            }

        }
    }
    double acRate = (double)accepted/(double)n_iters;
    cout<<"Acceptance Rate:"<<acRate<<endl;
    return chain;
}



void chainStatistics(vector<VectorXd>& chain,int burn_in,BatesPParams& meanP,BatesPParams& varP){
    if (chain.empty()) throw invalid_argument("Cannot summarize an empty MCMC chain.");
    burn_in = min<int>(burn_in, static_cast<int>(chain.size()) - 1);
    const int sample_count = static_cast<int>(chain.size()) - burn_in;

    double mu_mean = 0.0;
    double kappaP_mean = 0.0;
    double thetaP_mean = 0.0;
    double xi_mean = 0.0;
    double rho_mean = 0.0;
    double JumpIntensity_mean = 0.0;
    double JumpMean_mean = 0.0;
    double JumpVolatility_mean = 0.0;
    
    double mu_var = 0.0;
    double kappaP_var = 0.0;
    double thetaP_var = 0.0;
    double xi_var = 0.0;
    double rho_var = 0.0;
    double JumpIntensity_var = 0.0;
    double JumpMean_var = 0.0;
    double JumpVolatility_var = 0.0;
    
    for (int i = burn_in; i < chain.size();i++){
        BatesPParams tmp;
        tmp = xToParamsMCMC(chain[i]);
        mu_mean += tmp.mu;
        mu_var += (tmp.mu * tmp.mu);
        
        kappaP_mean += tmp.kappaP;
        kappaP_var += (tmp.kappaP * tmp.kappaP);

        thetaP_mean += tmp.thetaP;
        thetaP_var += (tmp.thetaP * tmp.thetaP);

        xi_mean += tmp.xi;
        xi_var += (tmp.xi * tmp.xi);

        rho_mean += tmp.rho;
        rho_var += (tmp.rho * tmp.rho);        

        JumpIntensity_mean += tmp.JumpIntensityP;
        JumpIntensity_var += (tmp.JumpIntensityP * tmp.JumpIntensityP);

        JumpMean_mean += tmp.JumpMeanP;
        JumpMean_var += (tmp.JumpMeanP * tmp.JumpMeanP); 

        JumpVolatility_mean += tmp.JumpVolatility;
        JumpVolatility_var += (tmp.JumpVolatility * tmp.JumpVolatility);        
    }
    mu_mean = mu_mean/sample_count;
    kappaP_mean = kappaP_mean/sample_count;
    thetaP_mean = thetaP_mean/sample_count;
    xi_mean = xi_mean/sample_count;
    rho_mean = rho_mean/sample_count;
    JumpIntensity_mean = JumpIntensity_mean/sample_count;
    JumpMean_mean = JumpMean_mean/sample_count;
    JumpVolatility_mean = JumpVolatility_mean/sample_count;
    
    mu_var = mu_var/sample_count - mu_mean * mu_mean;
    kappaP_var = kappaP_var/sample_count - kappaP_mean * kappaP_mean;
    thetaP_var = thetaP_var/sample_count - thetaP_mean * thetaP_mean;
    xi_var = xi_var/sample_count - xi_mean * xi_mean;
    rho_var = rho_var/sample_count - rho_mean * rho_mean;
    JumpIntensity_var = JumpIntensity_var/sample_count - JumpIntensity_mean * JumpIntensity_mean;
    JumpMean_var = JumpMean_var/sample_count - JumpMean_mean * JumpMean_mean;
    JumpVolatility_var = JumpVolatility_var/sample_count - JumpVolatility_mean * JumpVolatility_mean;

    meanP.mu = mu_mean;
    meanP.kappaP = kappaP_mean;
    meanP.thetaP = thetaP_mean;
    meanP.xi = xi_mean;
    meanP.rho = rho_mean;
    meanP.JumpIntensityP = JumpIntensity_mean;
    meanP.JumpMeanP = JumpMean_mean;
    meanP.JumpVolatility = JumpVolatility_mean;

    varP.mu = mu_var;
    varP.kappaP = kappaP_var;
    varP.thetaP = thetaP_var;
    varP.xi = xi_var;
    varP.rho = rho_var;
    varP.JumpIntensityP = JumpIntensity_var;
    varP.JumpMeanP = JumpMean_var;
    varP.JumpVolatility = JumpVolatility_var;
}


void mcmcOverLatent(BatesPParams& P, PPath& ppath, vector<double>& vProxy,VectorXd x0,double dt,BatesPParams& meanP,BatesPParams& varP,int n_iters = 5000){

    double logLikelihood = getPathLikelihood(P,ppath,vProxy,dt);
    cout<<"Likelihood for actual P:"<<logLikelihood<<endl;
    // mt19937 gen(123);
    // double eps = 0.5;
    // VectorXd x0(5);
    // uniform_real_distribution<> dist_mu(actualP.mu * (1-eps), actualP.mu * (1+eps));
    // uniform_real_distribution<> dist_kappaP(actualP.kappaP * (1-eps), actualP.kappaP * (1+eps));
    // uniform_real_distribution<> dist_thetaP(actualP.thetaP * (1-eps), actualP.thetaP * (1+eps));
    // uniform_real_distribution<> dist_xi(actualP.xi * (1-eps), actualP.xi * (1+eps));
    // uniform_real_distribution<> dist_rho(max(-0.95, actualP.rho - 0.4),min(-0.05, actualP.rho + 0.4));

    // x0[0] = dist_mu(gen);
    // x0[1] = dist_kappaP(gen);
    // x0[2] = dist_thetaP(gen);
    // x0[3] = dist_xi(gen);
    // x0[4] = dist_rho(gen);

    // int n_iters = 5000;
    vector<VectorXd>chain = AdaptiveMetropolis(ppath,vProxy,x0,n_iters,dt);
    cout<<"chain is built;"<<endl;

    chainStatistics(chain,1000,meanP,varP);
    // cout<<meanP<<endl;
    // cout<<varP<<endl;
    
}

void pmcmcOverLatent(BatesPParams& P, PPath& ppath,VectorXd x0,double dt,BatesPParams& meanP,BatesPParams& varP,int n_iters = 5000,int num_particles = 1500){

    //static thread_local std::mt19937 gen(69);
    // double logLikelihood = getPathLikelihood(actualP,ppath);
    // cout<<"Actual Likelihood with true P path:"<<logLikelihood<<endl;

    //mt19937 gen(123);
    double eps = 0.5;

    // VectorXd x0(5);
    // uniform_real_distribution<> dist_mu(actualP.mu * (1-eps), actualP.mu * (1+eps));
    // uniform_real_distribution<> dist_kappaP(actualP.kappaP * (1-eps), actualP.kappaP * (1+eps));
    // uniform_real_distribution<> dist_thetaP(actualP.thetaP * (1-eps), actualP.thetaP * (1+eps));
    // uniform_real_distribution<> dist_xi(actualP.xi * (1-eps), actualP.xi * (1+eps));
    // uniform_real_distribution<> dist_rho(max(-0.95, actualP.rho - 0.4),min(-0.05, actualP.rho + 0.4));

    // x0[0] = dist_mu(gen);
    // x0[1] = dist_kappaP(gen);
    // x0[2] = dist_thetaP(gen);
    // x0[3] = dist_xi(gen);
    // x0[4] = dist_rho(gen);

    double v0_actual = P.v0;
    // int n_iters = 5000;
    // int num_particles = 1500;
    vector<VectorXd>chain = AdaptiveMetropolis(ppath,x0,n_iters,v0_actual,P.thetaP,dt,num_particles); // 5000 --> n_iters (no of MH steps)
    cout<<"chain is built;"<<endl;
    // BatesPParams meanP;
    // BatesPParams varP;
    
    chainStatistics(chain,1000,meanP,varP); // 1000 is burn in
    // cout<<meanP<<endl;
    // cout<<varP<<endl;
}
}//namespace qe;
