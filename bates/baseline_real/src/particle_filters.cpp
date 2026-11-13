#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <algorithm>
#include "qe/params.hpp"
#include "qe/particle_filters.hpp"

using namespace std;


namespace qe{

namespace {

constexpr int MAX_JUMP_COUNT = 10;
constexpr double MIN_VARIANCE = 1e-12;

struct ReturnComponent {
    double log_joint_weight;
    double mean;
    double variance;
};

double logNormalFromVariance(double x, double mean, double variance) {
    const double safe_variance = max(variance, MIN_VARIANCE);
    const double residual = x - mean;
    return -0.5 * (log(2.0 * M_PI * safe_variance)
                 + residual * residual / safe_variance);
}

double logSumExp(const vector<double>& values) {
    const double max_value = *max_element(values.begin(), values.end());
    double scaled_sum = 0.0;
    for (double value : values) scaled_sum += exp(value - max_value);
    return max_value + log(scaled_sum);
}

vector<ReturnComponent> returnComponents(const BatesPParams& P, double r,
                                         double v, double dt) {
    const double vp = max(v, 1e-8);
    const double jump_intensity = max(P.JumpIntensityP, 0.0);
    const double jump_variance = P.JumpVolatility * P.JumpVolatility;
    const double jump_compensator = expm1(P.JumpMeanP + 0.5 * jump_variance);
    const double poisson_mean = jump_intensity * dt;
    const double base_mean =
        (P.mu - 0.5 * vp - jump_intensity * jump_compensator) * dt;
    const double base_variance = max(vp * dt, MIN_VARIANCE);

    vector<ReturnComponent> components;
    components.reserve(MAX_JUMP_COUNT + 1);
    for (int jump_count = 0; jump_count <= MAX_JUMP_COUNT; ++jump_count) {
        double log_poisson_weight;
        if (poisson_mean == 0.0) {
            if (jump_count != 0) continue;
            log_poisson_weight = 0.0;
        } else {
            log_poisson_weight = -poisson_mean
                + jump_count * log(poisson_mean)
                - lgamma(static_cast<double>(jump_count) + 1.0);
        }

        const double mean = base_mean + jump_count * P.JumpMeanP;
        const double variance =
            max(base_variance + jump_count * jump_variance, MIN_VARIANCE);
        components.push_back({
            log_poisson_weight + logNormalFromVariance(r, mean, variance),
            mean,
            variance
        });
    }
    return components;
}

double componentConditionalVarianceMean(const BatesPParams& P, double v,
                                        double r, double dt,
                                        const ReturnComponent& component) {
    const double vp = max(v, 1e-8);
    const double variance_drift = vp + P.kappaP * (P.thetaP - vp) * dt;
    const double covariance = P.rho * P.xi * vp * dt;
    return variance_drift
         + covariance / component.variance * (r - component.mean);
}

double componentConditionalVariance(const BatesPParams& P, double v, double dt,
                                    const ReturnComponent& component) {
    const double vp = max(v, 1e-8);
    const double covariance = P.rho * P.xi * vp * dt;
    const double unconditional_variance = P.xi * P.xi * vp * dt;
    return max(unconditional_variance
             - covariance * covariance / component.variance, MIN_VARIANCE);
}

double sampleConditionalVariance(const BatesPParams& P, double v, double r,
                                 double dt, double* observation_log_likelihood) {
    static thread_local mt19937 gen(random_device{}());
    static thread_local normal_distribution<double> normal(0.0, 1.0);

    const vector<ReturnComponent> components = returnComponents(P, r, v, dt);
    vector<double> log_weights;
    log_weights.reserve(components.size());
    for (const auto& component : components)
        log_weights.push_back(component.log_joint_weight);
    const double normalizer = logSumExp(log_weights);
    if (observation_log_likelihood != nullptr)
        *observation_log_likelihood = normalizer;

    vector<double> posterior_weights;
    posterior_weights.reserve(components.size());
    for (double log_weight : log_weights)
        posterior_weights.push_back(exp(log_weight - normalizer));
    discrete_distribution<size_t> component_distribution(
        posterior_weights.begin(), posterior_weights.end());
    const ReturnComponent& component = components[component_distribution(gen)];

    const double mean = componentConditionalVarianceMean(P, v, r, dt, component);
    const double variance = componentConditionalVariance(P, v, dt, component);
    return mean + sqrt(variance) * normal(gen);
}

} // namespace


double observationLikelihood(const BatesPParams& P, double r, double v, double dt) {
    const vector<ReturnComponent> components = returnComponents(P, r, v, dt);
    vector<double> log_weights;
    log_weights.reserve(components.size());
    for (const auto& component : components)
        log_weights.push_back(component.log_joint_weight);
    return logSumExp(log_weights);
}

double qLikelihood(const BatesPParams& P,double v,double r,double dt,double new_v){
    const vector<ReturnComponent> components = returnComponents(P, r, v, dt);
    vector<double> component_log_weights;
    component_log_weights.reserve(components.size());
    for (const auto& component : components)
        component_log_weights.push_back(component.log_joint_weight);
    const double observation_log_likelihood = logSumExp(component_log_weights);

    vector<double> proposal_terms;
    proposal_terms.reserve(components.size());
    for (const auto& component : components) {
        const double mean = componentConditionalVarianceMean(P, v, r, dt, component);
        const double variance = componentConditionalVariance(P, v, dt, component);
        proposal_terms.push_back(
            component.log_joint_weight - observation_log_likelihood
            + logNormalFromVariance(new_v, mean, variance));
    }
    return logSumExp(proposal_terms);
}

double priorLikelihood(const BatesPParams& P,double v,double dt,double new_v){
    const double vp = max(v, 1e-8);
    const double mean = vp + P.kappaP * (P.thetaP - vp) * dt;
    const double variance = P.xi * P.xi * vp * dt;
    return logNormalFromVariance(new_v, mean, variance);
}



double transitionFunction(const BatesPParams& P, double v,double r, double dt){
    return sampleConditionalVariance(P, v, r, dt, nullptr);
}

vector<int> systematicResample(const vector<double>& W, std::mt19937& gen) {
    
    int N = W.size();
    vector<int> idx(N);

    vector<double> C(N);
    C[0] = W[0];
    for (int i = 1; i < N; i++)
        C[i] = C[i-1] + W[i];
    C.back() = 1.0;

    uniform_real_distribution<double> U(0.0, 1.0 / N);
    double u0 = U(gen);

    int i = 0;

    for (int j = 0; j < N; j++) {

        double u = u0 + (double)j / N;

        while (i < N - 1 && u > C[i])
            i++;

        idx[j] = i;
    }

    return idx;
}


filterValues ParticleFilter(const BatesPParams& P, double v0_actual,const PPath& ppath,double dt,int N){
    if (N <= 0) throw invalid_argument("ParticleFilter requires at least one particle.");
    if (ppath.returns.empty()) throw invalid_argument("ParticleFilter requires at least one return.");

    static thread_local std::mt19937 gen(std::random_device{}());
    double eps = 0.5;
    // double v0_actual = P.v0;
    int resample_count = 0;
    uniform_real_distribution<double> uniform(v0_actual*(1-eps), v0_actual*(1+eps));
    
    vector<double>logReturns = ppath.returns;
    int n = logReturns.size();

    vector<double>logW (N,0.0);
    vector<double>v_chain (N,0.0);
    vector<double>W(N,0.0);
    vector<double>baseW(N,1.0 / N);
    vector<double>new_v_chain(N,0.0);
    vector<vector<double>> particles(n + 1, vector<double>(N, 0.0));
    vector<vector<int>> ancestors(n + 1, vector<int>(N, -1));
    
    

    for (int j = 0; j < N; j++){
        v_chain[j] = uniform(gen); 
        ancestors[0][j] = -1;
        particles[0][j] = v_chain[j];
    }

    //double v = uniform(gen);
    double logLik = 0.0;

    for (int i = 0; i < n; i++){
        double r = logReturns[i]; //logReturns[i] --> r_t
        for (int j = 0; j < N; j++){
            double v = v_chain[j]; // v_t for all chain

            double observation_log_likelihood;
            double new_v = sampleConditionalVariance(
                P, v, r, dt, &observation_log_likelihood);
            new_v_chain[j] = new_v; // store new transitions
            // transitionFunction samples from p(v_{t+1}|v_t,r_t), so the
            // incremental weight is the predictive Bates return density.
            logW[j] = observation_log_likelihood
                    + (baseW[j] > 0.0
                        ? log(baseW[j])
                        : -numeric_limits<double>::infinity());
        }
        double maxLogW = *max_element(logW.begin(), logW.end());
        double sumW = 0.0;
        for(int j = 0; j < N; j++){
            W[j] = exp(logW[j] - maxLogW);
            sumW += W[j];
        }
        logLik += maxLogW + log(sumW);

        double ess = 0.0;
        for(int j = 0; j < N; j++){
            W[j] = W[j]/sumW;
            ess += W[j] * W[j];
        }
        ess = 1/ess;
        if (ess < 0.3 * N){
            resample_count ++;
            //cout<<"Resampled at step:"<<i<<endl;
            vector<int> idx = systematicResample(W,gen);
            for(int j = 0; j < N;j++){
                v_chain[j] = new_v_chain[idx[j]];
                particles[i + 1][j] = v_chain[j];
                ancestors[i + 1][j] = idx[j];
                W[j] = 1.0/N;
            }    
        }
        else{
            for (int j = 0; j < N;j++){
                v_chain[j] = new_v_chain[j];
                particles[i + 1][j] = v_chain[j];
                ancestors[i + 1][j] = j;
            }
            
        }
        baseW = W;
    }
    //cout<<"Resample Count:"<<resample_count<<endl;
    filterValues filter;
    filter.resample_count = resample_count;
    filter.likelihood = logLik;
    filter.ancestors = ancestors;
    filter.particles = particles;
    filter.v_chain = v_chain;
    filter.W = W;
    return filter;
    
}


vector<double>  ancestralSampling(filterValues& filter, mt19937& gen){
    
    int N = filter.particles[0].size();
    int T = filter.particles.size();
    
    vector<double>sampledPath(T,0.0);

    discrete_distribution<>dist(filter.W.begin(),filter.W.end());
    int k = dist(gen);
    //cout<<"k:"<<k<<endl;
    for(int t = T-1; t >= 0; t--){
        sampledPath[t] = filter.particles[t][k];
        k = filter.ancestors[t][k];
        if ((k < 0) && (t > 0)) break;
    }
    return sampledPath;
}

void writeSampledPaths(vector<vector<double>>& sampledPaths,string filename){
    ofstream file(filename);
    int numSampledPaths = sampledPaths.size();
    int T = sampledPaths[0].size();
    for(int i = 0; i < numSampledPaths; i++){
        for(int t = 0; t < T; t++){
            file << sampledPaths[i][t];

            if(t < T-1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}
}//namespace qe
