// writted by codex
// [TODO] - review

#pragma once

#include <ql/quantlib.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <complex>

namespace qe {

double batesCallPrice(
    double spot,
    double strike,
    double maturityYears,
    double r,
    double q,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double jumpIntensity,
    double jumpMean,
    double jumpVolatility,
    const QuantLib::Date& evalDate) {

    using namespace std;

    if (!(spot > 0.0) || !(strike > 0.0)) {
        throw std::invalid_argument("spot and strike must be > 0");
    }
    if (!(maturityYears > 0.0)) {
        throw std::invalid_argument("maturityYears must be > 0");
    }
    if (!(v0 >= 0.0) || !(kappa > 0.0) || !(theta >= 0.0) || !(xi >= 0.0)) {
        throw std::invalid_argument("invalid Bates variance parameters");
    }
    if (!(jumpIntensity >= 0.0) || !(jumpVolatility >= 0.0)) {
        throw std::invalid_argument("invalid Bates jump parameters");
    }

    const double T = std::max(1.0 / (365.0 * 24.0), maturityYears);
    const double clampedRho = std::clamp(rho, -0.999, 0.999);

    const double x = std::log(spot);
    const double logK = std::log(strike);

    const complex<double> I(0.0, 1.0);

    auto heston_cf = [&](complex<double> u) {
        complex<double> d = std::sqrt(
            std::pow(clampedRho * xi * I * u - kappa, 2.0)
            + xi * xi * (I * u + u * u)
        );

        complex<double> g =
            (kappa - clampedRho * xi * I * u - d)
            / (kappa - clampedRho * xi * I * u + d);

        complex<double> C =
            (r - q) * I * u * T
            + (kappa * theta / (xi * xi))
            * (
                (kappa - clampedRho * xi * I * u - d) * T
                - 2.0 * std::log((1.0 - g * std::exp(-d * T)) / (1.0 - g))
            );

        complex<double> D =
            ((kappa - clampedRho * xi * I * u - d) / (xi * xi))
            * ((1.0 - std::exp(-d * T)) / (1.0 - g * std::exp(-d * T)));

        return std::exp(C + D * v0 + I * u * x);
    };

    auto bates_cf = [&](complex<double> u) {
        double jumpCompensator =
            std::exp(jumpMean + 0.5 * jumpVolatility * jumpVolatility) - 1.0;

        complex<double> jump_cf =
            std::exp(
                jumpIntensity * T
                * (
                    std::exp(
                        I * u * jumpMean
                        - 0.5 * jumpVolatility * jumpVolatility * u * u
                    )
                    - 1.0
                    - I * u * jumpCompensator
                )
            );

        return heston_cf(u) * jump_cf;
    };

    auto integrand = [&](double u, int j) {
        complex<double> uj(u, 0.0);

        complex<double> numerator;
        complex<double> denominator;

        if (j == 1) {
            numerator = std::exp(-I * uj * logK) * bates_cf(uj - I);
            denominator = I * uj * bates_cf(-I);
        } else {
            numerator = std::exp(-I * uj * logK) * bates_cf(uj);
            denominator = I * uj;
        }

        return std::real(numerator / denominator);
    };

    auto integrate = [&](int j) {
        const double upper = 150.0;
        const int N = 10000;
        const double du = upper / N;

        double sum = 0.0;

        for (int n = 1; n < N; ++n) {
            double u = n * du;
            sum += integrand(u, j);
        }

        return sum * du;
    };

    double P1 = 0.5 + integrate(1) / M_PI;
    double P2 = 0.5 + integrate(2) / M_PI;

    double call =
        spot * std::exp(-q * T) * P1
        - strike * std::exp(-r * T) * P2;

    return std::max(0.0, call);
}

}
