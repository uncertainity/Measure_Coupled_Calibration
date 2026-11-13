#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct MarketPoint {
    string date;
    int hour;
    double spot;
};

struct Summary {
    string method;
    size_t parameter_rows;
    size_t scored_rows;
    double mean_nll;
    double median_nll;
    double total_nll;
};

struct ParameterRow {
    double mu = numeric_limits<double>::quiet_NaN();
    double kappa = numeric_limits<double>::quiet_NaN();
    double theta = numeric_limits<double>::quiet_NaN();
    double vol_of_vol = numeric_limits<double>::quiet_NaN();
    double rho = numeric_limits<double>::quiet_NaN();
    double variance = numeric_limits<double>::quiet_NaN();
    double jump_intensity = numeric_limits<double>::quiet_NaN();
    double jump_mean = numeric_limits<double>::quiet_NaN();
    double jump_volatility = numeric_limits<double>::quiet_NaN();
    double observed_return = numeric_limits<double>::quiet_NaN();
    double one_step_nll = numeric_limits<double>::quiet_NaN();
};

constexpr double HOURLY_DT = 1.0 / (365.0 * 24.0);
constexpr int MAX_JUMP_COUNT = 10;

string toISODate(const string& input) {
    // Expected: "October 23rd, 2023"
    stringstream ss(input);

    string month, day, year;
    ss >> month >> day >> year;

    if (!year.empty() && year.back() == ',')
        year.pop_back();

    while (!day.empty() && !isdigit(day.back()))
        day.pop_back();

    static const unordered_map<string, int> monthMap = {
        {"January",1}, {"February",2}, {"March",3}, {"April",4},
        {"May",5}, {"June",6}, {"July",7}, {"August",8},
        {"September",9}, {"October",10}, {"November",11}, {"December",12}
    };

    ostringstream out;
    out << year << '-'
        << setw(2) << setfill('0') << monthMap.at(month)
        << '-'
        << setw(2) << setfill('0') << stoi(day);

    return out.str();
}

vector<string> splitCSVLine(const string& line) {
    stringstream ss(line);
    string token;
    vector<string> fields;

    while (getline(ss, token, ',')) {
        if (!token.empty() && token.back() == '\r') token.pop_back();
        fields.push_back(token);
    }

    return fields;
}

size_t requiredColumn(const vector<string>& header, const vector<string>& candidates,
                      const string& filename) {
    for (const auto& candidate : candidates) {
        const auto it = find(header.begin(), header.end(), candidate);
        if (it != header.end()) return static_cast<size_t>(it - header.begin());
    }

    ostringstream message;
    message << "Missing required column ";
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i != 0) message << " or ";
        message << candidates[i];
    }
    message << " in " << filename;
    throw runtime_error(message.str());
}

double readGeneratedCSVValue(const vector<string>& fields, size_t header_index,
                             const string& filename) {
    // QuantLib dates are currently emitted as e.g. "October 23rd, 2023"
    // without CSV quoting, so every non-date data field is shifted by one.
    const size_t data_index = header_index + 1;
    if (data_index >= fields.size() || fields[data_index].empty())
        throw runtime_error("Missing numeric value in " + filename);
    return stod(fields[data_index]);
}

vector<MarketPoint> readMarketPath(const string& filename) {
    ifstream file(filename);
    if (!file)
        throw runtime_error("Could not open market path file: " + filename);

    string line;
    getline(file, line);

    vector<MarketPoint> market_points;
    while (getline(file, line)) {
        vector<string> fields = splitCSVLine(line);
        if (fields.size() < 3)
            continue;

        market_points.push_back({fields[0], stoi(fields[1]), stod(fields[2])});
    }

    return market_points;
}

vector<double> getReturns(const vector<MarketPoint>& market_points) {
    vector<double> returns;

    for (size_t i = 1; i < market_points.size(); i++)
        returns.push_back(log(market_points[i].spot / market_points[i - 1].spot));

    return returns;
}

void readJointParams(const string& filename, bool eape, vector<string>& dates,
                     vector<int>& hours, vector<ParameterRow>& params) {
    ifstream file(filename);
    if (!file) throw runtime_error("Could not open parameter file: " + filename);
    string line;
    getline(file, line);
    const vector<string> header = splitCSVLine(line);
    const size_t mu_col = requiredColumn(header, {eape ? "mu_mean" : "mu"}, filename);
    const size_t kappa_col = requiredColumn(header, {eape ? "kappa_mean" : "kappa"}, filename);
    const size_t theta_col = requiredColumn(header, {eape ? "theta_mean" : "theta"}, filename);
    const size_t volvol_col = requiredColumn(
        header, {eape ? "vol-of-vol_mean" : "vol-of-vol"}, filename);
    const size_t rho_col = requiredColumn(header, {eape ? "rho_mean" : "rho"}, filename);
    const size_t variance_col = requiredColumn(header, {eape ? "v_t_mean" : "v_t"}, filename);
    const size_t jump_intensity_col = requiredColumn(
        header, {eape ? "jump_intensity_mean" : "jump_intensity"}, filename);
    const size_t jump_mean_col = requiredColumn(
        header, {eape ? "jump_mean_mean" : "jump_mean"}, filename);
    const size_t jump_volatility_col = requiredColumn(
        header, {eape ? "jump_volatility_mean" : "jump_volatility"}, filename);

    while (getline(file, line)) {
        const auto fields = splitCSVLine(line);
        if (fields.size() < header.size() + 1)
            throw runtime_error("Malformed joint parameter row in " + filename);
        dates.push_back(toISODate(fields[0] + "," + fields[1]));
        hours.push_back(static_cast<int>(readGeneratedCSVValue(fields, 1, filename)));

        ParameterRow row;
        row.mu = readGeneratedCSVValue(fields, mu_col, filename);
        row.kappa = readGeneratedCSVValue(fields, kappa_col, filename);
        row.theta = readGeneratedCSVValue(fields, theta_col, filename);
        row.vol_of_vol = readGeneratedCSVValue(fields, volvol_col, filename);
        row.rho = readGeneratedCSVValue(fields, rho_col, filename);
        row.variance = readGeneratedCSVValue(fields, variance_col, filename);
        row.jump_intensity = readGeneratedCSVValue(fields, jump_intensity_col, filename);
        row.jump_mean = readGeneratedCSVValue(fields, jump_mean_col, filename);
        row.jump_volatility = readGeneratedCSVValue(fields, jump_volatility_col, filename);
        params.push_back(row);
    }
}

void attachReturnsByKey(const vector<MarketPoint>& market, const vector<double>& returns,
                        const vector<string>& dates, const vector<int>& hours,
                        vector<ParameterRow>& params) {
    unordered_map<string, double> keyed;
    for (size_t i = 1; i < market.size(); ++i)
        keyed[market[i].date + ":" + to_string(market[i].hour)] = returns[i - 1];
    for (size_t i = 0; i < params.size(); ++i) {
        const string key = dates[i] + ":" + to_string(hours[i]);
        const auto it = keyed.find(key);
        // The first market observation has no preceding price and therefore no return.
        params[i].observed_return =
            it == keyed.end() ? numeric_limits<double>::quiet_NaN() : it->second;
    }
}

void sortChronologically(vector<string>& dates, vector<int>& hours,
                         vector<ParameterRow>& params) {
    vector<size_t> order(params.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return dates[a] == dates[b] ? hours[a] < hours[b] : dates[a] < dates[b];
    });
    vector<string> sorted_dates; vector<int> sorted_hours; vector<ParameterRow> sorted_params;
    for (size_t i : order) { sorted_dates.push_back(dates[i]); sorted_hours.push_back(hours[i]); sorted_params.push_back(params[i]); }
    dates.swap(sorted_dates); hours.swap(sorted_hours); params.swap(sorted_params);
}

void readParams(const string& filename, vector<string>& p_dates,
                vector<ParameterRow>& p_params) {
    ifstream file(filename);
    if (!file)
        throw runtime_error("Could not open parameter file: " + filename);

    string line;
    getline(file, line);
    const vector<string> header = splitCSVLine(line);
    const size_t mu_col = requiredColumn(header, {"mu_mean"}, filename);
    const size_t kappa_col = requiredColumn(header, {"kappa_mean"}, filename);
    const size_t theta_col = requiredColumn(header, {"theta_mean"}, filename);
    const size_t volvol_col = requiredColumn(header, {"vol-of-vol_mean"}, filename);
    const size_t rho_col = requiredColumn(header, {"rho_mean"}, filename);
    const size_t variance_col = requiredColumn(header, {"vproxy", "v_t_mean"}, filename);
    const size_t jump_intensity_col = requiredColumn(header, {"jump_intensity_mean"}, filename);
    const size_t jump_mean_col = requiredColumn(header, {"jump_mean_mean"}, filename);
    const size_t jump_volatility_col = requiredColumn(
        header, {"jump_volatility_mean"}, filename);

    while (getline(file, line)) {
        vector<string> fields = splitCSVLine(line);
        if (fields.size() < header.size() + 1)
            throw runtime_error("Malformed baseline parameter row in " + filename);

        string date = toISODate(fields[0] + "," + fields[1]);

        ParameterRow row;
        row.mu = readGeneratedCSVValue(fields, mu_col, filename);
        row.kappa = readGeneratedCSVValue(fields, kappa_col, filename);
        row.theta = readGeneratedCSVValue(fields, theta_col, filename);
        row.vol_of_vol = readGeneratedCSVValue(fields, volvol_col, filename);
        row.rho = readGeneratedCSVValue(fields, rho_col, filename);
        row.variance = readGeneratedCSVValue(fields, variance_col, filename);
        row.jump_intensity = readGeneratedCSVValue(fields, jump_intensity_col, filename);
        row.jump_mean = readGeneratedCSVValue(fields, jump_mean_col, filename);
        row.jump_volatility = readGeneratedCSVValue(fields, jump_volatility_col, filename);

        p_dates.push_back(date);
        p_params.push_back(row);
    }
}

void attachHourlyReturns(const vector<MarketPoint>& market_points,
                         const vector<double>& returns,
                         vector<string>& p_dates,
                         vector<int>& p_hours,
                         vector<ParameterRow>& p_params) {
    if (p_params.empty())
        return;

    if (market_points.size() < p_params.size() + 1)
        throw runtime_error("Not enough market path rows to align hourly parameter rows");

    const size_t first_market_idx = market_points.size() - p_params.size();

    for (size_t row_idx = 0; row_idx < p_params.size(); row_idx++) {
        const size_t market_idx = first_market_idx + row_idx;
        if (market_idx == 0)
            throw runtime_error("Cannot attach a return for the first market path row");
        if (p_dates[row_idx] != market_points[market_idx].date) {
            throw runtime_error("Calibration date does not match aligned hourly market row");
        }

        p_hours.push_back(market_points[market_idx].hour);
        p_params[row_idx].observed_return = returns[market_idx - 1];
    }
}

void writeFile(const string& filename, const string& variance_header,
               const vector<string>& p_dates, const vector<int>& p_hours,
               const vector<ParameterRow>& p_params) {
    ofstream out(filename);
    if (!out)
        throw runtime_error("Could not open output file: " + filename);

    out << "date,hour,mu_mean,kappa_mean,theta_mean,vol-of-vol_mean,rho_mean,"
        << variance_header
        << ",jump_intensity_mean,jump_mean_mean,jump_volatility_mean,return,one_step_nll\n";
    out << setprecision(10);

    for (size_t row_idx = 0; row_idx < p_params.size(); row_idx++) {
        const auto& row = p_params[row_idx];
        out << p_dates[row_idx] << ',' << p_hours[row_idx]
            << ',' << row.mu << ',' << row.kappa << ',' << row.theta
            << ',' << row.vol_of_vol << ',' << row.rho << ',' << row.variance
            << ',' << row.jump_intensity << ',' << row.jump_mean
            << ',' << row.jump_volatility << ',';
        if (row_idx + 1 < p_params.size()
            && isfinite(p_params[row_idx + 1].observed_return)) {
            out << p_params[row_idx + 1].observed_return;
        }
        out << ',';
        if (isfinite(row.one_step_nll)) out << row.one_step_nll;
        out << '\n';
    }
}

double batesNLL(double observed_return, const ParameterRow& row) {
    const double eps = 1e-12;
    const double variance = max(row.variance, eps);
    const double jump_intensity = max(row.jump_intensity, 0.0);
    const double jump_variance = max(row.jump_volatility * row.jump_volatility, 0.0);
    const double jump_compensator = expm1(row.jump_mean + 0.5 * jump_variance);
    const double base_mean =
        (row.mu - 0.5 * variance - jump_intensity * jump_compensator) * HOURLY_DT;
    const double base_variance = max(variance * HOURLY_DT, eps);
    const double poisson_mean = jump_intensity * HOURLY_DT;

    vector<double> log_terms;
    log_terms.reserve(MAX_JUMP_COUNT + 1);
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

        const double conditional_mean = base_mean + jump_count * row.jump_mean;
        const double conditional_variance =
            max(base_variance + jump_count * jump_variance, eps);
        const double residual = observed_return - conditional_mean;
        const double log_normal_density = -0.5 * (
            log(2.0 * M_PI * conditional_variance)
            + residual * residual / conditional_variance);
        log_terms.push_back(log_poisson_weight + log_normal_density);
    }

    const double max_log_term = *max_element(log_terms.begin(), log_terms.end());
    double scaled_sum = 0.0;
    for (double term : log_terms) scaled_sum += exp(term - max_log_term);
    return -(max_log_term + log(scaled_sum));
}

void oneStepNLL(vector<ParameterRow>& p_params) {
    if (p_params.size() < 2)
        return;

    for (size_t i = 0; i < p_params.size() - 1; i++) {
        const double next_return = p_params[i + 1].observed_return;
        if (isfinite(next_return)) p_params[i].one_step_nll = batesNLL(next_return, p_params[i]);
    }
}

Summary summarize(const string& method, const vector<ParameterRow>& rows) {
    vector<double> values;
    for (const auto& row : rows)
        if (isfinite(row.one_step_nll)) values.push_back(row.one_step_nll);
    Summary s{method, rows.size(), values.size(), numeric_limits<double>::quiet_NaN(),
              numeric_limits<double>::quiet_NaN(), numeric_limits<double>::quiet_NaN()};
    if (values.empty()) return s;
    s.total_nll = 0.0;
    for (double value : values) s.total_nll += value;
    s.mean_nll = s.total_nll / values.size();
    sort(values.begin(), values.end());
    const size_t m = values.size() / 2;
    s.median_nll = values.size() % 2 ? values[m] : 0.5 * (values[m - 1] + values[m]);
    return s;
}

void writeSummary(const fs::path& filename, const vector<Summary>& summaries) {
    ofstream out(filename);
    if (!out) throw runtime_error("Could not write " + filename.string());
    out << "method,parameter_rows,scored_rows,mean_one_step_nll,median_one_step_nll,total_one_step_nll\n";
    out << setprecision(10);
    for (const auto& s : summaries)
        out << s.method << ',' << s.parameter_rows << ',' << s.scored_rows << ','
            << s.mean_nll << ',' << s.median_nll << ',' << s.total_nll << '\n';
}

void runJoint(const fs::path& input, const fs::path& output, bool eape,
              const vector<MarketPoint>& market, const vector<double>& returns,
              const string& name, vector<Summary>& summaries) {
    vector<string> dates; vector<int> hours; vector<ParameterRow> params;
    readJointParams(input.string(), eape, dates, hours, params);
    attachReturnsByKey(market, returns, dates, hours, params);
    sortChronologically(dates, hours, params);
    oneStepNLL(params);
    writeFile(output.string(), "v_t", dates, hours, params);
    summaries.push_back(summarize(name, params));
}

int main() {
    const fs::path root = fs::current_path();
    const fs::path baseline_path = root / "Output/Baseline_Crypto";
    const fs::path result_path = root / "Results/Real_Crypto/P_errors";
    fs::create_directories(result_path);
    const string s_path = (root / "Data/Real_Crypto/S_path.csv").string();

    vector<MarketPoint> bates_market_points = readMarketPath(s_path);
    vector<double> bates_returns = getReturns(bates_market_points);
    vector<Summary> summaries;

    vector<string> bates_garch_dates;
    vector<int> bates_garch_hours;
    vector<ParameterRow> bates_garch_p_params;
    string bates_garch_params = (baseline_path / "garch_calibration_params.csv").string();
    readParams(bates_garch_params, bates_garch_dates, bates_garch_p_params);
    attachHourlyReturns(bates_market_points, bates_returns, bates_garch_dates,
                        bates_garch_hours, bates_garch_p_params);
    oneStepNLL(bates_garch_p_params);
    writeFile((result_path / "garch_mcmc_nll.csv").string(), "vproxy", bates_garch_dates,
              bates_garch_hours, bates_garch_p_params);
    summaries.push_back(summarize("garch_mcmc", bates_garch_p_params));

    vector<string> bates_pmcmc_dates;
    vector<int> bates_pmcmc_hours;
    vector<ParameterRow> bates_pmcmc_p_params;
    string bates_pmcmc_params = (baseline_path / "pmcmc_calibration_params.csv").string();
    readParams(bates_pmcmc_params, bates_pmcmc_dates, bates_pmcmc_p_params);
    attachHourlyReturns(bates_market_points, bates_returns, bates_pmcmc_dates,
                        bates_pmcmc_hours, bates_pmcmc_p_params);
    oneStepNLL(bates_pmcmc_p_params);
    writeFile((result_path / "pmcmc_nll.csv").string(), "v_t_mean", bates_pmcmc_dates,
              bates_pmcmc_hours, bates_pmcmc_p_params);
    summaries.push_back(summarize("pmcmc", bates_pmcmc_p_params));

    runJoint(root / "Output/Real_Crypto/EAPE/parameters.csv", result_path / "joint_eape_nll.csv",
             true, bates_market_points, bates_returns, "joint_eape", summaries);
    runJoint(root / "Output/Real_Crypto/MAPE/parameters.csv", result_path / "joint_mape_nll.csv",
             false, bates_market_points, bates_returns, "joint_mape", summaries);
    writeSummary(result_path / "summary.csv", summaries);
    cout << "Wrote P-space results to " << result_path << '\n';

    const fs::path lms_result = root / "Results/Real_Crypto_LMS/P_errors";
    fs::create_directories(lms_result);
    vector<Summary> lms_summaries;
    writeFile((lms_result / "garch_mcmc_nll.csv").string(), "vproxy", bates_garch_dates,
              bates_garch_hours, bates_garch_p_params);
    lms_summaries.push_back(summarize("garch_mcmc", bates_garch_p_params));
    writeFile((lms_result / "pmcmc_nll.csv").string(), "v_t_mean", bates_pmcmc_dates,
              bates_pmcmc_hours, bates_pmcmc_p_params);
    lms_summaries.push_back(summarize("pmcmc", bates_pmcmc_p_params));
    runJoint(root / "Output/Real_Crypto_LMS/EAPE/parameters.csv", lms_result / "joint_eape_nll.csv",
             true, bates_market_points, bates_returns, "joint_eape", lms_summaries);
    runJoint(root / "Output/Real_Crypto_LMS/MAPE/parameters.csv", lms_result / "joint_mape_nll.csv",
             false, bates_market_points, bates_returns, "joint_mape", lms_summaries);
    writeSummary(lms_result / "summary.csv", lms_summaries);
    cout << "Wrote P-space results to " << lms_result << '\n';

    return 0;
}
