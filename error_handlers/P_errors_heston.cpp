#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct ParameterRow {
    std::string date;
    double mu;
    double kappa;
    double theta;
    double vol_of_vol;
    double rho;
    double variance;
    double observed_return;
    double one_step_nll = std::numeric_limits<double>::quiet_NaN();
};

struct Summary {
    std::string method;
    std::size_t parameter_rows = 0;
    std::size_t scored_rows = 0;
    double mean_nll = std::numeric_limits<double>::quiet_NaN();
    double median_nll = std::numeric_limits<double>::quiet_NaN();
    double total_nll = std::numeric_limits<double>::quiet_NaN();
};

enum class Method { GarchMcmc, Pmcmc, JointEape, JointMape };

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

std::string longDateToISO(const std::string& input) {
    std::stringstream ss(input);
    std::string month, day, year;
    ss >> month >> day >> year;

    while (!day.empty() && !std::isdigit(static_cast<unsigned char>(day.back()))) {
        day.pop_back();
    }
    while (!year.empty() && !std::isdigit(static_cast<unsigned char>(year.back()))) {
        year.pop_back();
    }

    static const std::unordered_map<std::string, int> months = {
        {"January", 1}, {"February", 2}, {"March", 3}, {"April", 4},
        {"May", 5}, {"June", 6}, {"July", 7}, {"August", 8},
        {"September", 9}, {"October", 10}, {"November", 11}, {"December", 12}
    };
    const auto month_it = months.find(month);
    if (month_it == months.end() || day.empty() || year.empty()) {
        throw std::runtime_error("Invalid calibration date: " + input);
    }

    std::ostringstream out;
    out << year << '-' << std::setw(2) << std::setfill('0') << month_it->second
        << '-' << std::setw(2) << std::setfill('0') << std::stoi(day);
    return out.str();
}

std::string marketDateToISO(const std::string& input) {
    std::stringstream ss(input);
    std::string day, month, year;
    std::getline(ss, day, '-');
    std::getline(ss, month, '-');
    std::getline(ss, year, '-');

    static const std::unordered_map<std::string, int> months = {
        {"Jan", 1}, {"Feb", 2}, {"Mar", 3}, {"Apr", 4},
        {"May", 5}, {"Jun", 6}, {"Jul", 7}, {"Aug", 8},
        {"Sep", 9}, {"Oct", 10}, {"Nov", 11}, {"Dec", 12}
    };
    const auto month_it = months.find(month);
    if (month_it == months.end() || day.empty() || year.empty()) {
        throw std::runtime_error("Invalid market date: " + input);
    }

    std::ostringstream out;
    out << year << '-' << std::setw(2) << std::setfill('0') << month_it->second
        << '-' << std::setw(2) << std::setfill('0') << std::stoi(day);
    return out.str();
}

fs::path findRepoRoot() {
    fs::path current = fs::current_path();
    while (true) {
        if (fs::exists(current / "Data/Real_Stock/S_path.csv") &&
            fs::exists(current / "Output")) {
            return current;
        }
        if (current == current.root_path()) break;
        current = current.parent_path();
    }
    throw std::runtime_error(
        "Could not find the repository root. Run this program from anywhere inside "
        "Joint_Calibration_Framework_Options."
    );
}

std::unordered_map<std::string, double> readLogReturns(const fs::path& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open " + filename.string());

    std::string line;
    std::getline(file, line);
    std::vector<std::string> dates;
    std::vector<double> prices;
    std::size_t line_number = 1;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() < 2) {
            throw std::runtime_error(filename.string() + ": malformed row " +
                                     std::to_string(line_number));
        }
        dates.push_back(marketDateToISO(fields[0]));
        prices.push_back(std::stod(fields[1]));
    }

    std::unordered_map<std::string, double> returns;
    for (std::size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1] <= 0.0 || prices[i] <= 0.0) {
            throw std::runtime_error("Stock prices must be positive to compute log returns");
        }
        returns[dates[i]] = std::log(prices[i] / prices[i - 1]);
    }
    return returns;
}

ParameterRow parseParameterRow(const std::vector<std::string>& f, Method method,
                               const std::string& filename, std::size_t line_number) {
    const std::size_t required =
        method == Method::GarchMcmc ? 13 :
        method == Method::Pmcmc ? 14 :
        method == Method::JointEape ? 16 : 9;
    if (f.size() < required) {
        throw std::runtime_error(filename + ": expected at least " +
                                 std::to_string(required) + " fields on row " +
                                 std::to_string(line_number) + ", got " +
                                 std::to_string(f.size()));
    }

    ParameterRow row;
    row.date = longDateToISO(f[0] + "," + f[1]);
    if (method == Method::GarchMcmc || method == Method::Pmcmc) {
        row.mu = std::stod(f[2]);
        row.kappa = std::stod(f[4]);
        row.theta = std::stod(f[6]);
        row.vol_of_vol = std::stod(f[8]);
        row.rho = std::stod(f[10]);
        row.variance = std::stod(f[12]);
    } else if (method == Method::JointEape) {
        row.mu = std::stod(f[2]);
        row.vol_of_vol = std::stod(f[4]);
        row.kappa = std::stod(f[6]);
        row.theta = std::stod(f[8]);
        row.rho = std::stod(f[10]);
        row.variance = std::stod(f[14]);
    } else {
        row.mu = std::stod(f[2]);
        row.vol_of_vol = std::stod(f[3]);
        row.kappa = std::stod(f[4]);
        row.theta = std::stod(f[5]);
        row.rho = std::stod(f[6]);
        row.variance = std::stod(f[8]);
    }
    return row;
}

std::vector<ParameterRow> readParameters(
    const fs::path& filename, Method method,
    const std::unordered_map<std::string, double>& returns) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open " + filename.string());

    std::vector<ParameterRow> rows;
    std::string line;
    std::getline(file, line);
    std::size_t line_number = 1;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;
        ParameterRow row = parseParameterRow(splitCsv(line), method, filename.string(), line_number);
        const auto return_it = returns.find(row.date);
        if (return_it == returns.end()) {
            throw std::runtime_error(filename.string() + ": no market return for date " + row.date);
        }
        row.observed_return = return_it->second;
        rows.push_back(row);
    }
    if (rows.empty()) throw std::runtime_error("No parameter rows read from " + filename.string());
    return rows;
}

double nll(double observed_return, double mu, double variance) {
    constexpr double dt = 1.0 / 252.0;
    constexpr double eps = 1e-12;
    variance = std::max(variance, eps);
    const double conditional_mean = (mu - 0.5 * variance) * dt;
    const double conditional_variance = variance * dt;
    const double residual = observed_return - conditional_mean;
    return 0.5 * std::log(2.0 * std::acos(-1.0) * conditional_variance)
         + 0.5 * residual * residual / conditional_variance;
}

void scoreOneStepAhead(std::vector<ParameterRow>& rows) {
    for (std::size_t i = 0; i + 1 < rows.size(); ++i) {
        rows[i].one_step_nll = nll(rows[i + 1].observed_return, rows[i].mu, rows[i].variance);
    }
}

void writeRows(const fs::path& filename, const std::vector<ParameterRow>& rows) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Could not write " + filename.string());
    out << "date,mu,kappa,theta,vol-of-vol,rho,variance,return,one_step_nll\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.date << ',' << row.mu << ',' << row.kappa << ',' << row.theta
            << ',' << row.vol_of_vol << ',' << row.rho << ',' << row.variance
            << ',' << row.observed_return << ',';
        if (std::isfinite(row.one_step_nll)) out << row.one_step_nll;
        out << '\n';
    }
}

Summary summarize(const std::string& method, const std::vector<ParameterRow>& rows) {
    std::vector<double> values;
    for (const auto& row : rows) {
        if (std::isfinite(row.one_step_nll)) values.push_back(row.one_step_nll);
    }
    Summary summary;
    summary.method = method;
    summary.parameter_rows = rows.size();
    summary.scored_rows = values.size();
    if (values.empty()) return summary;

    summary.total_nll = 0.0;
    for (double value : values) summary.total_nll += value;
    summary.mean_nll = summary.total_nll / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    summary.median_nll = values.size() % 2 == 0
        ? 0.5 * (values[middle - 1] + values[middle])
        : values[middle];
    return summary;
}

void writeSummary(const fs::path& filename, const std::vector<Summary>& summaries) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Could not write " + filename.string());
    out << "method,parameter_rows,scored_rows,mean_one_step_nll,median_one_step_nll,total_one_step_nll\n";
    out << std::setprecision(10);
    for (const auto& summary : summaries) {
        out << summary.method << ',' << summary.parameter_rows << ',' << summary.scored_rows << ',';
        if (std::isfinite(summary.mean_nll)) out << summary.mean_nll;
        out << ',';
        if (std::isfinite(summary.median_nll)) out << summary.median_nll;
        out << ',';
        if (std::isfinite(summary.total_nll)) out << summary.total_nll;
        out << '\n';
    }
}

void runAnalysis(const fs::path& market_path, const fs::path& baseline_output,
                 const fs::path& joint_output,
                 const fs::path& result_dir) {
        fs::create_directories(result_dir);
        const auto returns = readLogReturns(market_path);
        struct Job {
            std::string name;
            fs::path input;
            Method method;
        };
        const std::vector<Job> jobs = {
            {"garch_mcmc", baseline_output / "garch_calibration_params.csv", Method::GarchMcmc},
            {"pmcmc", baseline_output / "pmcmc_calibration_params.csv", Method::Pmcmc},
            {"joint_eape", joint_output / "EAPE/parameters.csv", Method::JointEape},
            {"joint_mape", joint_output / "MAPE/parameters.csv", Method::JointMape}
        };

        std::vector<Summary> summaries;
        for (const auto& job : jobs) {
            auto rows = readParameters(job.input, job.method, returns);
            scoreOneStepAhead(rows);
            writeRows(result_dir / (job.name + "_nll.csv"), rows);
            summaries.push_back(summarize(job.name, rows));
        }
        writeSummary(result_dir / "summary.csv", summaries);
        std::cout << "Wrote P-space results to " << result_dir << '\n';
}

int main() {
    try {
        const fs::path root = findRepoRoot();
        runAnalysis(root / "Data/Real_Stock/S_path.csv",
                    root / "Output/Baseline_Stock",
                    root / "Output/Real_Stock",
                    root / "Results/Real_Stock/P_errors");
        runAnalysis(root / "Data/Real_Stock/S_path.csv",
                    root / "Output/Baseline_Stock",
                    root / "Output/Real_Stock_LMS",
                    root / "Results/Real_Stock_LMS/P_errors");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P_errors: " << error.what() << '\n';
        return 1;
    }
}
