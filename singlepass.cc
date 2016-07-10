#include <algorithm>  // std::min, std::max
#include <array>
#include <chrono>
#include <cmath>      // std::exp, std::sqrt, etc.
#include <cstdint>    // std::uint64_t
#include <cstdlib>    // std::exit, EXIT_FAILURE
#include <fstream>
#include <iostream>
#include <limits>     // negative infinity
#include <map>
#include <memory>     // std::make_unique
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::chrono::steady_clock;
using std::chrono::duration;

template<typename T>
using place_dict = std::unordered_map<std::uint64_t, T>;


//
// Parameters
//

// Time of day
// (We average two time of day histograms with different numbers of bins.)
constexpr int time_of_day_bins = 12;
constexpr int time_of_day_bins_2 = 18;

// Day of week
constexpr int day_of_week_bins = 7;

// Accuracy
constexpr int accuracy_bins = 100;
constexpr double accuracy_std = 0.035;  // std for Gaussian kernel

// Space
constexpr int x_bins = 900;
constexpr int y_bins = 1350;

// Space kernel
constexpr int kernel_x_half_width = 1;  // how many bins to left and right (or
constexpr int kernel_y_half_width = 1;  // top and bottom) to evaluate kernel at
constexpr double x_std = 0.01;
constexpr double y_std = 0.005;

// Trend (i.e., relative popularity over time of places)
constexpr int trend_bins = 50;
constexpr double trend_scale = 0.05;  // scale for exponential kernel

// Online learning
constexpr bool learn_from_predictions = true;
constexpr int learn_K = 15;  // the top how many predictions to learn from
constexpr double learn_rate = 0.6;


//
// Constants
//

constexpr double negative_infinity = -std::numeric_limits<double>::infinity();

constexpr std::uint64_t minutes_per_day = 24 * 60;
constexpr std::uint64_t minutes_per_week = 7 * minutes_per_day;

// max time in test.csv is 1006589 ~ a little less than 100 weeks
constexpr std::uint64_t max_test_time = 100 * minutes_per_week;

constexpr double grid_dim = 10.0 + 1e-15;     // (x,y) in [0, 10] x [0, 10]
constexpr double max_accuracy = 3.0 + 1e-15;  // log10 accuracy in [1, 3]


//
// Periodic histogram
//

template <int num_bins, std::uint64_t period>
class PeriodicHist {
public:
    void add_time(std::uint64_t time, double weight = 1.0) {
        int bin = get_bin(time);
        double scaled_time = get_scaled_time(time);
        double rem = scaled_time - bin;
        if (rem >= 0.0) {
            hist[bin] += weight * (1.0 - rem);
            hist[(bin + 1) % num_bins] += weight * rem;
        } else {
            hist[bin] += weight * (1.0 + rem);
            hist[(bin + num_bins - 1) % num_bins] -= weight * rem;
        }
        sum += weight;
    }

    double get_score(std::uint64_t time, bool normalized = true) const {
        double score;
        int bin = get_bin(time);
        double scaled_time = get_scaled_time(time);
        double rem = scaled_time - bin;
        if (rem >= 0.0) {
            score = (1.0 - rem) * hist[bin] + rem * hist[(bin + 1) % num_bins];
        } else {
            score = (1.0 + rem) * hist[bin]
                    - rem * hist[(bin + num_bins - 1) % num_bins];
        }
        if (normalized) {
            score /= sum;
        }
        return score;
    }

private:
    static int get_bin(std::uint64_t time) {
        return (time % period) / (period / num_bins);
    }

    // if time falls into bin I, then scaled_time falls into
    //   [I - 0.5, I + 0.5)
    static double get_scaled_time(std::uint64_t time) {
        return static_cast<double>(time % period) / (period / num_bins) - 0.5;
    }

    std::array<double, num_bins> hist{};
    double sum = 0.0;
};


//
// Space histogram
//

class SpaceHist {
public:
    void add_point(std::uint64_t place_id, double x, double y, double weight) {
        int xb = x_bin(x);
        int yb = y_bin(y);
        int xb_min = std::max(0, xb - kernel_x_half_width);
        int xb_max = std::min(x_bins - 1, xb + kernel_x_half_width);
        int yb_min = std::max(0, yb - kernel_y_half_width);
        int yb_max = std::min(y_bins - 1, yb + kernel_y_half_width);
        double kern[1 + 2 * kernel_x_half_width][1 + 2 * kernel_y_half_width];
        double sum = 0.0, tmp;

        for (int xbi = xb_min; xbi <= xb_max; xbi++) {
            double xbc = x_bin_center(xbi);
            for (int ybi = yb_min; ybi <= yb_max; ybi++) {
                double ybc = y_bin_center(ybi);
                tmp = kernel(x, y, xbc, ybc);
                kern[xbi - xb_min][ybi - yb_min] = tmp;
                sum += tmp;
            }
        }

        for (int xbi = xb_min; xbi <= xb_max; xbi++) {
            for (int ybi = yb_min; ybi <= yb_max; ybi++) {
                candidates[idx(xbi, ybi)].insert(place_id);
                hist[idx(xbi, ybi)][place_id] +=
                    weight * kern[xbi - xb_min][ybi - yb_min] / sum;
            }
        }

        hist_sum[place_id] += weight;
    }

    double get_score(std::uint64_t place_id, double x, double y,
                     std::uint64_t accuracy, bool normalized = true) {
        double score = 0.0;
        int xb = x_bin(x);
        int yb = y_bin(y);
        int xb_min = std::max(0, xb - kernel_x_half_width);
        int xb_max = std::min(x_bins - 1, xb + kernel_x_half_width);
        int yb_min = std::max(0, yb - kernel_y_half_width);
        int yb_max = std::min(y_bins - 1, yb + kernel_y_half_width);

        for (int xbi = xb_min; xbi <= xb_max; xbi++) {
            double xbc = x_bin_center(xbi);
            for (int ybi = yb_min; ybi <= yb_max; ybi++) {
                double ybc = y_bin_center(ybi);
                score +=
                    kernel(x, y, xbc, ybc) * hist[idx(xbi, ybi)][place_id];
            }
        }

        if (normalized) {
            score /= hist_sum[place_id];
        }

        return score;
    }

    const std::unordered_set<std::uint64_t>
    get_candidates(double x, double y) const {
        return candidates[idx(x, y)];
    }

private:
    // We model the uncertainty in the x, y coordinates using a
    // Gaussian distribution. (Note: We do not need it to be normalized,
    // so we save ourselves the trouble of dividing by various constants.)
    static double kernel(double x, double y, double xbc, double ybc) {
        double zx = (x - xbc) / x_std;
        double zy = (y - ybc) / y_std;
        return std::exp(-(zx * zx + zy * zy) / 2.0);
    }

    static int x_bin(double x) {
        return std::floor(x * x_bins / grid_dim);
    }

    static int y_bin(double y) {
        return std::floor(y * y_bins / grid_dim);
    }

    static double x_bin_center(int xb) {
        return (xb + 0.5) * grid_dim / x_bins;
    }

    static double y_bin_center(int yb) {
        return (yb + 0.5) * grid_dim / y_bins;
    }

    static int idx(int xb, int yb) {
        return xb * y_bins + yb;
    }

    static int idx(double x, double y) {
        return idx(x_bin(x), y_bin(y));
    }

    std::vector<place_dict<double>> hist =
        std::vector<place_dict<double>>(x_bins * y_bins);
    place_dict<double> hist_sum;
    std::vector<std::unordered_set<std::uint64_t>> candidates =
        std::vector<std::unordered_set<std::uint64_t>>(x_bins * y_bins);
};


//
// Trend histogram
//

template <int num_bins, std::uint64_t max_time>
class TrendHist {
public:
    void add_time(std::uint64_t time, double weight = 1.0) {
        int bin = get_bin(time);
        double scaled_time = get_scaled_time(time);
        double rem = scaled_time - bin;
        if (rem >= 0.0) {
            hist[bin] += weight * (1.0 - rem);
            hist[bin + 1] += weight * rem;
        } else {
            hist[bin] += weight * (1.0 + rem);
            hist[bin - 1] -= weight * rem;
        }
        sum += weight;
    }

    double get_score(std::uint64_t time, bool normalized = true) const {
        double scaled_time = get_scaled_time(time);
        double score = 0.0;
        for (int bin = 0; bin < num_bins + 2; bin++) {
            double dist =
                std::abs(scaled_time - (bin + 0.5)) / num_bins;  // in [0, 1)
            score += hist[bin] * std::exp(-dist / trend_scale);
        }
        if (normalized) {
            score /= sum;
        }
        return score;
    }

private:
    static int get_bin(std::uint64_t time) {
        return 1 + time / (max_time / num_bins);
    }

    // if time falls into bin I, then scaled_time falls into
    //   [I - 0.5, I + 0.5)
    static double get_scaled_time(std::uint64_t time) {
        return 1.0 + static_cast<double>(time) / (max_time / num_bins) - 0.5;
    }

    std::array<double, num_bins + 2> hist{};  // add extra bin at each end
    double sum = 0.0;
};


//
// Accuracy histogram
//

template <int num_bins>
class AccuracyHist {
public:
    void add(double val, double weight = 1.0) {
        int bin = get_bin(val);
        double scaled_val = get_scaled_val(val);
        double rem = scaled_val - bin;
        if (rem >= 0.0) {
            hist[bin] += weight * (1.0 - rem);
            hist[bin + 1] += weight * rem;
        } else {
            hist[bin] += weight * (1.0 + rem);
            hist[bin - 1] -= weight * rem;
        }
        sum += weight;
    }

    double get_score(double val, bool normalized = true) const {
        double scaled_val = get_scaled_val(val);
        double score = 0.0;
        for (int bin = 0; bin < num_bins + 2; bin++) {
            double dist =
                std::abs(scaled_val - (bin + 0.5)) / num_bins;  // in [0, 1)
            double z = dist / accuracy_std;
            score += hist[bin] * std::exp(-z*z/2.0);
        }
        if (normalized) {
            score /= sum;
        }
        return score;
    }

private:
    std::array<double, num_bins + 2> hist{};  // add extra bin at each end
    double sum = 0.0;

    static int get_bin(double val) {
        return 1 + static_cast<int>(val / (max_accuracy / num_bins));
    }

    // if accuracy falls into bin I, then scaled_accuracy falls into
    //   [I - 0.5, I + 0.5)
    static double get_scaled_val(double val) {
        return 1.0 + val / (max_accuracy / num_bins) - 0.5;
    }
};


//
// Predictions
//

class Predictions {
public:
    explicit Predictions(int K)
    : K(K)
    , ids(K, 0)
    , scores(K, negative_infinity)
    {}

    // maintain K best scores: scores[K-1] > scores[K-2] > ... > scores[0]
    void maybe_insert(double score, std::uint64_t id) {
        int i = -1;
        while ((i < K - 1) && (score > scores[i + 1])) {
            i++;
        }
        if (i >= 0) {
            for (int j = 0; j < i; j++) {
                scores[j] = scores[j + 1];
                ids[j] = ids[j + 1];
            }
            scores[i] = score;
            ids[i] = id;
        }
    }

    double ap(std::uint64_t actual) const {
        if (actual == ids[K - 1]) {
            return 1.0;
        } else if (actual == ids[K - 2]) {
            return 0.5;
        } else if (actual == ids[K - 3]) {
            return 1.0 / 3.0;
        } else {
            return 0.0;
        }
    }

    std::uint64_t get_id(int i) const {
        return ids[K - 1 - i];
    }

    double get_score(int i) const {
        return scores[K - 1 - i];
    }

    void print(std::uint64_t row_id, std::ostream& ofs) const {
        ofs << row_id << ',' << ids[K - 1] << ' ' << ids[K - 2] << ' '
            << ids[K - 3] << std::endl;
    }

private:
    int K;
    std::vector<std::uint64_t> ids;
    std::vector<double> scores;
};


//
// Global variables
//

class Model {
public:
    void train_on_file(const std::string& filename);

    void predict_on_file(const std::string& filename,
                         const std::string& subfilename);

private:
    // learn from a labeled checkin
    void process(double x, double y, std::uint64_t accuracy,
                 std::uint64_t time, std::uint64_t place_id,
                 double weight = 1.0, bool training = true);

    // detrended_score computes the product of:
    // - P(x , y | place_id)
    // - P(time of day | place_id)
    // - P(day of week | place_id)
    // - P(accuracy | place_id)
    double detrended_score(std::uint64_t id, double x, double y,
                           std::uint64_t accuracy, std::uint64_t time);

    Predictions predict(double x, double y, std::uint64_t accuracy,
                        std::uint64_t time);

    std::uint64_t max_train_time = 0;
    place_dict<PeriodicHist<time_of_day_bins, minutes_per_day>>
        time_of_day_hist;
    place_dict<PeriodicHist<time_of_day_bins_2, minutes_per_day>>
        time_of_day_hist_2;
    place_dict<PeriodicHist<day_of_week_bins, minutes_per_week>>
        day_of_week_hist;
    SpaceHist space_hist;
    place_dict<AccuracyHist<accuracy_bins>> accuracy_hist;
    place_dict<TrendHist<trend_bins, max_test_time>> trend_hist;
};

void Model::process(double x, double y, std::uint64_t accuracy,
                    std::uint64_t time, std::uint64_t place_id,
                    double weight, bool training) {
    time_of_day_hist[place_id].add_time(time, weight);
    time_of_day_hist_2[place_id].add_time(time, weight);
    day_of_week_hist[place_id].add_time(time, weight);
    trend_hist[place_id].add_time(time, weight);

    if (training) {
        max_train_time = std::max(time, max_train_time);
    }
    double dtime = static_cast<double>(training ? time : max_train_time);

    // for space_hist and accuracy hist, weight later training data more,
    // but stop increasing the weights for test data
    double time_weight = std::exp(4.0 * dtime / max_train_time);

    space_hist.add_point(place_id, x, y, time_weight * weight);
    accuracy_hist[place_id].add(std::log10(accuracy), time_weight * weight);
}

double Model::detrended_score(std::uint64_t id, double x, double y,
                              std::uint64_t accuracy, std::uint64_t time) {
    double score = 1.0;

    // space
    score *= space_hist.get_score(id, x, y, accuracy, true);

    // time-of-day
    score *= std::sqrt(time_of_day_hist[id].get_score(time, true));
    score *= std::sqrt(time_of_day_hist_2[id].get_score(time, true));

    // day-of-week
    score *= day_of_week_hist[id].get_score(time, true);

    // accuracy
    double log_acc = std::log10(accuracy);
    score *= accuracy_hist[id].get_score(log_acc, true);

    return score;
}

Predictions Model::predict(double x, double y, std::uint64_t accuracy,
                           std::uint64_t time) {
    Predictions pred(learn_K);

    auto candidates = space_hist.get_candidates(x, y);

    for (auto const id : candidates) {
        double dt_score = detrended_score(id, x, y, accuracy, time);
        double score = trend_hist[id].get_score(time, false) * dt_score;
        pred.maybe_insert(score, id);
    }

    return pred;
}

void Model::train_on_file(const std::string& filename) {
    std::ifstream infile;
    std::string line;
    double x, y;
    std::uint64_t row_id, accuracy, time, place_id;
    char c;
    std::uint64_t count = 0;
    std::istringstream iss;

    auto start_time = steady_clock::now();
    auto last_time = start_time;
    auto now = start_time;

    std::cout << "Training on " << filename << std::endl;

    infile.open(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::getline(infile, line);
    if (line != "row_id,x,y,accuracy,time,place_id") {
        std::cerr << "Invalid train file." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    while (std::getline(infile, line)) {
        count++;
        iss.clear();
        iss.str(line);
        iss >> row_id >> c >> x >> c >> y >> c >> accuracy >> c >> time >>
            c >> place_id;
        process(x, y, accuracy, time, place_id);
        if (count % 1000000 == 0) {
            now = steady_clock::now();
            std::cout
                << count << '\t'
                << duration<double>(now - last_time).count() << 's' << '\t'
                << duration<double>(now - start_time).count()/60 << 'm'
                << std::endl;
            last_time = now;
        }
    }
}

void Model::predict_on_file(const std::string& filename,
                            const std::string& subfilename) {
    std::ifstream infile;
    std::ofstream subfile;
    std::string line;
    double x, y;
    std::uint64_t row_id, accuracy, time, place_id;
    char c;
    Predictions pred(learn_K);
    double ap;
    bool validate;
    std::uint64_t count = 0;
    double total_ap = 0.0;
    std::map<double, double> ap_hist;
    std::istringstream iss;

    auto start_time = steady_clock::now();
    auto last_time = start_time;
    auto now = start_time;

    std::cout << "Making predictions for " << filename << std::endl;

    // open file for writing submission
    subfile.open(subfilename);
    if (!subfile.is_open()) {
        std::cerr << "Failed to open file: " << subfilename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    subfile << "row_id,place_id" << std::endl;  // print header

    // open file to make predictions on
    infile.open(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // check if test file has a place_id column for validation
    std::getline(infile, line);
    if (line == "row_id,x,y,accuracy,time") {
        validate = false;
    } else if (line == "row_id,x,y,accuracy,time,place_id") {
        validate = true;
    } else {
        std::cerr << "Invalid test file." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    while (std::getline(infile, line)) {
        count++;

        // read line and make predictions
        iss.clear();
        iss.str(line);
        iss >> row_id >> c >> x >> c >> y >> c >> accuracy >> c >> time;
        pred = predict(x, y, accuracy, time);

        // store predictions
        if (validate) {
            iss >> c >> place_id;
            ap = pred.ap(place_id);
            ap_hist[ap]++;
            total_ap += ap;
        }
        pred.print(row_id, subfile);

        // try to learn trends from predictions
        if (learn_from_predictions) {
            double top_score = pred.get_score(0);
            for (int i = 0; i < learn_K; i++) {
                double learn_weight = 0.5 * pred.get_score(i) / top_score;
                std::uint64_t id = pred.get_id(i);
                trend_hist[id].add_time(time, learn_rate * learn_weight);
                process(x, y, accuracy, time, id,
                        /* weight = */ learn_rate * learn_weight,
                        /* training = */ false);
            }
        }

        // output some info on progress
        if (count % 100000 == 0) {
            now = steady_clock::now();
            std::cout
                << count << '\t'
                << duration<double>(now - last_time).count() << 's' << '\t'
                << duration<double>(now - start_time).count()/60 << 'm';
            if (validate) {
                std::cout << '\t' << total_ap / count;
            }
            std::cout << std::endl;
            last_time = now;
        }
    }

    // output validation score
    if (validate) {
        std::cout << "Validation score:\t" << total_ap / count << '\t';
        for (auto const& kv : ap_hist) {
            std::cout << kv.second / count << ' ';
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Expected three arguments: train_filename test_filename"
                     " submission_filename" << std::endl;
        return EXIT_FAILURE;
    }

    Model model;
    model.train_on_file(argv[1]);
    model.predict_on_file(argv[2], argv[3]);

    return 0;
}
