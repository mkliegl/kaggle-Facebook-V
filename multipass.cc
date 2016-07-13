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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>    // std::move
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
constexpr int kernel_x_half_width = 2;  // how many bins to left and right (or
constexpr int kernel_y_half_width = 2;  // top and bottom) to evaluate kernel at
constexpr double x_std = 0.01;
constexpr double y_std = 0.005;

// Trend (i.e., relative popularity over time of places)
constexpr int trend_bins = 50;
constexpr double trend_scale = 0.05;  // scale for exponential kernel

// Semi-supervised learning
constexpr int K = 25;  // number of candidates to consider in passes after first

constexpr bool learn_from_predictions = true;
constexpr int passes = 20;  // number of predict/learn iterations

constexpr int learn_K = 15;  // the top how many predictions to learn from
static_assert(learn_K <= K, "learn_K cannot be larger than K");

constexpr double trend_learn_weight = 1.25;
constexpr double histogram_learn_rate = 0.05;

// For the first pass, we use
//   P(place_id) = (number of checkins for place_id)^{checkins_power}
//
// The idea behind raising to a power <1 is to make the algorithm
// less confident in its initial guess based on the training data. We prefer
// it to learn P(place_id) from the test data instead.
constexpr double checkins_power = 0.3;


//
// Constants
//

constexpr double pi = 3.14159265358979323846;
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
// GMM
//

namespace GMM {

struct GmmComponent {
    double weight;
    double x_mean;
    double y_mean;
    double x_std;
    double y_std;

    double pdf(double x, double y) const {
        double zx = (x - x_mean) / x_std;
        double zy = (y - y_mean) / y_std;
        return (weight * std::exp(-0.5 * (zx*zx + zy*zy))) /
            (2.0 * pi * x_std * y_std);
    }
};

class GmmDist {
public:
    void add_component(double weight, double x_mean, double y_mean,
                       double x_var, double y_var) {
        double x_std = std::sqrt(x_var);
        double y_std = std::sqrt(y_var);
        auto component = GmmComponent{weight, x_mean, y_mean, x_std, y_std};
        components.push_back(component);
    }

    double get_score(double x, double y) const {
        double score = 0.0;
        for (auto const& component : components) {
            score += component.pdf(x, y);
        }
        return score;
    }

private:
    std::vector<GmmComponent> components;
};

class GMM {
public:
    void load_from_file(std::string filename) {
        std::ifstream infile;
        std::string line;
        double weight, x_mean, y_mean, x_var, y_var;
        int n_components, n_points;
        std::uint64_t place_id;
        char c;

        std::cerr << "Loading GMM info from " << filename << std::endl;

        infile.open(filename);
        if (!infile.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::getline(infile, line);  // skip header
        std::uint64_t count = 0;
        std::istringstream iss;
        while (std::getline(infile, line)) {
            count++;
            iss.clear();
            iss.str(line);
            iss >> place_id >> c >> n_points >> c >> n_components;
            num_points[place_id] = static_cast<double>(n_points);
            if (n_components == 0) continue;
            auto& dist = distributions[place_id];
            for (int i = 0; i < n_components; ++i) {
                iss >> c >> weight >> c >> x_mean >> c >> y_mean
                    >> c >> x_var >> c >> y_var;
                dist.add_component(weight, x_mean, y_mean, x_var, y_var);
            }
        }
    }

    double get_score(std::uint64_t place_id, double x, double y,
                     std::uint64_t accuracy) {
        if (num_points[place_id] < min_checkins) {
            return 0.0;
        } else {
            return distributions[place_id].get_score(x, y);
        }
    }

private:
    static constexpr int min_checkins = 10;  // return score 0 if fewer checkins
    std::unordered_map<std::uint64_t, GmmDist> distributions;
    std::unordered_map<std::uint64_t, double> num_points;
};

}  // namespace GMM


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

    int nth_correct(std::uint64_t actual, int learn_K) const {
        for (int i = 0; i < learn_K; i++) {
            if (actual == ids[K - 1 - i]) {
                return i;
            }
        }
        return learn_K;  // if none of 0..(learn_K - 1) were correct
    }

    std::uint64_t get_id(int i) const {
        return ids[K - 1 - i];
    }

    const std::vector<std::uint64_t> get_ids() const {
        return ids;
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
// LearnWeights: In validation, we keep track of the empirical probabilities of
// the predictions 1..learn_K being correct in each semi-supervised pass.
//

class LearnWeights {
public:
    explicit LearnWeights(int learn_K)
    : learn_K(learn_K)
    , learn_weights(learn_K + 1, 0.0)
    , total(0.0)
    {}

    void add_count(int num) {
        learn_weights[num] += 1.0;
        total += 1.0;
    }

    void normalize() {
        for (int i = 0; i <= learn_K; i++) {
            learn_weights[i] /= total;
        }
        total = 1.0;
    }

    double get_weight(int i) {
        return learn_weights[i];
    }

    void read(std::istringstream& iss) {
        char c;
        double learn_weight;

        for (int i = 0; i <= learn_K; i++) {
            iss >> c >> learn_weight;
            learn_weights[i] = learn_weight;
        }
    }

    void write(std::ostream& ofs) const {
        for (double weight : learn_weights) {
            ofs << ',' << weight;
        }
        ofs << std::endl;
    }

private:
    int learn_K;
    std::vector<double> learn_weights;
    double total;
};


//
// Model
//

class Model {
public:
    void train_on_file(const std::string& filename,
                       const std::string& gmm_filename);

    void predict_on_file(const std::string& filename,
                         const std::string& valfilename,
                         const std::string& subfilename_prefix);

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
                        std::uint64_t time, bool first_pass = true,
                        std::uint64_t count = 0);

    std::uint64_t max_train_time = 0;
    place_dict<double> num_checkins;
    place_dict<PeriodicHist<time_of_day_bins, minutes_per_day>>
        time_of_day_hist;
    place_dict<PeriodicHist<time_of_day_bins_2, minutes_per_day>>
        time_of_day_hist_2;
    place_dict<PeriodicHist<day_of_week_bins, minutes_per_week>>
        day_of_week_hist;
    SpaceHist space_hist;
    GMM::GMM gmm;
    place_dict<AccuracyHist<accuracy_bins>> accuracy_hist;
    place_dict<TrendHist<trend_bins, max_test_time>>
        trend_hist, trend_hist_train, trend_hist_new;
    std::vector<std::vector<std::uint64_t>> semi_candidates;
    LearnWeights lw{learn_K}, lw_new{learn_K};
};

void Model::process(double x, double y, std::uint64_t accuracy,
                    std::uint64_t time, std::uint64_t place_id,
                    double weight, bool training) {
    time_of_day_hist[place_id].add_time(time, weight);
    time_of_day_hist_2[place_id].add_time(time, weight);
    day_of_week_hist[place_id].add_time(time, weight);
    trend_hist_train[place_id].add_time(time, weight);

    if (training) {
        num_checkins[place_id] += weight;
        max_train_time = std::max(time, max_train_time);
    }

    // for space_hist and accuracy hist, weight later training data more,
    // but stop increasing the weights for test data
    double dtime = static_cast<double>(training ? time : max_train_time);
    double time_weight = std::exp(4.0 * dtime / max_train_time);

    space_hist.add_point(place_id, x, y, time_weight * weight);
    accuracy_hist[place_id].add(std::log10(accuracy), time_weight * weight);
}

double Model::detrended_score(std::uint64_t id, double x, double y,
                              std::uint64_t accuracy, std::uint64_t time) {
    double score = 1.0;

    // space
    score *= std::sqrt(gmm.get_score(id, x, y, accuracy));
    score *= std::sqrt(space_hist.get_score(id, x, y, accuracy, true));

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
                           std::uint64_t time, bool first_pass,
                           std::uint64_t count) {
    Predictions pred(K);
    double score, dt_score;

    if (first_pass) {
        auto candidates = space_hist.get_candidates(x, y);

        for (auto const id : candidates) {
            dt_score = detrended_score(id, x, y, accuracy, time);
            score = std::pow(num_checkins[id], checkins_power) * dt_score;
            pred.maybe_insert(score, id);
        }
        semi_candidates.push_back(pred.get_ids());
    } else {
        for (auto const id : semi_candidates[count]) {
            dt_score = detrended_score(id, x, y, accuracy, time);
            score = trend_hist[id].get_score(time, false) * dt_score;
            pred.maybe_insert(score, id);
        }
    }

    return pred;
}

void Model::train_on_file(const std::string& filename,
                          const std::string& gmm_filename) {
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

    gmm.load_from_file(gmm_filename);

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
                            const std::string& valfilename,
                            const std::string& subfilename_prefix) {
    std::ifstream infile;
    std::fstream valfile;
    std::ofstream subfile;
    std::string line;
    double x, y;
    std::uint64_t row_id, accuracy, time, place_id;
    char c;
    Predictions pred(K);
    bool validate;
    std::istringstream iss;

    auto start_time = steady_clock::now();
    auto last_time = start_time;
    auto now = start_time;

    std::cout << "Making predictions for " << filename << std::endl;

    // check if test file has a place_id column to determine whether
    // we are in validation or test mode
    infile.open(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::getline(infile, line);
    if (line == "row_id,x,y,accuracy,time") {
        validate = false;
    } else if (line == "row_id,x,y,accuracy,time,place_id") {
        validate = true;
    } else {
        std::cerr << "Invalid test file." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    infile.close();

    // open file for reading/writing validation score & learn weights
    auto valmode = validate ? std::fstream::out : std::fstream::in;
    valfile.open(valfilename, valmode);
    if (!valfile.is_open()) {
        std::cerr << "Failed to open file: " << valfilename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (validate) {
        // print header
        valfile << "pass,score";
        for (int i = 0; i <= learn_K; i++) {
            valfile << ",weight_" << i;
        }
        valfile << std::endl;
    } else {
        std::getline(valfile, line);  // skip header
    }

    for (int pass = 1; pass <= passes; pass++) {
        std::cout << "Pass " << pass << std::endl;
        trend_hist_new = trend_hist_train;

        // open test file (file to make predictions on)
        infile.open(filename);
        std::getline(infile, line);  // skip header

        // open file for writing submission
        auto subfilename = subfilename_prefix + std::to_string(pass) + ".csv";
        subfile.open(subfilename);
        if (!subfile.is_open()) {
            std::cerr << "Failed to open file: " << subfilename << std::endl;
            std::exit(EXIT_FAILURE);
        }
        subfile << "row_id,place_id" << std::endl;  // print header

        if (validate) {
            // in validation mode, we learn the learn weights
            lw_new = LearnWeights(learn_K);
        } else {
            // in test mode, we read the learn weights found during validation
            // from the validation info file
            std::string line;
            std::getline(valfile, line);
            std::istringstream iss(line);
            int pass;
            char c;
            double val_score;

            iss >> pass >> c >> val_score;
            lw.read(iss);
        }

        std::uint64_t count = 0;
        double total_ap = 0.0;

        while (std::getline(infile, line)) {
            count++;

            // read line and make predictions
            iss.clear();
            iss.str(line);
            iss >> row_id >> c >> x >> c >> y >> c >> accuracy >> c >> time;
            bool first_pass = (pass == 1);  // on first pass, don't use trends
            pred = predict(x, y, accuracy, time, first_pass, count - 1);

            // store predictions
            if (validate) {
                iss >> c >> place_id;
                total_ap += pred.ap(place_id);
                lw_new.add_count(pred.nth_correct(place_id, learn_K));
            }
            pred.print(row_id, subfile);

            // learn from predictions
            if (learn_from_predictions) {
                double top_score = pred.get_score(0);
                for (int i = 0; i < learn_K; i++) {
                    double learn_weight = 0.5 * pred.get_score(i) / top_score;
                    if (!(validate && (pass == 1))) {
                        // empirical probabilities not available in first
                        // validation pass
                        learn_weight =
                            std::sqrt(learn_weight * lw.get_weight(i));
                    }
                    std::uint64_t id = pred.get_id(i);
                    trend_hist_new[id].add_time(
                        time, trend_learn_weight * learn_weight);
                    process(x, y, accuracy, time, id,
                            /* weight = */ histogram_learn_rate * learn_weight,
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

        // close all files
        infile.close();
        subfile.close();

        // output validation score
        if (validate) {
            std::cout << "Validation score:\t" << total_ap / count << std::endl;
        }

        // use trends we learned for next pass
        trend_hist = std::move(trend_hist_new);

        if (validate) {
            // use learn_weights from this pass for next pass
            lw_new.normalize();
            lw = std::move(lw_new);

            // save validation score and learn weights
            valfile << pass << ',' << total_ap / count;
            lw.write(valfile);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Expected five arguments: train_filename gmm_filename"
                     " test_filename validation_info_filename"
                     " submission_filename_prefix" << std::endl;
        return EXIT_FAILURE;
    }

    Model model;
    model.train_on_file(argv[1], argv[2]);
    model.predict_on_file(argv[3], argv[4], argv[5]);

    return 0;
}
