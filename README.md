# 2nd place solution to *Facebook V* competition on Kaggle

This is the repository for my work on the
[Facebook V: Predicting Check Ins][kaggle] competition on Kaggle.

An overview of the ideas and some discussion may be found in
[this thread][forum] in the Kaggle forum.

Overview of the performance of the solutions below:

Version | Score | Position
--------|-------|---------
*[Tom Van de Wiele's 1st place solution][ttvand]* | 0.62279 | 1
*my actual final submission* | 0.62170 | 2
multipass, GMM | 0.62036 | 2
*[Jack (Japan)'s 3rd place solution][jack]* | 0.61394 | 3
single pass, learn also on test, GMM | 0.61047 | 6
single pass, learn also on test | 0.60456 | 13
single pass, learn only on train | 0.58973 | 85

(The above leaderboard positions are where I would have finished if
 I hadn't submitted my actual final submission.)


## Version 1: Single-pass online learning version

This simplified version would have scored 0.60456. It runs
in around 90 minutes on my laptop. With `learn_from_predictions` set
to `false`, this runs in around 50 minutes and would have
scored 0.58973.

As it goes through the test data, it line-by-line makes a prediction
and then learns from it. If one first sorts the test data by time, then
this version never uses information about the future.

Instructions:

1. Download the `test.csv` and `train.csv` files from [Kaggle][kaggle].
2. Run `python sort_by_time.py test.csv test_sorted.csv` to sort the test
   data by time.
3. `make CXX=<your C++ compiler>`
4. `time ./singlepass train.csv test_sorted.csv sub_singlepass.csv`


## Version 2: Adding GMM

Averaging the computation of P(x, y | place_id) in the previous version with
probabilities obtained from fitting Gaussian Mixture Models using scikit-learn
improves the score a fair amount. Indeed, this version would have
scored 0.61047.

However, this gain comes at quite a loss in speed. Fitting the GMM's for the
full train set took around 40 hours on my laptop (using 1 core).

Instructions:

1. Download the `test.csv` and `train.csv` files from [Kaggle][kaggle].
2. Run `python sort_by_time.py test.csv test_sorted.csv` to sort the test
   data by time.
3. `make CXX=<your C++ compiler>`
4. `time ./gmm.py train.csv gmm_train.csv`
   (Or to use my precomputed version, run `7z e gmm_train.7z`.)
5. `time ./singlepass_gmm train.csv gmm_train.csv test_sorted.csv sub_singlepass.csv`


## Version 3: Semi-supervised learning: multiple passes

This version is fairly close to my actual final submission. It would have
scored 0.62036.

Compared to the simplified version, the main difference is that this version
does 20 iterations of learning from predictions.
A disadvantage compared to the simplified version is that we use information
from the future (i.e., from checkins at times greater than t when making a
prediction for a checkin at time t). If the application demanded real-time
predictions, this would obviously not make sense.

Running this takes around 50-60 CPU-hours. To speed things up, I split
the train and test data into slices and ran several slices in parallel with
the help of [GNU parallel][gnu-parallel].

Instructions:

1. Download the `test.csv` and `train.csv` files from [Kaggle][kaggle].
2. `make CXX=<your C++ compiler>`
3. `time ./gmm.py train.csv gmm_train.csv`
   (Or to use my precomputed version, run `7z e gmm_train.7z`.)
4. `./create_slices.py train.csv test.csv slices`
   (This splits the train and test data into 32 slices and also creates
    validation train and test sets for each of those slices.)
5. Edit `NUM_JOBS` in `run_multipass.sh` to specify how many slices you want
   to run in parallel.
6. `./run_multipass.sh gmm_train.csv slices multipass_output`
   (All output will be in the directory `multipass_output`.
    You can look at the files `log_*` and `time_*` to keep track of progress.)
7. `./combine.sh multipass_output 20`
   (This will combine the pass 20
    submission files for each slice into a single submission file
    `multipass_output/sub_20.csv`.)

**Note:** I did not train separate GMM's for the validation slices, so
technically there is some leakage there. Also, they use only the last 10 weeks
for testing to speed things up. The explanation is that the purpose of these
validation slices was not for testing new ideas (for that I used
a different, much smaller set). Rather, the purpose is as follows:

1. Monitor whether the convergence is stable over the 20 iterations.
   (Ideally, it should be monotonically increasing. At least, the score
    at pass 20 should be pretty close to the maximum score over all passes.)
2. Empirically learn the probabilities of predictions 1..learn_K being
   correct in a given pass. These are used to adjust the learn weights
   when learning from the actual test data.


## License

MIT. See the [LICENSE](LICENSE) file for more details.


[kaggle]: https://www.kaggle.com/c/facebook-v-predicting-check-ins
[forum]: https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/22078/solution-sharing/126235#post126235
[ttvand]: https://ttvand.github.io/Winning-approach-of-the-Facebook-V-Kaggle-competition/and/Facebook-V
[jack]: https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/22078/solution-sharing/126290#post126290
[gnu-parallel]: http://www.gnu.org/software/parallel/
