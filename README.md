# 2nd place solution to *Facebook V* competition on Kaggle

This is the repository for my work on the
[Facebook V: Predicting Check Ins][kaggle] competition on Kaggle.

An overview of the ideas and some discussion may be found in
[this thread][forum] in the Kaggle forum.


## Simplified single-pass online learning version

This simplified version would have scored 0.60456 (14th place). It runs
in around 90 minutes on my laptop. With `learn_from_predictions` set
to `false`, this runs in around 50 minutes and would have
scored 0.58973 (86th place).

As it goes through the test data, it line-by-line makes a prediction
and then learns from it. If one first sorts the test data by time, then
this version never uses information about the future.

Instructions:

1. Download the `test.csv` and `train.csv` files from [Kaggle][kaggle].
2. Run `python sort_by_time.py test.csv test_sorted.csv` to sort the test
   data by time.
3. `make CXX=<your C++ compiler>`
4. `time ./singlepass train.csv test_sorted.csv sub_singlepass.csv`


## Adding GMM

Averaging the computation of $P(x, y | place_id)$ in the previous version with
probabilities obtained from fitting Gaussian Mixture Models using scikit-learn
improves the score a fair amount. Indeed, this version would have
scored 0.61047 (7th place).

However, this gain comes at quite a loss in speed. Fitting the GMM's for the
full train set took over 20 hours on my laptop.

Instructions:

1. Download the `test.csv` and `train.csv` files from [Kaggle][kaggle].
2. Run `python sort_by_time.py test.csv test_sorted.csv` to sort the test
   data by time.
3. `make CXX=<your C++ compiler>`
4. `time ./gmm.py train.csv gmm_train.csv` (*Warning:* This takes long!)
5. `time ./singlepass_gmm train.csv gmm_train.csv test_sorted.csv sub_singlepass.csv`


## Semi-supervised learning: multiple passes

Compared to the simplified version, the main difference is that this version
does 20 iterations of predicting and then learning from the predictions.
A disadvantage compared to the simplified version is that we use information
from checkins at times greater than $t$ when making a prediction for a checkin
at time $t$. If the application demanded real-time predictions, this would
obviously not make sense.

I still need to clean up this code a bit. I will post it soon...


## License

MIT. See the [LICENSE](LICENSE) file for more details.


[kaggle]: https://www.kaggle.com/c/facebook-v-predicting-check-ins
[forum]: https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/22078/solution-sharing/126235#post126235
