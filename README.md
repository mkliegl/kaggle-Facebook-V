# 2nd place solution to *Facebook V* competition on Kaggle

This is the repository for my work on the
[Facebook V: Predicting Check Ins][kaggle] competition on Kaggle.

An overview of the ideas and some discussion may be found in
[this thread][forum] in the Kaggle forum.


## Simplified single-pass online learning version

This simplified version would have scored 0.60456 (14th place). It runs
in around 90 minutes on my laptop.

As it goes through the test data, it line-by-line makes a prediction
and then learns from it.

Instructions:

1. Download the `test.csv` and `train.csv` files from [Kaggle][kaggle].
2. Run `python sort_by_time.py test.csv test_sorted.csv` to sort the test
   data by time.
3. `make CXX=<your C++ compiler>`
4. `time ./singlepass train.csv test_sorted.csv sub_singlepass.csv`

**Note**: With `learn_from_predictions` set to `false`, this runs in around 50
minutes on my laptop and would have scored 0.58973 (86th place).


## Final submission

The main differences to the simplified version are:
1. Spatial histogram probabilities are averaged with probabilities from GMM's
   fit using scikit-learn.
2. Do 20 iterations of learning from predictions.

I still need to clean up this code. I will try to post it soon...


## License

MIT. See the [LICENSE](LICENSE) file for more details.


[kaggle]: https://www.kaggle.com/c/facebook-v-predicting-check-ins
[forum]: https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/22078/solution-sharing/126235#post126235
