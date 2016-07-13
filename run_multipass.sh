#! /bin/bash

NUM_JOBS=4
NUM_SLICES=32

if [ $# != 3 ] ; then
  echo "Expected 3 arguments: gmm_file slice_dir output_dir" >&2
  exit 1
fi

export GMMFILE=$1
export SLICEDIR=$2
export SUBDIR=$3

make multipass
mkdir -p $SUBDIR

date > $SUBDIR/time  # start time

run_slice() {
  date > "$SUBDIR/time_$1$2"  # start time
  if [ $1 = "v" ] ; then  # validation
    TRAINFILE="$SLICEDIR/s$2_vtrain.csv"
    TESTFILE="$SLICEDIR/s$2_vtest.csv"
    SUBPREFIX="$SUBDIR/s$2_valid_"
  else  # testing
    TRAINFILE="$SLICEDIR/s$2_train.csv"
    TESTFILE="$SLICEDIR/s$2_test.csv"
    SUBPREFIX="$SUBDIR/s$2_sub_"
  fi
  INFOFILE="$SUBDIR/s$2_info.csv"
  ./multipass $TRAINFILE $GMMFILE $TESTFILE $INFOFILE $SUBPREFIX \
    >"$SUBDIR/log_$1$2" 2>&1
  date >> "$SUBDIR/time_$1$2"  # stop time
}
export -f run_slice

parallel -j$NUM_JOBS run_slice ::: v t ::: `seq 0 $((NUM_SLICES-1))`

date >> $SUBDIR/time  # stop time
