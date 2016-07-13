#! /bin/bash

if [ $# != 2 ] ; then
  echo "Expected 2 arguments: output_directory pass_number" >&2
  exit 1
fi

SUBDIR=$1
PASS=$2
NUM_SLICES=32

echo "Combining slices for pass $PASS in directory $SUBDIR"

cd $SUBDIR
cat s0_sub_${PASS}.csv > sub_${PASS}.csv
for (( SLICE = 1; SLICE < $NUM_SLICES; SLICE++ )) ; do
  sed '1d' s${SLICE}_sub_${PASS}.csv >> sub_${PASS}.csv
done
