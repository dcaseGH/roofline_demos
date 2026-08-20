Toy code to measure roofline - use intel to measure

Build the benchmark with (default gnu):
make

Run the tests with:
make test

source /opt/intel/oneapi/advisor/latest/advisor-vars.sh 
advisor --collect=survey --project-dir=./advisor_roofline -- ./main
advisor --collect=tripcounts --flop --project-dir=./advisor_roofline -- ./main
advisor-gui ./advisor_roofline
