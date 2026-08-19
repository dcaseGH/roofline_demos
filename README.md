Toy code to measure roofline - use intel to measure

eg 1 (simpler.cpp)  - mem bound
g++ -g -O3 -mavx2 -mfma -fopenmp -std=c++17 simpler.cpp -o simpler
source /opt/intel/oneapi/advisor/latest/advisor-vars.sh 
advisor --collect=survey --project-dir=./advisor_roofline -- ./simpler
advisor --collect=tripcounts --flop --project-dir=./advisor_roofline -- ./simpler
advisor-gui ./advisor_roofline
