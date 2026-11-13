set -e

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--single-only" ]; }; then
    echo "Usage: $0 [--single-only]" >&2
    exit 1
fi

trap 'rm -f ./run_main' EXIT

TIMES_CSV="./baselines_times.csv"
INP_FILE="./inp.txt"

extract_value() {
    local key="$1"
    sed -n "s/^${key}:[[:space:]]*//p" "$INP_FILE" | head -n 1
}

folder_path=$(extract_value "folder path")
output_path=$(extract_value "Output path")

if [ ! -f "$TIMES_CSV" ]; then
    echo "run_timestamp,elapsed_seconds,folder_path,output_path" > "$TIMES_CSV"
fi

# macOS build
# g++ -std=c++17 ./Heston/baseline_real/app/main.cpp ./Heston/baseline_real/src/*.cpp \
#     -I/usr/local/include \
#     -I/opt/homebrew/include \
#     -I/opt/homebrew/include/eigen3 \
#     -I/usr/local/include/eigen3 \
#     -I./Heston/baseline_real/include \
#     -I../../../eigen-3.4.0 \
#     -L/usr/local/lib \
#     -L/opt/homebrew/lib \
#     -lQuantLib \
#     -Wl,-rpath,/usr/local/lib \
#     -Wl,-rpath,/opt/homebrew/lib \
#     -o run_main

# Linux build
g++ -std=c++17 ./Heston/baseline_real/app/main.cpp ./Heston/baseline_real/src/*.cpp \
    -I../eigen-3.4.0 \
    -I../LBFGSpp/include \
    -I$HOME/local/include \
    -I./Heston/baseline_real/include \
    -L$HOME/local/lib \
    -lQuantLib \
    -Wl,-rpath,$HOME/local/lib \
    -o run_main

start_time=$(date +%s)
run_timestamp=$(date +"%Y-%m-%d %H:%M:%S")

./run_main "$@"

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))

echo "\"$run_timestamp\",$elapsed_seconds,\"$folder_path\",\"$output_path\"" >> "$TIMES_CSV"
