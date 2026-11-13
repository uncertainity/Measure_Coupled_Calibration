set -e

trap 'rm -f ./run_main' EXIT

TIMES_CSV="./joint_calibration_times.csv"
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

# g++ -std=c++17 ./options/joint_calibration_real/app/main.cpp \
#     -I/usr/local/include \
#     -I/opt/homebrew/include \
#     -I/opt/homebrew/include/eigen3 \
#     -I/usr/local/include/eigen3 \
#     -I../../../eigen-3.4.0 \
#     -Iinclude \
#     -L/usr/local/lib \
#     -L/opt/homebrew/lib \
#     -lQuantLib \
#     -Wl,-rpath,/usr/local/lib \
#     -Wl,-rpath,/opt/homebrew/lib \
#     -o run_main

g++ -std=c++17 -pthread ./Heston/joint_calibration_real/app/main.cpp \
    -I$HOME/local-quantlib-sessions/include \
    -I$HOME/local/include \
    -I../eigen-3.4.0 \
    -I./Heston/joint_calibration_real/include \
    -L$HOME/local-quantlib-sessions/lib \
    -lQuantLib \
    -Wl,-rpath,$HOME/local-quantlib-sessions/lib \
    -o run_main

start_time=$(date +%s)
run_timestamp=$(date +"%Y-%m-%d %H:%M:%S")

./run_main

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))

echo "\"$run_timestamp\",$elapsed_seconds,\"$folder_path\",\"$output_path\"" >> "$TIMES_CSV"
