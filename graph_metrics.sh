#!/usr/bin/env bash

# In order for this script to work, the MQLib project must be cloned & complied locally on a Unix OS !
# ----------------------------- #
# @ First argument: Provide the path for the MQLib executable, which typically should be under:
# /home/user/projecct/MQLib/bin/MQLib
# @ Second argument: path to the text file which contains the Max-Cut graph instance in the appropriate format
# See the expected format here: [COMPLETE THIS]
# ------------------------------ #
# OUTPUT:
# The script writes a temporary csv file with all metrics of the given graph-instance.
# returns the path of the output file (to stdout) for python to utilize and for the subsequent script for permanent deletion. 
# This scrip should be used only from within python run in order to take advantage of MQLib metrics calculations over an  nx.Graph/Graph.txt external instance
# ./bin/MQLib -m -mh -fM /home/ronl/Q_Opt/max_cut/datasets/gset/instance/G23/G23.txt> test.csv

set -euo pipefail

echo "Extracting metrics using MQLib-CPP executable (bash).." >&2



# checking correct amount of arguments:
if [ $# -lt 3 ]; then
    echo "Usage: $0 ::  <MQLib_Exec_Path> <Graph_File_Path> <TMP_Out_Path>" >&2
    exit 1
fi


# grbbing input/output paths:
MQLib_Exec_Path="$1"
Graph_File_Path="$2"
TMP_OUT_PATH="$3" 



# validating graph file:
if [ ! -f "$Graph_File_Path" ]; then
    echo "Error: could not locate graph file under: $Graph_File_Path" >&2
    exit 1
fi

if [[ ! "$Graph_File_Path" == *.txt ]]; then
    echo "Error: Expecting graph in a text format only: $Graph_File_Path" >&2
    exit 1
fi


 # validating excutable file:
if [ ! -f "$MQLib_Exec_Path" ]; then
    echo "Error: could not locate execuable file under: $MQLib_Exec_Path" >&2
    exit 1
fi

if [ ! -x "$MQLib_Exec_Path" ]; then 
    echo "Error: File exists but is not executable: $MQLib_Exec_Path" >&2
    exit 1
fi


# creating output dir and changing dir:
mkdir -p "$TMP_OUT_PATH"
cd "$TMP_OUT_PATH" || {
    echo "Error: Failed to cd into $TMP_OUT_PATH" >&2
    exit 1
}


#  printing output path to user:
TEMP_FILE_NAME="$(mktemp graph_metrics_XXXXXXXXXXX.csv)"
echo "Creating temporary metrics csv file: $TMP_OUT_PATH/$TEMP_FILE_NAME" >&2


# executing MQLib metrics on graph.txt:
"$MQLib_Exec_Path" -m -mh -fM "$Graph_File_Path" >  "$TEMP_FILE_NAME" # is this the correct way to execute ?
MQLIB_STATUS=$? ## what is happening here ? who is "$?"  ?? who fills this up ? 

# Checking exit status of execution and existance of output file:
if [ $MQLIB_STATUS -ne 0 ]; then
    echo "Error: MQLib exited with status $MQLIB_STATUS" >&2
    exit 1
fi

if [ ! -s  "$TEMP_FILE_NAME" ]; then 
    echo "Error: output file is missin or empty" >&2
    echo "The output file was not created properly!: $TMP_OUT_PATH/$TEMP_FILE_NAME " >&2
    exit 1
fi

#else: run was successful 
echo "The output file was successfully created under: $TMP_OUT_PATH/$TEMP_FILE_NAME " >&2
echo "$TMP_OUT_PATH/$TEMP_FILE_NAME" #echoing the output path for python to grab (no stderr channeling)
exit 0 # successfull status 