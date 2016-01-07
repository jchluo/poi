
# Kill ALL python thread in system.
# because poi.Evaluation use multiple thread to run, 
# CTRL+C may not stop it, this script will do.
# usage: 
#     sh kill-python.sh

ps -elf | grep " python "|grep -v "grep"|while read l
do
    pid=`echo $l | cut -f 4 -d " "`
    kill -9  $pid
    echo $l
done
