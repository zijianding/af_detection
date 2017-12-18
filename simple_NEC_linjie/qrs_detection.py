#done by a shell script, coding as follows

"""
#!/bin/bash

for line in `cat RECORDS`
do
	gqrs -r $line
	rdann -a qrs -r $line > ${line}"_qrs.txt"
done

"""