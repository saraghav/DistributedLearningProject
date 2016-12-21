#!/bin/bash
/usr/bin/time -v ./analysis_model_averaging.py |& tee ./analysis_model_averaging.py.log
echo 'finished model averaging 2' | mail -s 'finished model averaging 2' araghav.s@gmail.com
