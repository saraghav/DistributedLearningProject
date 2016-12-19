#!/bin/bash
/usr/bin/time -v ./analysis_model_divergence.py |& tee ./analysis_model_divergence.py.log
echo 'finished dist simulation' | mail -s 'finished dist simulation' araghav.s@gmail.com
