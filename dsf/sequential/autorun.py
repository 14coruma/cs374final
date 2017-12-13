#!/usr/bin/env python
import csv, subprocess

parameter_file_full_path = "./job_params.csv"

with open(parameter_file_full_path, "rb") as csvfile:
    reader = csv.reader(csvfile)
    for job in reader:
        command = """mpirun -np 1 ./dsf-seq {0} {1} 1""".format(*job)

        exit_status = subprocess.call(command, shell=True)
        if exit_status is 1:  # Check to make sure the job submitted
            print "Job {0}x{1} failed to submit".format(command)
print "Done submitting jobs!"
