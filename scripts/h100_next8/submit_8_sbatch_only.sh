sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=B200_NG scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=B200_BY scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=CPU100_NG scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=CPU100_BY scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=HOTPOT_NG scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=HOTPOT_BY scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=Q1000_NG scripts/h100_next8/run_h100_next8_cell.sbatch
sbatch --dependency=singleton --job-name=kvq650-10batch --export=ALL,CASE_ID=Q1000_BY scripts/h100_next8/run_h100_next8_cell.sbatch
