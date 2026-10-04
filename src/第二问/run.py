"""Q2 final Q80 model. python run.py --help"""
import sys,os
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from q2.final_pipeline import main
if __name__=='__main__':main()
