from common import *
import bridge
from bridge import dispatch
from planner import plan
from accounting import components
import datetime
EXP=Path(os.environ.get('Q3_OUTPUT_DIR',str(PACKAGE/'outputs'))).resolve()
ROOT=PACKAGE
FROZEN=EXP/'frozen'
BASE=PACKAGE
def date(day):return str(DATES[day])
