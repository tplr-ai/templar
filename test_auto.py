

import time
from tplr import tplr

SPEC_VERSION = 0 # Run version.

while True:
    print( tplr.get_pm2_process_name(), tplr.get_remote_spec_version() )
    time.sleep(10)
