

import sys
import time
from tplr import tplr

SPEC_VERSION = tplr.get_remote_spec_version() # Run version.

while True:
    new_spec = tplr.get_remote_spec_version()
    print( tplr.get_pm2_process_name(), new_spec )
    if SPEC_VERSION != new_spec:
        print ('restart')
        time.sleep(5)
        tplr.update_and_restart()
        sys.exit(0)
    time.sleep(10)
